import os
import torch
import wandb
import json
import torch.nn as nn
from abc import ABC, abstractmethod

from edmine.trainer.EpochTrainRecord import TrainRecord
from edmine.trainer.utils import *
from edmine.utils.log import get_now_time
from edmine.metric.exercise_recommendation import *


class SingleModelStepTrainer(ABC):
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects
        self.best_model = None
        self.objects["optimizers"] = {}
        self.objects["schedulers"] = {}
        self.logs_every_step = []
        self.logs_performance = []
        self.best_valid_main_metric = -100
        self.init_trainer()

    def init_trainer(self):
        # 初始化optimizer和scheduler
        models = self.objects["models"]
        optimizers = self.objects["optimizers"]
        schedulers = self.objects["schedulers"]
        optimizers_config = self.params["optimizers_config"]
        schedulers_config = self.params["schedulers_config"]

        for model_name, optimizer_config in optimizers_config.items():
            if models[model_name] is not None:
                scheduler_config = schedulers_config[model_name]
                optimizers[model_name] = create_optimizer(models[model_name].parameters(), optimizer_config)

                if scheduler_config["use_scheduler"]:
                    schedulers[model_name] = create_scheduler(optimizers[model_name], scheduler_config)
                else:
                    schedulers[model_name] = None

    def train(self):
        trainer_config = self.params["trainer_config"]
        model_name = trainer_config["model_name"]
        grad_clip_config = self.params["grad_clips_config"][model_name]
        scheduler_config = self.params["schedulers_config"][model_name]
        train_loader = self.objects["data_loaders"]["train_loader"]
        optimizer = self.objects["optimizers"][model_name]
        scheduler = self.objects["schedulers"][model_name]
        model = self.objects["models"][model_name]

        max_step = trainer_config["max_step"]
        num_step2evaluate = trainer_config["num_step2evaluate"]
        use_early_stop = trainer_config["use_early_stop"]
        num_early_stop = trainer_config["num_early_stop"]
        use_multi_metrics = trainer_config["use_multi_metrics"]
        main_metric_key = trainer_config["main_metric"]
        multi_metrics = trainer_config["multi_metrics"]
        accumulation_step = trainer_config["accumulation_step"]
        save_model = trainer_config["save_model"]
        use_wandb = trainer_config["use_wandb"]
        log_loss_step = 100

        best_index = 0
        for step in range(1, max_step + 1):
            model.train()
            optimizer.zero_grad()

            loss_result = model.train_one_step(next(train_loader))
            loss_result["total_loss"].backward()
            if grad_clip_config["use_clip"]:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
            optimizer.step()
            if scheduler_config["use_scheduler"]:
                scheduler.step()

            loss_record = {}
            for loss_name, loss_info in loss_result["losses_value"].items():
                if loss_name != "total_loss":
                    loss_record[loss_name] = {
                        "num_sample": loss_info["num_sample"],
                        "value": loss_info["value"]
                    }
            log_step = {"loss_record": loss_record}
            self.logs_every_step.append(log_step)

            to_log_loss = (step > 0) and (step % log_loss_step == 0)
            to_evaluate = (step > 0) and (step % num_step2evaluate == 0)

            all_losses = {}
            if to_log_loss:
                logs_step = self.logs_every_step[step - log_loss_step:]
                loss_str = f""
                for loss_name in logs_step[0]["loss_record"].keys():
                    loss_value = 0
                    num_sample = 0
                    for log_one_step in logs_step:
                        loss_value += log_one_step["loss_record"][loss_name]["value"]
                        num_sample += log_one_step["loss_record"][loss_name]["num_sample"]
                    loss_value = loss_value / num_sample
                    loss_str += f"{loss_name}: {loss_value:<12.6}, "
                    all_losses[loss_name] = loss_value
                self.objects["logger"].info(f"{get_now_time()} step {step:<9}: train loss is {loss_str}")

                if use_wandb and not to_evaluate:
                    wandb.log(all_losses)

            if to_evaluate:
                model.eval()
                log_performance = {}

                valid_data_loader = self.objects["data_loaders"]["valid_loader"]
                valid_performance = self.evaluate_dataset(model, valid_data_loader)
                # todo: 这里后面要修改，通用的trainer不应该调用get_average_performance_top_ns
                average_valid_performance = get_average_performance_top_ns(valid_performance)
                if use_multi_metrics:
                    valid_main_metric = TrainRecord.cal_main_metric(average_valid_performance, multi_metrics)
                else:
                    valid_main_metric = ((-1 if main_metric_key in ["RMSE", "MAE"] else 1) *
                                         average_valid_performance[main_metric_key])
                log_performance["valid_performance"] = valid_performance
                self.logs_performance.append(log_performance)
                performance_str = self.get_performance_str(valid_performance)
                self.objects["logger"].info(f"{get_now_time()} step {step:<9}, valid performance are\n"
                                            f"main metric: {valid_main_metric:<9}\n{performance_str}")

                if use_wandb:
                    valid_performance_ = {}
                    for top_n, top_n_performance in valid_performance.items():
                        for metric_name, metric_value in top_n_performance.items():
                            valid_performance_[f"top {top_n}-{metric_name}"] = metric_value
                    valid_performance_.update(all_losses)
                    valid_performance_["main_metric"] = valid_main_metric
                    wandb.log(valid_performance_)

                if use_early_stop:
                    current_index = int(step / num_step2evaluate) - 1
                    if (valid_main_metric - self.best_valid_main_metric) >= 0.001:
                        self.best_valid_main_metric = valid_main_metric
                        best_index = current_index
                        if save_model:
                            save_model_dir = self.params["save_model_dir"]
                            model_weight_path = os.path.join(save_model_dir, "saved.ckt")
                            torch.save({"best_valid": model.state_dict()}, model_weight_path)

                    if ((current_index - best_index) >= num_early_stop) or ((max_step - step) < num_step2evaluate):
                        best_log_performance = self.logs_performance[best_index]
                        self.objects["logger"].info(
                            f"best valid step: {num_step2evaluate * (best_index + 1):<9}\n"
                            f"valid performance by best valid epoch is "
                            f"{json.dumps(best_log_performance['valid_performance'])}\n"
                        )
                        break

    @abstractmethod
    def evaluate_dataset(self, model, data_loader):
        pass

    @abstractmethod
    def get_performance_str(self, performance):
        pass
