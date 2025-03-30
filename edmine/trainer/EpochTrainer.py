import os
import torch
import wandb
import torch.nn as nn
from copy import deepcopy
from abc import ABC, abstractmethod

from edmine.trainer.EpochTrainRecord import TrainRecord
from edmine.trainer.utils import *
from edmine.utils.log import get_now_time


class SingleModelEpochTrainer(ABC):
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects
        self.best_model = None
        self.objects["optimizers"] = {}
        self.objects["schedulers"] = {}
        self.train_record = TrainRecord()
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
        max_epoch = trainer_config["max_epoch"]
        accumulation_step = trainer_config["accumulation_step"]
        train_loader = self.objects["data_loaders"]["train_loader"]
        optimizer = self.objects["optimizers"][model_name]
        scheduler = self.objects["schedulers"][model_name]
        model = self.objects["models"][model_name]

        for epoch in range(1, max_epoch + 1):
            model.train()
            for batch_i, batch in enumerate(train_loader):
                loss_result = model.get_predict_loss(batch)
                for loss_name, loss_info in loss_result["losses_value"].items():
                    if loss_name != "total loss":
                        self.train_record.add_loss(loss_name, loss_info["value"], loss_info["num_sample"])
                loss = loss_result["total_loss"] / accumulation_step
                loss.backward()
                if (batch_i + 1) % accumulation_step == 0:
                    if grad_clip_config["use_clip"]:
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
                    optimizer.step()
                    if hasattr(model, 'apply_clipper') and callable(getattr(model, 'apply_clipper')):
                        model.apply_clipper()
                    optimizer.zero_grad()
            if scheduler_config["use_scheduler"]:
                scheduler.step()
            self.evaluate()
            if self.stop_train():
                break

    def evaluate(self):
        trainer_config = self.params["trainer_config"]
        use_wandb = trainer_config["use_wandb"]
        model_name = trainer_config["model_name"]
        save_model = trainer_config["save_model"]
        use_multi_metrics = trainer_config["use_multi_metrics"]
        main_metric = trainer_config["main_metric"]
        mutil_metrics = trainer_config["mutil_metrics"]
        data_loaders = self.objects["data_loaders"]
        train_loader = data_loaders["train_loader"]
        valid_loader = data_loaders["valid_loader"]
        model = self.objects["models"][model_name]

        train_performance = self.evaluate_dataset(model, train_loader)
        valid_performance = self.evaluate_dataset(model, valid_loader) if valid_loader is not None else None
        self.train_record.next_epoch(train_performance, valid_performance, main_metric, use_multi_metrics, mutil_metrics)
        best_epoch = self.train_record.get_best_epoch()
        if valid_loader is not None:
            valid_performance_str = self.train_record.get_performance_str("valid")
            self.objects["logger"].info(
                f"{get_now_time()} epoch {self.train_record.get_current_epoch():<3} , valid performances are "
                f"{valid_performance_str}train loss is {self.train_record.get_loss_str()}, current best epoch is "
                f"{best_epoch}")
        else:
            self.objects["logger"].info(
                f"{get_now_time()} epoch {self.train_record.get_current_epoch():<3} , train loss is "
                f"{self.train_record.get_loss_str()}, current best epoch is {best_epoch}")
        if use_wandb:
            valid_performance = self.train_record.get_performance("valid")
            loss = self.train_record.get_loss()
            valid_performance.update(loss)
            wandb.log(valid_performance)
        self.train_record.clear_loss()

        best_epoch = self.train_record.get_best_epoch("valid")
        current_epoch = self.train_record.get_current_epoch()
        if best_epoch == current_epoch:
            # 节省显存
            self.best_model = deepcopy(model).to("cpu")
            if save_model:
                save_model_dir = self.params["save_model_dir"]
                model_weight_path = os.path.join(save_model_dir, "saved.ckt")
                torch.save({"best_valid": model.state_dict()}, model_weight_path)

    def stop_train(self):
        trainer_config = self.params["trainer_config"]
        max_epoch = trainer_config["max_epoch"]
        use_early_stop = trainer_config["use_early_stop"]
        num_epoch_early_stop = trainer_config["num_epoch_early_stop"]
        has_valid_loader = self.objects["data_loaders"]["valid_loader"] is not None

        stop_flag = self.train_record.stop_training(max_epoch, use_early_stop, num_epoch_early_stop)
        if stop_flag:
            if has_valid_loader:
                best_train_performance_by_valid = self.train_record.get_evaluate_result_str("train", "valid")
                best_valid_performance_by_valid = self.train_record.get_evaluate_result_str("valid", "valid")
                self.objects["logger"].info(
                    f"best valid epoch: {self.train_record.get_best_epoch('valid'):<3} , "
                    f"train performances in best epoch by valid are {best_train_performance_by_valid}\n"
                    f"valid performances in best epoch by valid are {best_valid_performance_by_valid}\n"
                )

        return stop_flag

    @abstractmethod
    def evaluate_dataset(self, model, data_loader):
        pass
