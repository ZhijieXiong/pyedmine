import torch
import wandb
import os
import json
from torch.nn.utils import clip_grad_norm_

from edmine.trainer.EpochTrainRecord import TrainRecord
from edmine.trainer.StepTrainer import SingleModelStepTrainer
from edmine.model.module.Memory import LPRMemory
from edmine.trainer.utils import *
from edmine.utils.log import get_now_time
from edmine.metric.learning_path_recommendation import promotion_report
from edmine.metric.exercise_recommendation import get_average_performance_top_ns


def add_memories(memories, data, idx, n):
    for _ in range(n):
        if idx >= len(data):
            break
        user_data = data[idx]
        memory = LPRMemory()
        if "learning_goal" in user_data:
            memory.reset([user_data["learning_goal"]], user_data=user_data)
        else:
            memory.reset(user_data["learning_goals"], user_data=user_data)
        memories.append(memory)
        idx += 1
    return idx


class LPROnlineDRLTrainer(SingleModelStepTrainer):
    def __init__(self, params, objects):
        super().__init__(params, objects)

    def init_trainer(self):
        # 初始化optimizer和scheduler
        # 因为LPR要用到KT模型，所以这里需要区分
        models = self.objects["lpr_models"]
        optimizers = self.objects["optimizers"]
        schedulers = self.objects["schedulers"]
        optimizers_config = self.params["optimizers_config"]
        schedulers_config = self.params["schedulers_config"]

        for model_name, optimizer_config in optimizers_config.items():
            if models.get(model_name, None) is not None:
                scheduler_config = schedulers_config[model_name]
                optimizers[model_name] = create_optimizer(models[model_name].parameters(), optimizer_config)

                if scheduler_config["use_scheduler"]:
                    schedulers[model_name] = create_scheduler(optimizers[model_name], scheduler_config)
                else:
                    schedulers[model_name] = None
                    
    def remove_done_memories(self, memories, batch_observation, batch_state):
        trainer_config = self.params["trainer_config"]
        agent_name = trainer_config["agent_name"]
        master_th = trainer_config["master_threshold"]
        agent = self.objects["agents"][agent_name]
        
        done_data = []
        remain_indices = []
        for i, memory in enumerate(memories):
            done = agent.judge_done(memory, master_th)
            if done:
                done_data.append(memory.output_learning_history())
                memories[i] = None
            else:
                remain_indices.append(i)
        memories = [memories[remain_idx] for remain_idx in remain_indices]
        batch_observation = batch_observation[remain_indices]
        batch_state = batch_state[remain_indices]
        
        return memories, batch_observation, batch_state, done_data
        
    def collect_rl_data(self, user_data):
        trainer_config = self.params["trainer_config"]
        agent_name = trainer_config["agent_name"]
        master_th = trainer_config["master_threshold"]
        agent = self.objects["agents"][agent_name]
        env = self.objects["env_simulator"]
        memory = LPRMemory()
        if "learning_goal" in user_data:
            memory.reset([user_data["learning_goal"]], user_data=user_data)
        else:
            memory.reset(user_data["learning_goals"], user_data=user_data)
        
        env_input_data = {"history_data": [memory.history_data]}
        observation, batch_state = env.step(env_input_data)
        state = batch_state[0]
        memory.update_history_data(current_state=state)
        done = agent.judge_done(memory, master_th=master_th)
        while not done:
            next_rec_data = []
            rec_concept, rec_question = agent.recommend_qc(memory)
            memory.update_rec_data(int(rec_concept), int(rec_question))
            next_rec_data.append({
                "question_seq": int(rec_question),
                "correctness_seq": 0,
                "mask_seq": 1    
            })
            env_input_data = {
                "history_data": [memory.history_data],
                "next_rec_data": next_rec_data
            }
            observation, batch_state = env.step(env_input_data)
            state = batch_state[0]
            q_id = next_rec_data[0]["question_seq"]
            # 如果后面有用需要其它信息（例如时间信息）的KT模型，需要在这更改数据更新
            next_rec_result = (q_id, int(observation > 0.5))
            memory.update_history_data(current_state=state, next_rec_result=next_rec_result)
            done = agent.judge_done(memory, master_th=master_th)
        return agent.done_data2rl_data([memory.output_learning_history()])
        
    def train(self):
        trainer_config = self.params["trainer_config"]
        max_step = trainer_config["max_step"]
        num_step2evaluate = trainer_config["num_step2evaluate"]
        use_early_stop = trainer_config["use_early_stop"]
        num_early_stop = trainer_config["num_early_stop"]
        agent_name = trainer_config["agent_name"]
        accumulation_step = trainer_config["accumulation_step"]
        save_model = trainer_config["save_model"]
        use_wandb = trainer_config["use_wandb"]
        use_multi_metrics = trainer_config["use_multi_metrics"]
        main_metric = trainer_config["main_metric"]
        multi_metrics = trainer_config["multi_metrics"]
        interval_execute_config = trainer_config.get("interval_execute_config", None)
        grad_clips_config = self.params["grad_clips_config"]
        schedulers_config = self.params["schedulers_config"]
        optimizers = self.objects["optimizers"]
        schedulers = self.objects["schedulers"]
        env = self.objects["env_simulator"]
        agent = self.objects["agents"][agent_name]
        models = self.objects["lpr_models"]
        
        train_data = self.objects["data"]["train"]
        valid_data = self.objects["data"]["valid"]
        log_loss_step = 100
        idx = 0
        num_continue = 0
        self.objects["logger"].info(f"{get_now_time()} start training")
        for step_ in range(1, max_step + 1):
            # 因为后面要根据step判断是否log和evaluate
            step = step_ - num_continue
            if idx >= len(train_data):
                idx = 0
            user_data = train_data[idx]
            idx += 1
            agent.eval()
            rl_data = self.collect_rl_data(user_data)
            agent.train()
            if len(rl_data) == 0:
                num_continue += 1
                continue
            batch = agent.data2batch(rl_data)
            all_loss = agent.get_all_loss(batch)
            for model_name, loss_result in all_loss.items():
                model = models[model_name]
                grad_clip_config = grad_clips_config[model_name]
                optimizer = optimizers[model_name]
                loss = loss_result["total_loss"]
                loss.backward()
                if grad_clip_config["use_clip"]:
                    clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
                optimizer.step()
                optimizer.zero_grad()
            for model_name in all_loss.keys():
                scheduler_config = schedulers_config[model_name]
                scheduler = schedulers[model_name]
                if scheduler_config["use_scheduler"]:
                    scheduler.step()
                    
            loss_record = {}
            for loss_result in all_loss.values():
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

                valid_performance = self.evaluate_dataset(env, agent, valid_data)
                # todo: 这里后面要修改，通用的trainer不应该调用get_average_performance_top_ns
                average_valid_performance = get_average_performance_top_ns(valid_performance)
                if use_multi_metrics:
                    valid_main_metric = TrainRecord.cal_main_metric(average_valid_performance, multi_metrics)
                else:
                    valid_main_metric = ((-1 if main_metric in ["RMSE", "MAE"] else 1) *
                                         average_valid_performance[main_metric])
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
                            torch.save({
                                model_name: model.state_dict() for model_name, model in models.items()
                            }, model_weight_path)

                    if ((current_index - best_index) >= num_early_stop) or ((max_step - step) < num_step2evaluate):
                        best_log_performance = self.logs_performance[best_index]
                        self.objects["logger"].info(
                            f"best valid step: {num_step2evaluate * (best_index + 1):<9}\n"
                            f"valid performance by best valid epoch is "
                            f"{json.dumps(best_log_performance['valid_performance'])}\n"
                        )
                        break
                    
            # 自由配置每隔多少个step，做什么操作
            # step_action_config: dict[int, list[func]]
            if interval_execute_config is not None:
                for interval, target_funcs in interval_execute_config.items():
                    to_execute_func = (step > 1) and ((step - 1) % interval == 0)
                    if to_execute_func:
                        for target_func in target_funcs:
                            self.objects["logger"].info(f"{get_now_time()} step {step:<9}: execute {target_func.__name__}")
                            target_func()

    def evaluate_dataset(self, env, agent, kt_data):
        trainer_config = self.params["trainer_config"]
        steps = trainer_config["target_steps"]
        kt_data_start_idx = 0
        kt_batch_size = 64

        all_done_data = []
        memories = []
        kt_data_start_idx = add_memories(memories, kt_data, kt_data_start_idx, kt_batch_size)
        env_input_data = {"history_data": [memory.history_data for memory in memories]}
        batch_observation, batch_state = env.step(env_input_data)
        for memory, state in zip(memories, batch_state):
            memory.update_history_data(current_state=state)
        memories, batch_observation, batch_state, done_data = self.remove_done_memories(
            memories, batch_observation, batch_state
        )
        all_done_data.extend(done_data)
        while len(memories) > 0:
            # 推荐习题
            next_rec_data = []
            for memory, observation, state in zip(memories, batch_observation, batch_state):
                rec_concept, rec_question = agent.recommend_qc(memory)
                memory.update_rec_data(int(rec_concept), int(rec_question))
                next_rec_data.append({
                    "question_seq": int(rec_question),
                    "correctness_seq": 0,
                    "mask_seq": 1
                })
            env_input_data = {
                "history_data": [memory.history_data for memory in memories],
                "next_rec_data": next_rec_data
            }
            batch_observation, batch_state = env.step(env_input_data)
            for i, (memory, observation, state) in enumerate(zip(memories, batch_observation, batch_state)):
                q_id = next_rec_data[i]["question_seq"]
                # 如果后面有用需要其它信息（例如时间信息）的KT模型，需要在这更改数据更新
                next_rec_result = (q_id, int(observation > 0.5))
                memory.update_history_data(current_state=state, next_rec_result=next_rec_result)
            memories, batch_observation, batch_state, done_data = \
                self.remove_done_memories(memories, batch_observation, batch_state)
            all_done_data.extend(done_data)
            # 添加KT数据用于模拟（要获取新加入数据的state）
            if (len(memories) < kt_batch_size) and (kt_data_start_idx < len(kt_data)):
                n = len(memories)
                kt_data_start_idx = add_memories(memories, kt_data, kt_data_start_idx, kt_batch_size - n)
                env_input_data = {"history_data": [memory.history_data for memory in memories]}
                batch_observation, batch_state = env.step(env_input_data)
                for memory, state in zip(memories[n:], batch_state[n:]):
                    memory.update_history_data(current_state=state)
                memories, batch_observation, batch_state, done_data = self.remove_done_memories(
                    memories, batch_observation, batch_state
                )
                all_done_data.extend(done_data)

        samples = list(filter(lambda x: len(x["state_history"]) > 1, all_done_data))
        steps.sort()
        data2evaluate = {
            step: {
                "initial_scores": [],
                "final_scores": [],
                "path_lens": [],
            }
            for step in steps
        }
        for sample in samples:
            learning_goal = sample["learning_goals"][0]
            states = list(map(lambda x: float(x[learning_goal]), sample["state_history"]))
            for step in steps:
                data2evaluate[step]["path_lens"].append(min(step, len(states) - 1))
                data2evaluate[step]["initial_scores"].append(states[0])
                data2evaluate[step]["final_scores"].append(states[min(step, len(states) - 1)])
                if step > len(states):
                    break

        performances = {}
        for step in steps:
            initial_scores = data2evaluate[step]["initial_scores"]
            final_scores = data2evaluate[step]["final_scores"]
            path_lens = data2evaluate[step]["path_lens"]
            performances[step] = promotion_report(initial_scores, final_scores, path_lens)

        return performances
    
    def get_performance_str(self, performance):
        performance_str = ""
        for step, step_n_performance in performance.items():
            performance_str += f"step{step}, "
            for metric_name, metric_value in step_n_performance.items():
                performance_str += f"{metric_name}: {metric_value:<9.5}, "
            performance_str += "\n"
        return performance_str