import torch
import wandb
import os
from torch.nn.utils import clip_grad_norm_

from edmine.model.module.Memory import LPRMemory
from edmine.trainer.EpochTrainRecord import TrainRecordOnlyValid as TrainRecord
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


class LPROfflineDRLTrainer:
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
            done = agent.judge_done(memory, master_th=master_th)
            if done:
                done_data.append(memory.output_learning_history())
                memories[i] = None
            else:
                remain_indices.append(i)
        memories = [memories[remain_idx] for remain_idx in remain_indices]
        batch_observation = batch_observation[remain_indices]
        batch_state = batch_state[remain_indices]
        
        return memories, batch_observation, batch_state, done_data
        
    def collect_rl_data(self, kt_data, kt_data_start_idx, kt_batch_size=64):
        trainer_config = self.params["trainer_config"]
        agent_name = trainer_config["agent_name"]
        buffer_size = trainer_config["buffer_size"]
        epsilon = trainer_config["epsilon"]
        agent = self.objects["agents"][agent_name]
        env = self.objects["env_simulator"]
        
        rl_data = []
        memories = []
        kt_data_start_idx = add_memories(memories, kt_data, kt_data_start_idx, kt_batch_size)
        env_input_data = {"history_data": [memory.history_data for memory in memories]}
        batch_observation, batch_state = env.step(env_input_data)
        for memory, state in zip(memories, batch_state):
            memory.update_history_data(current_state=state)
        memories, batch_observation, batch_state, _ = self.remove_done_memories(
            memories, batch_observation, batch_state
        )
        while (len(memories) > 0) and (len(rl_data) < buffer_size):
            # 推荐习题
            next_rec_data = []
            for memory, observation, state in zip(memories, batch_observation, batch_state):
                rec_concept, rec_question = agent.recommend_qc(memory, epsilon=epsilon)
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
            rl_data.extend(agent.done_data2rl_data(done_data))
            # 添加KT数据用于模拟（要获取新加入数据的state）
            if (len(memories) < kt_batch_size) and (kt_data_start_idx < len(kt_data)):
                n = len(memories)
                kt_data_start_idx = add_memories(memories, kt_data, kt_data_start_idx, kt_batch_size - n)
                env_input_data = {"history_data": [memory.history_data for memory in memories]}
                batch_observation, batch_state = env.step(env_input_data)
                for memory, state in zip(memories[n:], batch_state[n:]):
                    memory.update_history_data(current_state=state)
                memories, batch_observation, batch_state, _ = self.remove_done_memories(
                    memories, batch_observation, batch_state
                )
        return rl_data, kt_data_start_idx
        
    def train(self):
        trainer_config = self.params["trainer_config"]
        max_epoch = trainer_config["max_epoch"]
        train_batch_size = trainer_config["train_batch_size"]
        agent_name = trainer_config["agent_name"]
        accumulation_step = trainer_config["accumulation_step"]
        interval_execute_config = trainer_config.get("interval_execute_config", None)
        models = self.objects["lpr_models"]
        grad_clips_config = self.params["grad_clips_config"]
        schedulers_config = self.params["schedulers_config"]
        optimizers = self.objects["optimizers"]
        schedulers = self.objects["schedulers"]
        agent = self.objects["agents"][agent_name]
        
        train_data = self.objects["data"]["train"]
        idx = 0
        self.objects["logger"].info(f"{get_now_time()} start training")
        for epoch in range(1, max_epoch + 1):
            for model in models.values():
                model.eval()
            self.objects["random_generator"].shuffle(train_data)
            self.objects["logger"].info(f"{get_now_time()} epoch {epoch:<4}, start collecting data")
            agent.eval()
            rl_data, idx = self.collect_rl_data(train_data, idx)
            agent.train()
            # 数据遍历完了，重新遍历
            if idx >= (len(train_data) - 3):
                idx = 0
            self.objects["logger"].info(f"{get_now_time()} epoch {epoch:<4}, gathered {len(rl_data)} data samples")
            self.objects["random_generator"].shuffle(rl_data)

            for model in models.values():
                model.train()
            for i in range(0, len(rl_data), train_batch_size):
                batch_i = i // train_batch_size
                batch_data = rl_data[i:i+train_batch_size]
                batch = agent.data2batch(batch_data)
                all_loss = agent.get_all_loss(batch)
                for model_name, loss_result in all_loss.items():
                    for loss_name, loss_info in loss_result["losses_value"].items():
                        if loss_name != "total loss":
                            self.train_record.add_loss(loss_name, loss_info["value"], loss_info["num_sample"])
                    model = models[model_name]
                    grad_clip_config = grad_clips_config[model_name]
                    optimizer = optimizers[model_name]
                    loss = loss_result["total_loss"] / accumulation_step
                    loss.backward()
                    if (batch_i + 1) % accumulation_step == 0:
                        if grad_clip_config["use_clip"]:
                            clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
                        optimizer.step()
                        optimizer.zero_grad()
                for model_name in all_loss.keys():
                    scheduler_config = schedulers_config[model_name]
                    scheduler = schedulers[model_name]
                    if scheduler_config["use_scheduler"]:
                        scheduler.step()
            for model in models.values():
                model.eval()
            self.evaluate()
            if self.stop_train():
                break
            
            # 自由配置每隔多少个epoch，做什么操作
            # step_action_config: dict[int, list[func]]
            if interval_execute_config is not None:
                for interval, target_funcs in interval_execute_config.items():
                    to_execute_func = (epoch - 1) % interval == 0
                    if to_execute_func:
                        for target_func in target_funcs:
                            self.objects["logger"].info(f"{get_now_time()} epoch {epoch:<4}, execute {target_func.__name__}")
                            target_func()

    def evaluate(self):
        trainer_config = self.params["trainer_config"]
        use_wandb = trainer_config["use_wandb"]
        agent_name = trainer_config["agent_name"]
        save_model = trainer_config["save_model"]
        use_multi_metrics = trainer_config["use_multi_metrics"]
        main_metric = trainer_config["main_metric"]
        multi_metrics = trainer_config["multi_metrics"]
        valid_data = self.objects["data"]["valid"]
        env = self.objects["env_simulator"]
        agent = self.objects["agents"][agent_name]
        models = self.objects["lpr_models"]

        valid_performance = self.evaluate_dataset(env, agent, valid_data)
        # todo: 这里后面要修改，通用的trainer不应该调用get_average_performance_top_ns
        average_valid_performance = get_average_performance_top_ns(valid_performance)
        self.train_record.next_epoch(average_valid_performance, main_metric, use_multi_metrics, multi_metrics)
        best_epoch = self.train_record.get_best_epoch()
        valid_performance_str = self.train_record.get_performance_str()
        self.objects["logger"].info(
            f"{get_now_time()} epoch {self.train_record.get_current_epoch():<3} , valid performances are "
            f"{valid_performance_str}train loss is {self.train_record.get_loss_str()}, current best epoch is "
            f"{best_epoch}")
        if use_wandb:
            valid_performance = self.train_record.get_performance("valid")
            loss = self.train_record.get_loss()
            valid_performance.update(loss)
            wandb.log(valid_performance)
        self.train_record.clear_loss()

        best_epoch = self.train_record.get_best_epoch()
        current_epoch = self.train_record.get_current_epoch()
        if best_epoch == current_epoch:
            if save_model:
                save_model_dir = self.params["save_model_dir"]
                model_weight_path = os.path.join(save_model_dir, "saved.ckt")
                torch.save({
                    model_name: model.state_dict() for model_name, model in models.items()
                }, model_weight_path)

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
                rec_concept, rec_question = agent.recommend_qc(memory, epsilon=0)
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
    
    def stop_train(self):
        trainer_config = self.params["trainer_config"]
        max_epoch = trainer_config["max_epoch"]
        use_early_stop = trainer_config["use_early_stop"]
        num_epoch_early_stop = trainer_config["num_epoch_early_stop"]

        stop_flag = self.train_record.stop_training(max_epoch, use_early_stop, num_epoch_early_stop)
        if stop_flag:
            best_valid_performance_by_valid = self.train_record.get_evaluate_result_str()
            self.objects["logger"].info(
                f"best valid epoch: {self.train_record.get_best_epoch():<3} , "
                f"valid performances in best epoch by valid are {best_valid_performance_by_valid}\n"
            )

        return stop_flag
    