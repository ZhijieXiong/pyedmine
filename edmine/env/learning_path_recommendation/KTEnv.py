import gym
import torch


class DLSequentialKTEnv(gym.Env):
    def __init__(self, params, objects):
        super().__init__()
        self.params = params
        self.objects = objects
        
    def _batch(self, history_data, next_rec_data=None):
        has_next_rec_data = int(next_rec_data is not None)
        if type(history_data) is dict:
            history_data = [history_data]
        if type(next_rec_data) is dict:
            next_rec_data = [next_rec_data]
        if next_rec_data is None:
            next_rec_data = [None] * len(history_data)
        if type(history_data) is list and type(next_rec_data) is list:
            max_seq_len = max(list(map(lambda x: len(x["correctness_seq"]), history_data))) + has_next_rec_data
            batch = {k: [] for k in history_data[0].keys()}
            for item_data, next_rec in zip(history_data, next_rec_data):
                seq_len = len(item_data["correctness_seq"]) + has_next_rec_data
                batch["seq_len"].append(seq_len)
                for k, v in item_data.items():
                    if type(v) is list:
                        if next_rec is None:
                            seq = v + [0] * (max_seq_len - seq_len)
                        else:
                            seq = v + [next_rec[k]] + [0] * (max_seq_len - seq_len)
                        batch[k].append(seq)
                    else:
                        if k != "seq_len":
                            batch[k].append(v)
            for k in batch.keys():
                if k not in ["weight_seq", "hint_factor_seq", "attempt_factor_seq", "time_factor_seq", "correct_float"]:
                    batch[k] = torch.tensor(batch[k]).long().to(self.params["device"])
                else:
                    batch[k] = torch.tensor(batch[k]).float().to(self.params["device"])
            return batch
        else:
            raise NotImplemented()
    
    def step(self, data):
        history_data = data["history_data"]
        next_rec_data = data.get("next_rec_data", None)
        batch = self._batch(history_data, next_rec_data)
        model_name = self.params["env_config"]["model_name"]
        model = self.objects["models"][model_name]
        model.eval()
        with torch.no_grad():
            state = model.get_knowledge_state(batch)
            batch_size = batch["correctness_seq"].shape[0]
            predict_score_batch = model.get_predict_score(batch)["predict_score_batch"]
            observation = predict_score_batch[torch.arange(batch_size), batch["seq_len"] - 2]
        return observation, state
  