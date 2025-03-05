import torch


class DLKTRoster:
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects

    def process_batch4sequential_kt_model(self, batch):
        if type(batch) is list:
            max_seq_len = max(list(map(lambda x: len(x["correctness_seq"]), batch)))
            batch_converted = {k: [] for k in batch[0].keys()}
            for item_data in batch:
                for seq_name, seq in item_data.items():
                    if type(seq) is list:
                        seq += [0] * (max_seq_len - len(seq))
                    batch_converted[seq_name].append(seq)
            for k in batch_converted.keys():
                if k not in ["weight_seq", "hint_factor_seq", "attempt_factor_seq", "time_factor_seq", "correct_float"]:
                    batch_converted[k] = torch.tensor(batch_converted[k]).long().to(self.params["device"])
                else:
                    batch_converted[k] = torch.tensor(batch_converted[k]).float().to(self.params["device"])
            return batch_converted
        else:
            return batch

    def get_knowledge_state(self, batch):
        model_name = self.params["roster_config"]["model_name"]
        model = self.objects["models"][model_name]
        model.eval()
        if model.model_type == "DLSequentialKTModel":
            batch = self.process_batch4sequential_kt_model(batch)
        else:
            pass

        return model.get_knowledge_state(batch)
