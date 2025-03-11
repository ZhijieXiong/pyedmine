import torch
import numpy as np


class DLCDRoster:
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects

    def process_batch4cd_model(self, batch):
        assert type(batch) in [list, np.ndarray, torch.Tensor], "type of batch must in [list, np.ndarray, torch.Tensor]"
        if type(batch) is not torch.Tensor:
            return torch.tensor(batch).long().to(self.params["device"])
        else:
            return batch

    def get_knowledge_state(self, batch):
        model_name = self.params["roster_config"]["model_name"]
        model = self.objects["models"][model_name]
        model.eval()
        batch = self.process_batch4cd_model(batch)
        return model.get_knowledge_state(batch)
