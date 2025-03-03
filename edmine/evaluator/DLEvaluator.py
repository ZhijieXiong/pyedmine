import torch
from abc import ABC, abstractmethod


class DLEvaluator(ABC):
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects
        self.inference_results = {data_loader_name: {} for data_loader_name in self.objects["data_loaders"].keys()}

    def evaluate(self):
        model_name = self.params["evaluator_config"]["model_name"]
        model = self.objects["models"][model_name]
        for data_loader_name in self.objects["data_loaders"].keys():
            data_loader = self.objects["data_loaders"][data_loader_name]
            model.eval()
            with torch.no_grad():
                inference_result = self.inference(model, data_loader)
                self.inference_results[data_loader_name] = inference_result
        self.log_inference_results()

    @abstractmethod
    def log_inference_results(self):
        pass

    @abstractmethod
    def inference(self, model, data_loader):
        pass
