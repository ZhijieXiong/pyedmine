import torch.nn as nn


class PredictorLayer(nn.Module):
    def __init__(self, predictor_config):
        super(PredictorLayer, self).__init__()
        self.predictor_config = predictor_config
        if predictor_config["type"] == "direct":
            dropout = predictor_config["dropout"]
            num_predict_layer = predictor_config["num_predict_layer"]
            dim_predict_in = predictor_config["dim_predict_in"]
            dim_predict_mid = predictor_config["dim_predict_mid"]
            activate_type = predictor_config["activate_type"]

            if activate_type == "tanh":
                act_func = nn.Tanh
            elif activate_type == "relu":
                act_func = nn.ReLU
            else:
                act_func = nn.Sigmoid

            dim_predict_out = predictor_config["dim_predict_out"]
            self.predict_layer = []
            if num_predict_layer == 1:
                self.predict_layer.append(nn.Dropout(dropout))
                self.predict_layer.append(nn.Linear(dim_predict_in, dim_predict_out))
                self.predict_layer.append(nn.Sigmoid())
            else:
                self.predict_layer.append(nn.Linear(dim_predict_in, dim_predict_mid))
                for _ in range(num_predict_layer - 1):
                    self.predict_layer.append(act_func())
                    self.predict_layer.append(nn.Dropout(dropout))
                    self.predict_layer.append(nn.Linear(dim_predict_mid, dim_predict_mid))
                self.predict_layer.append(nn.Dropout(dropout))
                self.predict_layer.append(nn.Linear(dim_predict_mid, dim_predict_out))
                self.predict_layer.append(nn.Sigmoid())
            self.predict_layer = nn.Sequential(*self.predict_layer)
        elif predictor_config["type"] == "dot":
            pass
        else:
            raise NotImplementedError()

    def forward(self, batch):
        y = self.predict_layer(batch)
        if self.predictor_config["type"] == "direct":
            last_layer_max_value = self.predictor_config["last_layer_max_value"]
            y = y * last_layer_max_value
        return y
