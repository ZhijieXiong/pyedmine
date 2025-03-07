import numpy as np
from copy import deepcopy


class TrainRecord:
    def __init__(self):
        self.record = {
            "current_epoch": 0,
            "best_train_main_metric": -10000,
            "best_valid_main_metric": -10000,
            "best_epoch_by_train": 1,
            "best_epoch_by_valid": 1,
            "loss_train": {},
            "performance_train": [],
            "performance_valid": []
        }

    def get_loss(self):
        loss = {}
        for loss_name in self.record["loss_train"]:
            if len(self.record["loss_train"][loss_name]["loss_all"]) != 0:
                loss_mean = np.sum(self.record["loss_train"][loss_name]["loss_all"]) / \
                            np.sum(self.record["loss_train"][loss_name]["num_sample_all"])
                loss[loss_name] = loss_mean
        return loss

    def get_loss_str(self):
        loss_str = ""
        for loss_name in self.record["loss_train"]:
            if len(self.record["loss_train"][loss_name]["loss_all"]) == 0:
                loss_str += f"{loss_name}: {'not record':<12}, "
            else:
                loss_mean = np.sum(self.record["loss_train"][loss_name]["loss_all"]) / \
                            np.sum(self.record["loss_train"][loss_name]["num_sample_all"])
                loss_str += f"{loss_name}: {loss_mean:<12.6}, "
        return loss_str[:-2]

    def add_loss(self, loss_name, loss, num_sample):
        if loss_name in self.record["loss_train"]:
            self.record["loss_train"][loss_name]["loss_all"].append(loss)
            self.record["loss_train"][loss_name]["num_sample_all"].append(num_sample)
        else:
            self.record["loss_train"][loss_name] = {
                "loss_all": [loss],
                "num_sample_all": [num_sample]
            }

    def clear_loss(self):
        for v in self.record["loss_train"].values():
            v["loss_all"] = []
            v["num_sample_all"] = []

    def next_epoch(self, train_performance, valid_performance=None, main_metric_key=None, use_multi_metrics=False,
                   multi_metrics=None):
        self.record["current_epoch"] += 1
        self.update_best_metric(train_performance, False, main_metric_key, use_multi_metrics,
                                multi_metrics)
        if valid_performance is not None:
            self.update_best_metric(valid_performance, True, main_metric_key, use_multi_metrics,
                                    multi_metrics)

    def get_current_epoch(self):
        return self.record["current_epoch"]

    def get_best_epoch(self, performance_by_valid=True):
        if performance_by_valid:
            performance_index = self.record["best_epoch_by_valid"]
        else:
            performance_index = self.record["best_epoch_by_train"]
        return performance_index

    def update_best_metric(self, performance, update_by_valid=True, main_metric_key=None, use_multi_metrics=False,
                           multi_metrics=None):
        if use_multi_metrics:
            main_metric = self.cal_main_metric(performance, multi_metrics)
        else:
            main_metric = (-1 if main_metric_key in ["RMSE", "MAE"] else 1) * performance[main_metric_key]
        performance["main_metric"] = main_metric

        if update_by_valid:
            self.record["performance_valid"].append(performance)
            if (main_metric - self.record["best_valid_main_metric"]) >= 0.001:
                self.record["best_valid_main_metric"] = main_metric
                self.record["best_epoch_by_valid"] = self.record["current_epoch"]
        else:
            self.record["performance_train"].append(performance)
            if (main_metric - self.record["best_train_main_metric"]) >= 0.001:
                self.record["best_train_main_metric"] = main_metric
                self.record["best_epoch_by_train"] = self.record["current_epoch"]

    def get_performance(self, performance_type, index=-1):
        if performance_type == "train":
            performance = deepcopy(self.record["performance_train"][index])
        elif performance_type == "valid":
            performance = deepcopy(self.record["performance_valid"][index])
        else:
            raise ValueError(f"performance_type must be train or valid")
        del performance["main_metric"]
        return performance

    def get_performance_str(self, performance_type, index=-1):
        if performance_type == "train":
            performance = self.record["performance_train"][index]
        elif performance_type == "valid":
            performance = self.record["performance_valid"][index]
        else:
            raise ValueError(f"performance_type must be train or valid")
        result = f"main metric: {performance['main_metric']:<9.5}, "
        for metric_name, metric_value in performance.items():
            if metric_name == "main_metric":
                continue
            result += f"{metric_name}: {metric_value:<9.5}, "
        return result

    def stop_training(self, max_epoch, use_early_stop, num_epoch_early_stop):
        current_epoch = self.record["current_epoch"]
        best_epoch_by_valid = self.record["best_epoch_by_valid"]
        stop_flag = current_epoch >= max_epoch
        if use_early_stop:
            stop_flag = stop_flag or ((current_epoch - best_epoch_by_valid) >= num_epoch_early_stop)
        return stop_flag

    def get_evaluate_result_str(self, performance_type, performance_by):
        performance = self.get_evaluate_result(performance_type, performance_by)
        result = f"main metric: {performance['main_metric']:<9.5}, "
        for metric_name, metric_value in performance.items():
            result += f"{metric_name}: {metric_value:<9.5}, "
        return result

    def get_evaluate_result(self, performance_type, performance_by):
        if performance_type == "train":
            all_performance = self.record["performance_train"]
        elif performance_type == "valid":
            all_performance = self.record["performance_valid"]
        else:
            raise ValueError(f"performance_type must be train or valid")
        if performance_by == "train":
            performance_index = self.record["best_epoch_by_train"] - 1
        elif performance_by == "valid":
            performance_index = self.record["best_epoch_by_valid"] - 1
        else:
            raise ValueError(f"performance_by must be train or valid")
        performance = all_performance[performance_index]
        return performance

    @staticmethod
    def cal_main_metric(performance: dict[str, float], multi_metrics: list[tuple[str, float, bool]]) -> float:
        """
        Computes a composite evaluation metric by combining multiple performance metrics with their respective weights and directions. This allows for a unified score to compare models based on multiple criteria.
        :param performance: A dictionary containing performance metrics, e.g., {"AUC": 0.85, "ACC": 0.90, "MAE": 0.10, "RMSE": 0.15}.
        :param multi_metrics: A list of tuples specifying how to combine the metrics. Each tuple contains. || metric_name: The name of the metric (e.g., "AUC"). || metric_weight: The weight of the metric in the final score. || is_ascending_metric: A boolean indicating whether the metric is ascending (higher is better, e.g., True for AUC) or descending (lower is better, e.g., False for RMSE).
        :return: A single composite metric (final_metric) representing the weighted combination of the input metrics.
        """
        final_metric = 0
        for metric_name, metric_weight, is_ascending_metric in multi_metrics:
            if is_ascending_metric:
                final_metric += performance[metric_name] * metric_weight
            else:
                final_metric -= performance[metric_name] * metric_weight
        return final_metric
