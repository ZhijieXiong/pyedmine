from torch.utils.data import DataLoader

from edmine.utils.use_torch import set_seed
from edmine.dataset.SequentialKTDataset import *
from edmine.trainer.SequentialDLKTTrainer import SequentialDLKTTrainer

current_best_performance = -100

def get_objective_func(parser, config_func, model_name, model_class):
    def objective(parameters):
        global current_best_performance
        args = parser.parse_args()
        params = vars(args)
        set_seed(params["seed"])

        # 替换参数
        params["search_params"] = True
        params["save_model"] = False
        params["debug_mode"] = False
        params["use_cpu"] = False
        if model_name in ["DIMKT", "LPKT"]:
            params["max_epoch"] = 100
            params["num_epoch_early_stop"] = 10
        if model_name in ["LBKT"]:
            params["max_epoch"] = 50
            params["num_epoch_early_stop"] = 5
        for param_name in parameters:
            params[param_name] = parameters[param_name]
        global_params, global_objects = config_func(params)

        if model_name == "DIMKT":
            dataset_train = DIMKTDataset(global_params["datasets_config"]["train"], global_objects)
            dataset_valid = DIMKTDataset(global_params["datasets_config"]["valid"], global_objects)
        elif model_name == "LPKT":
            dataset_train = LPKTDataset(global_params["datasets_config"]["train"], global_objects)
            dataset_valid = LPKTDataset(global_params["datasets_config"]["valid"], global_objects)
        elif model_name == "LBKT":
            dataset_train = LBKTDataset(global_params["datasets_config"]["train"], global_objects)
            dataset_valid = LBKTDataset(global_params["datasets_config"]["valid"], global_objects)
        elif model_name == "QDCKT":
            dataset_train = QDCKTDataset(global_params["datasets_config"]["train"], global_objects, train_mode=True)
            dataset_valid = QDCKTDataset(global_params["datasets_config"]["valid"], global_objects, train_mode=False)
        elif model_name == "DKTForget":
            dataset_train = DKTForgetDataset(global_params["datasets_config"]["train"], global_objects)
            dataset_valid = DKTForgetDataset(global_params["datasets_config"]["valid"], global_objects)
        elif model_name == "ATDKT":
            dataset_train = ATDKTDataset(global_params["datasets_config"]["train"], global_objects, train_mode=True)
            dataset_valid = ATDKTDataset(global_params["datasets_config"]["valid"], global_objects, train_mode=False)
        else:
            dataset_train = BasicSequentialKTDataset(global_params["datasets_config"]["train"], global_objects)
            dataset_valid = BasicSequentialKTDataset(global_params["datasets_config"]["valid"], global_objects)
        dataloader_train = DataLoader(dataset_train, batch_size=params["train_batch_size"], shuffle=True)
        dataloader_valid = DataLoader(dataset_valid, batch_size=params["train_batch_size"], shuffle=False)

        global_objects["data_loaders"] = {
            "train_loader": dataloader_train,
            "valid_loader": dataloader_valid
        }
        global_objects["models"] = {
            model_name: model_class(global_params, global_objects).to(global_params["device"])
        }
        trainer = SequentialDLKTTrainer(global_params, global_objects)
        trainer.train()
        performance_this = trainer.train_record.get_evaluate_result("valid", "valid")["main_metric"]

        if (performance_this - current_best_performance) >= 0.001:
            current_best_performance = performance_this
            print(f"current best params (performance is {performance_this}):\n    " +
                    ", ".join(list(map(lambda s: f"{s}: {parameters[s]}", parameters.keys()))))
        return -performance_this
    return objective