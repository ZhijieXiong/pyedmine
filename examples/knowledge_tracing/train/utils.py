from torch.utils.data import DataLoader

from edmine.utils.use_torch import set_seed
from edmine.dataset.SequentialKTDataset import BasicSequentialKTDataset
from edmine.trainer.SequentialDLKTTrainer import SequentialDLKTTrainer

current_best_performance = -100

def get_objective_func(parser, config_func, model_name, model_class):
    def objective(parameters):
        global current_best_performance
        args = parser.parse_args()
        params = vars(args)

        # 替换参数
        params["search_params"] = True
        params["save_model"] = False
        params["debug_mode"] = False
        params["use_cpu"] = False
        for param_name in parameters:
            params[param_name] = parameters[param_name]
        set_seed(params["seed"])
        global_params, global_objects = config_func(params)

        dataset_train = BasicSequentialKTDataset(global_params["datasets_config"]["train"], global_objects)
        dataloader_train = DataLoader(dataset_train, batch_size=params["train_batch_size"], shuffle=True)
        dataset_valid = BasicSequentialKTDataset(global_params["datasets_config"]["valid"], global_objects)
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