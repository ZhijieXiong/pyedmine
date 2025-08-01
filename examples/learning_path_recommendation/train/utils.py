from edmine.utils.use_torch import set_seed
from edmine.dataset.SequentialKTDataset import *
from edmine.trainer.LPROfflineDRLTrainer import LPROfflineDRLTrainer
from edmine.trainer.LPROnlineDRLTrainer import LPROnlineDRLTrainer

current_best_performance = -100

def get_objective_func(parser, config_func, agent_name, agent_class):
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
        global_params, global_objects = config_func(params)

        setting_dir = global_objects["file_manager"].get_setting_dir(params["setting_name"])
        global_objects["data"] = {
            "train": read_kt_file(os.path.join(setting_dir, params["train_file_name"])),
            "valid": read_kt_file(os.path.join(setting_dir, params["valid_file_name"]))
        }
        global_objects["agents"] = {
            agent_name: agent_class(global_params, global_objects)
        }

        if agent_name in ["D3QN"]:
            trainer = LPROfflineDRLTrainer(global_params, global_objects)
        else:
            trainer = LPROnlineDRLTrainer(global_params, global_objects)
        trainer.train()
        if agent_name in ["D3QN"]:
            performance_this = trainer.train_record.get_evaluate_result()["main_metric"]
        else:
            performance_this = trainer.best_valid_main_metric

        if (performance_this - current_best_performance) >= 0.001:
            current_best_performance = performance_this
            print(f"current best params (performance is {performance_this}):\n    " +
                    ", ".join(list(map(lambda s: f"{s}: {parameters[s]}", parameters.keys()))))
        return -performance_this
    return objective
