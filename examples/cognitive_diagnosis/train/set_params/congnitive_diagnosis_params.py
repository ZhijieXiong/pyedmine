import argparse

from edmine.utils.parse import str2bool


def setup_common_args():
    parser = argparse.ArgumentParser(description="认知诊断模型的公共配置")
    parser.add_argument("--setting_name", type=str, default="ncd_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--train_file_name", type=str, default="assist2009_train_fold_0.txt")
    parser.add_argument("--valid_file_name", type=str, default="assist2009_valid_fold_0.txt")
    # 训练策略
    parser.add_argument("--max_epoch", type=int, default=50)
    parser.add_argument("--use_early_stop", type=str2bool, default=True)
    parser.add_argument("--num_epoch_early_stop", type=int, default=5)
    # 评价指标选择
    parser.add_argument("--main_metric", type=str, default="AUC")
    parser.add_argument("--use_multi_metrics", type=str2bool, default=False)
    parser.add_argument("--multi_metrics", type=str, default="[('AUC', 1, 1), ('ACC', 1, 1), ('RMSE', 1, -1)]")
    # 其它配置
    parser.add_argument("--debug_mode", type=str2bool, default=False)
    parser.add_argument("--use_cpu", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    return parser
