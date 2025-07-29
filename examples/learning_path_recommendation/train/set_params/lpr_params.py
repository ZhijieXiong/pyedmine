import argparse

from edmine.utils.parse import str2bool


def setup_common_args():
    parser = argparse.ArgumentParser(description="sequential kt模型的公共配置", add_help=False)
    parser.add_argument("--setting_name", type=str, default="LPR_offline_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--train_file_name", type=str, default="assist2009_single_goal_train.txt")
    parser.add_argument("--valid_file_name", type=str, default="assist2009_single_goal_valid.txt")
    parser.add_argument("--kt_setting_name", type=str, default="pykt_setting")
    # 模拟器配置
    parser.add_argument("--model_dir_name", type=str,
                        default=r"qDKT@@pykt_setting@@assist2009_train@@seed_0@@2025-07-18@20-22-14")
    parser.add_argument("--model_file_name", type=str, help="文件名", default="saved.ckt")
    parser.add_argument("--model_name_in_ckt", type=str, help="文件名", default="best_valid")
    # 掌握阈值
    parser.add_argument("--master_threshold", type=float, default=0.6)
    # 训练策略
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--use_early_stop", type=str2bool, default=True)
    parser.add_argument("--num_epoch_early_stop", type=int, default=10)
    # 评价指标选择
    parser.add_argument("--target_steps", type=str, default="[5,10,20]")
    parser.add_argument("--main_metric", type=str, default="NRPR")
    parser.add_argument("--use_multi_metrics", type=str2bool, default=False)
    parser.add_argument("--multi_metrics", type=str, default="[('NRPR', 1, 1), ('APR', 1, 1), ('RPR', 1, 1)]")
    # 其它配置
    parser.add_argument("--debug_mode", type=str2bool, default=False)
    parser.add_argument("--use_cpu", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    return parser
