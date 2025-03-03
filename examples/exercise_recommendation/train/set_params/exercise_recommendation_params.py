import argparse

from edmine.utils.parse import str2bool


def setup_common_args():
    parser = argparse.ArgumentParser(description="习题推荐模型的公共配置")
    parser.add_argument("--setting_name", type=str, default="ER_offline_setting")
    parser.add_argument("--dataset_name", type=str, default="statics2011")
    # 优化器相关参数选择
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=("adam", "sgd"))
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0.9)
    # 训练策略
    parser.add_argument("--max_step", type=int, default=30000)
    parser.add_argument("--use_early_stop", type=str2bool, default=True)
    parser.add_argument("--num_early_stop", type=int, default=5, help="num_early_stop * num_step2evaluate")
    parser.add_argument("--num_step2evaluate", type=int, default=1000)
    # 其它配置
    parser.add_argument("--debug_mode", type=str2bool, default=False)
    parser.add_argument("--use_cpu", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    return parser
