import argparse

from edmine.utils.parse import str2bool


def setup_step_trainer_args():
    parser = argparse.ArgumentParser(description="Step trainer的配置", add_help=False)
    parser.add_argument("--max_step", type=int, default=50000)
    parser.add_argument("--use_early_stop", type=str2bool, default=True)
    parser.add_argument("--num_early_stop", type=int, default=10, help="num_early_stop * num_step2evaluate")
    parser.add_argument("--num_step2evaluate", type=int, default=500)
    return parser
