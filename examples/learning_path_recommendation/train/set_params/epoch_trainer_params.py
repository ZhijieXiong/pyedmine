import argparse

from edmine.utils.parse import str2bool


def setup_epoch_trainer_args():
    parser = argparse.ArgumentParser(description="Epoch trainer的配置", add_help=False)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--use_early_stop", type=str2bool, default=True)
    parser.add_argument("--num_epoch_early_stop", type=int, default=10)
    return parser
