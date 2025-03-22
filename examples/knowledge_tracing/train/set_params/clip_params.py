import argparse

from edmine.utils.parse import str2bool


def setup_clip_args():
    parser = argparse.ArgumentParser(description="梯度裁剪", add_help=False)
    parser.add_argument("--enable_clip_grad", type=str2bool, default=False)
    parser.add_argument("--grad_clipped", type=float, default=10.0)
    return parser
