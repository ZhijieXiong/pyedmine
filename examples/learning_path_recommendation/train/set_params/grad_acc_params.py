import argparse


def setup_grad_acc_args():
    parser = argparse.ArgumentParser(description="梯度累计", add_help=False)
    parser.add_argument("--accumulation_step", type=int, default=1,
                        help="1表示不使用，大于1表示使用accumulation_step的梯度累计")
    return parser
