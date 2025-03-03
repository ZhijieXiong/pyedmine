import logging
import sys


def config_logger(local_params, global_objects, log_path=None):
    global_objects["logger"] = logging.getLogger("train_log")
    global_objects["logger"].setLevel(4)
    ch = logging.StreamHandler(stream=sys.stdout)
    if not local_params.get("search_params", False):
        ch.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.ERROR)
    global_objects["logger"].addHandler(ch)
    if log_path is not None:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        global_objects["logger"].addHandler(fh)
