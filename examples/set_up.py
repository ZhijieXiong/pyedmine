import json
import os
import inspect

from edmine.data.FileManager import FileManager

current_file_name = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(current_file_name)
settings_path = os.path.join(current_dir, "./settings.json")
with open(settings_path, "r") as f:
    settings = json.load(f)
FILE_MANAGER_ROOT = settings["FILE_MANAGER_ROOT"]


if __name__ == "__main__":
    kt_file_manager = FileManager(root_dir=FILE_MANAGER_ROOT, init_dirs=True)
