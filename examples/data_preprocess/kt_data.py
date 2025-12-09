import argparse
import os
import inspect

from edmine.data.FileManager import FileManager
from edmine.data.KTDataProcessor import KTDataProcessor
from edmine.utils.data_io import write_kt_file, read_json


current_file_name = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(current_file_name)
settings_path = os.path.join(current_dir, "../settings.json")
settings = read_json(settings_path)
FILE_MANAGER_ROOT = settings["FILE_MANAGER_ROOT"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="DBE-KT22",
                        choices=("assist2009", "assist2009-full", "assist2012", "assist2015", "assist2017",
                                 "algebra2005", "algebra2006", "algebra2008",
                                 "bridge2algebra2006", "bridge2algebra2008",
                                 "edi2020-task1", "edi2020-task34",
                                 "SLP-bio", "SLP-chi", "SLP-eng", "SLP-geo", "SLP-his", "SLP-mat", "SLP-phy",
                                 "ednet-kt1", "slepemapy-anatomy", "xes3g5m", "statics2011", "junyi2015", "poj",
                                 "DBE-KT22"))

    args = parser.parse_args()
    params = vars(args)
    file_manager = FileManager(FILE_MANAGER_ROOT)

    params["data_path"] = file_manager.get_dataset_raw_path(params["dataset_name"])
    print(f"processing {params['dataset_name']} ...")
    data_processor = KTDataProcessor(params, file_manager)
    data_uniformed = data_processor.preprocess_data()
    Q_table = data_processor.Q_table
    data_statics_raw = data_processor.statics_raw
    data_statics_preprocessed = data_processor.statics_preprocessed

    print(f"saving data of {params['dataset_name']} ...")
    dataset_name = params["dataset_name"]
    data_path = file_manager.get_preprocessed_path(dataset_name)
    write_kt_file(data_uniformed, data_path)
    file_manager.save_data_statics_raw(data_statics_raw, params["dataset_name"])
    file_manager.save_data_statics_processed(data_statics_preprocessed, dataset_name)
    file_manager.save_q_table(Q_table, dataset_name)
    file_manager.save_data_id_map(data_processor.get_all_id_maps(), dataset_name)
    print(f"finsh processing and saving successfully")
