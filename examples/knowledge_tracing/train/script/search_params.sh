# 大数据集
#dataset_names=("assist2012" "junyi2015" "edi2020-task1" "xes3g5m")
# 小数据集
#dataset_names=("assist2009" "assist2009-full" "assist2015" "assist2017" "statics2011" "slepemapy-anatomy" "poj" "edi2020-task34")
#
#for dataset_name in "${dataset_names[@]}"; do
#  echo "search DKT params in ${dataset_name}"
#  python /root/code/pydemine/examples/knowledge_tracing/train/dkt_search_params.py \
#    --setting_name "pykt_setting" --dataset_name "${dataset_name}" \
#    --train_file_name "${dataset_name}_train.txt" --valid_file_name "${dataset_name}_valid.txt"
#
#  echo "search qDKT params in ${dataset_name}"
#  python /root/code/pydemine/examples/knowledge_tracing/train/qdkt_search_params.py \
#    --setting_name "pykt_setting" --dataset_name "${dataset_name}" \
#    --train_file_name "${dataset_name}_train.txt" --valid_file_name "${dataset_name}_valid.txt"
#done

dataset_name="slepemapy-anatomy"
echo "search qDKT params in ${dataset_name}"
python /root/code/pydemine/examples/knowledge_tracing/train/qdkt_search_params.py \
  --setting_name "pykt_setting" --dataset_name "${dataset_name}" \
  --train_file_name "${dataset_name}_train.txt" --valid_file_name "${dataset_name}_valid.txt"