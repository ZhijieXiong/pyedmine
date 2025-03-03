model_dir_names=("DKT@@pykt_setting@@assist2009_train_fold_0@@seed_0@@2025-03-02@01-56-01"
"DKT@@pykt_setting@@assist2009_train_fold_1@@seed_0@@2025-03-02@01-57-19"
"DKT@@pykt_setting@@assist2009_train_fold_2@@seed_0@@2025-03-02@01-58-39"
"DKT@@pykt_setting@@assist2009_train_fold_3@@seed_0@@2025-03-02@01-59-53"
"DKT@@pykt_setting@@assist2009_train_fold_4@@seed_0@@2025-03-02@02-01-11"
)
dataset_name="assist2009"
{
  for model_dir_name in "${model_dir_names[@]}"; do
    python /root/code/pydemine/examples/knowledge_tracing/evaluate/sequential_dlkt.py \
      --model_dir_name "${model_dir_name}" --dataset_name "${dataset_name}" --test_file_name "${dataset_name}_test.txt"
  done
} >> "/root/code/pydemine/examples/knowledge_tracing/evaluate/scrip_result/dkt_${dataset_name}.txt"