model_dir_names=("qDKT@@pykt_setting@@assist2009_train_fold_0@@seed_0@@2025-03-02@02-27-51"
"qDKT@@pykt_setting@@assist2009_train_fold_1@@seed_0@@2025-03-02@02-28-37"
"qDKT@@pykt_setting@@assist2009_train_fold_2@@seed_0@@2025-03-02@02-29-31"
"qDKT@@pykt_setting@@assist2009_train_fold_3@@seed_0@@2025-03-02@02-30-31"
"qDKT@@pykt_setting@@assist2009_train_fold_4@@seed_0@@2025-03-02@02-31-21"
)
dataset_name="assist2009"
{
  for model_dir_name in "${model_dir_names[@]}"; do
    python /root/code/pydemine/examples/knowledge_tracing/evaluate/sequential_dlkt.py \
      --model_dir_name "${model_dir_name}" --dataset_name "${dataset_name}" --test_file_name "${dataset_name}_test.txt"
  done
} >> "/root/code/pydemine/examples/knowledge_tracing/evaluate/scrip_result/qdkt_${dataset_name}.txt"