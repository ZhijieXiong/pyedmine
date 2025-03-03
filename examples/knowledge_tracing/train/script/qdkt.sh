#dataset_name="assist2009"
#dim_concept=128
#dim_correctness=128
#dim_latent=64
#dim_question=64
#dropout=0.1
#weight_decay=0
#for fold in 0 1 2 3 4
#do
#  python /root/code/pydemine/examples/knowledge_tracing/train/qdkt.py \
#    --setting_name "pykt_setting" --dataset_name "${dataset_name}" \
#    --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" \
#    --weight_decay "${weight_decay}" --dim_concept "${dim_concept}" --dim_question "${dim_question}" --dim_correctness "${dim_correctness}" --dim_latent "${dim_latent}" --dropout "${dropout}" \
#    --save_model True --use_wandb True
#done



#dataset_name="assist2009-full"
#dim_concept=128
#dim_correctness=128
#dim_latent=64
#dim_question=64
#dropout=0.1
#weight_decay=0.00001
#for fold in 0 1 2 3 4
#do
#  python /root/code/pydemine/examples/knowledge_tracing/train/qdkt.py \
#    --setting_name "pykt_setting" --dataset_name "${dataset_name}" \
#    --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" \
#    --weight_decay "${weight_decay}" --dim_concept "${dim_concept}" --dim_question "${dim_question}" --dim_correctness "${dim_correctness}" --dim_latent "${dim_latent}" --dropout "${dropout}" \
#    --save_model True --use_wandb True
#done



#dataset_name="assist2017"
#dim_concept=128
#dim_correctness=64
#dim_latent=64
#dim_question=64
#dropout=0.1
#weight_decay=0.0001
#for fold in 0 1 2 3 4
#do
#  python /root/code/pydemine/examples/knowledge_tracing/train/qdkt.py \
#    --setting_name "pykt_setting" --dataset_name "${dataset_name}" \
#    --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" \
#    --weight_decay "${weight_decay}" --dim_concept "${dim_concept}" --dim_question "${dim_question}" --dim_correctness "${dim_correctness}" --dim_latent "${dim_latent}" --dropout "${dropout}" \
#    --save_model True --use_wandb True
#done
#
#
#
#dataset_name="statics2011"
#dim_concept=128
#dim_correctness=128
#dim_latent=64
#dim_question=128
#dropout=0.2
#weight_decay=0
#for fold in 0 1 2 3 4
#do
#  python /root/code/pydemine/examples/knowledge_tracing/train/qdkt.py \
#    --setting_name "pykt_setting" --dataset_name "${dataset_name}" \
#    --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" \
#    --weight_decay "${weight_decay}" --dim_concept "${dim_concept}" --dim_question "${dim_question}" --dim_correctness "${dim_correctness}" --dim_latent "${dim_latent}" --dropout "${dropout}" \
#    --save_model True --use_wandb True
#done



dataset_name="edi2020-task34"
dim_concept=64
dim_correctness=64
dim_latent=64
dim_question=64
dropout=0.2
weight_decay=0.00001
for fold in 0 1 2 3 4
do
  python /root/code/pydemine/examples/knowledge_tracing/train/qdkt.py \
    --setting_name "pykt_setting" --dataset_name "${dataset_name}" \
    --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" \
    --weight_decay "${weight_decay}" --dim_concept "${dim_concept}" --dim_question "${dim_question}" --dim_correctness "${dim_correctness}" --dim_latent "${dim_latent}" --dropout "${dropout}" \
    --save_model True --use_wandb True
done

dataset_name="slepemapy-anatomy"
dim_concept=64
dim_correctness=128
dim_latent=256
dim_question=64
dropout=0.3
weight_decay=0.0001
for fold in 0 1 2 3 4
do
  python /root/code/pydemine/examples/knowledge_tracing/train/qdkt.py \
    --setting_name "pykt_setting" --dataset_name "${dataset_name}" \
    --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" \
    --weight_decay "${weight_decay}" --dim_concept "${dim_concept}" --dim_question "${dim_question}" --dim_correctness "${dim_correctness}" --dim_latent "${dim_latent}" --dropout "${dropout}" \
    --save_model True --use_wandb True
done