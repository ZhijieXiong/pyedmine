#dataset_name="assist2009"
#dim_concept=128
#dim_correctness=64
#dim_latent=128
#dropout=0.3
#weight_decay=0.0001
#python /root/code/pydemine/examples/knowledge_tracing/train/dkt_kg4ex.py \
#  --setting_name "pykt_setting" --dataset_name "${dataset_name}" \
#  --train_file_name "${dataset_name}_train.txt" --valid_file_name "${dataset_name}_valid.txt" \
#  --weight_decay "${weight_decay}" --dim_concept "${dim_concept}" --dim_correctness "${dim_correctness}" --dim_latent "${dim_latent}" --dropout "${dropout}" \
#  --save_model True --use_wandb True --main_metric "RMSE" --w_aux_loss 0.01
#
#
#
#dataset_name="assist2009-full"
#dim_concept=128
#dim_correctness=128
#dim_latent=128
#dropout=0.1
#weight_decay=0
#python /root/code/pydemine/examples/knowledge_tracing/train/dkt_kg4ex.py \
#  --setting_name "pykt_setting" --dataset_name "${dataset_name}" \
#  --train_file_name "${dataset_name}_train.txt" --valid_file_name "${dataset_name}_valid.txt" \
#  --weight_decay "${weight_decay}" --dim_concept "${dim_concept}" --dim_correctness "${dim_correctness}" --dim_latent "${dim_latent}" --dropout "${dropout}" \
#  --save_model True --use_wandb True --main_metric "RMSE" --w_aux_loss 0.01
#
#
#dataset_name="assist2015"
#dim_concept=64
#dim_correctness=64
#dim_latent=64
#dropout=0.1
#weight_decay=0.00001
#python /root/code/pydemine/examples/knowledge_tracing/train/dkt_kg4ex.py \
#  --setting_name "pykt_setting" --dataset_name "${dataset_name}" \
#  --train_file_name "${dataset_name}_train.txt" --valid_file_name "${dataset_name}_valid.txt" \
#  --weight_decay "${weight_decay}" --dim_concept "${dim_concept}" --dim_correctness "${dim_correctness}" --dim_latent "${dim_latent}" --dropout "${dropout}" \
#  --save_model True --use_wandb True --main_metric "RMSE" --w_aux_loss 0.5



#dataset_name="assist2017"
#dim_concept=128
#dim_correctness=128
#dim_latent=256
#dropout=0.1
#weight_decay=0
#python /root/code/pydemine/examples/knowledge_tracing/train/dkt_kg4ex.py \
#  --setting_name "pykt_setting" --dataset_name "${dataset_name}" \
#  --train_file_name "${dataset_name}_train.txt" --valid_file_name "${dataset_name}_valid.txt" \
#  --weight_decay "${weight_decay}" --dim_concept "${dim_concept}" --dim_correctness "${dim_correctness}" --dim_latent "${dim_latent}" --dropout "${dropout}" \
#  --save_model True --use_wandb True --main_metric "RMSE" --w_aux_loss 0.01



#dataset_name="statics2011"
#dim_concept=64
#dim_correctness=64
#dim_latent=256
#dropout=0.1
#weight_decay=0
#python /root/code/pydemine/examples/knowledge_tracing/train/dkt_kg4ex.py \
#  --setting_name "pykt_setting" --dataset_name "${dataset_name}" \
#  --train_file_name "${dataset_name}_train.txt" --valid_file_name "${dataset_name}_valid.txt" \
#  --weight_decay "${weight_decay}" --dim_concept "${dim_concept}" --dim_correctness "${dim_correctness}" --dim_latent "${dim_latent}" --dropout "${dropout}" \
#  --save_model True --use_wandb True --main_metric "RMSE"  --w_aux_loss 0.5
#
#
#dataset_name="slepemapy-anatomy"
#dim_concept=64
#dim_correctness=64
#dim_latent=128
#dropout=0.2
#weight_decay=0.00001
#python /root/code/pydemine/examples/knowledge_tracing/train/dkt_kg4ex.py \
#  --setting_name "pykt_setting" --dataset_name "${dataset_name}" \
#  --train_file_name "${dataset_name}_train.txt" --valid_file_name "${dataset_name}_valid.txt" \
#  --weight_decay "${weight_decay}" --dim_concept "${dim_concept}" --dim_correctness "${dim_correctness}" --dim_latent "${dim_latent}" --dropout "${dropout}" \
#  --save_model True --use_wandb True --main_metric "RMSE" --w_aux_loss 0.5


dataset_name="edi2020-task34"
dim_concept=64
dim_correctness=64
dim_latent=256
dropout=0.1
weight_decay=0
python /root/code/pydemine/examples/knowledge_tracing/train/dkt_kg4ex.py \
  --setting_name "pykt_setting" --dataset_name "${dataset_name}" \
  --train_file_name "${dataset_name}_train.txt" --valid_file_name "${dataset_name}_valid.txt" \
  --weight_decay "${weight_decay}" --dim_concept "${dim_concept}" --dim_correctness "${dim_correctness}" --dim_latent "${dim_latent}" --dropout "${dropout}" \
  --save_model True --use_wandb True --main_metric "RMSE" --w_aux_loss 0.001


dataset_name="edi2020-task34"
dim_concept=64
dim_correctness=64
dim_latent=256
dropout=0.1
weight_decay=0
python /root/code/pydemine/examples/knowledge_tracing/train/dkt_kg4ex.py \
  --setting_name "pykt_setting" --dataset_name "${dataset_name}" \
  --train_file_name "${dataset_name}_train.txt" --valid_file_name "${dataset_name}_valid.txt" \
  --weight_decay "${weight_decay}" --dim_concept "${dim_concept}" --dim_correctness "${dim_correctness}" --dim_latent "${dim_latent}" --dropout "${dropout}" \
  --save_model True --use_wandb True --main_metric "RMSE" --w_aux_loss 0.0001

