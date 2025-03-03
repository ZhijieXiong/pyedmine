kt_model="dkt"
theta=0.2
dataset_name="statics2011"
echo "search KG4EX params in ${dataset_name}, kt model: ${kt_model}, theta: ${theta}"
python /root/code/pydemine/examples/exercise_recommendation/train/kg4ex_search_params.py \
--setting_name "ER_offline_setting" --dataset_name "${dataset_name}" \
--user_data_file_name "${dataset_name}_user_data.txt" \
--train_file_name "${dataset_name}_train_triples_${kt_model}_${theta}.txt" \
--valid_file_name "${dataset_name}_valid_triples_${kt_model}_${theta}.txt" \
--valid_mlkc_file_name "${dataset_name}_${kt_model}_mlkc_valid.txt" \
--valid_pkc_file_name "${dataset_name}_pkc_valid.txt" \
--valid_efr_file_name "${dataset_name}_efr_${theta}_valid.txt" \
--evaluate_batch_size 8 \

dataset_name="assist2009"
echo "search KG4EX params in ${dataset_name}, kt model: ${kt_model}, theta: ${theta}"
python /root/code/pydemine/examples/exercise_recommendation/train/kg4ex_search_params.py \
--setting_name "ER_offline_setting" --dataset_name "${dataset_name}" \
--user_data_file_name "${dataset_name}_user_data.txt" \
--train_file_name "${dataset_name}_train_triples_${kt_model}_${theta}.txt" \
--valid_file_name "${dataset_name}_valid_triples_${kt_model}_${theta}.txt" \
--valid_mlkc_file_name "${dataset_name}_${kt_model}_mlkc_valid.txt" \
--valid_pkc_file_name "${dataset_name}_pkc_valid.txt" \
--valid_efr_file_name "${dataset_name}_efr_${theta}_valid.txt" \
--evaluate_batch_size 1 \


kt_model="qdkt"
dataset_name="statics2011"
echo "search KG4EX params in ${dataset_name}, kt model: ${kt_model}, theta: ${theta}"
python /root/code/pydemine/examples/exercise_recommendation/train/kg4ex_search_params.py \
--setting_name "ER_offline_setting" --dataset_name "${dataset_name}" \
--user_data_file_name "${dataset_name}_user_data.txt" \
--train_file_name "${dataset_name}_train_triples_${kt_model}_${theta}.txt" \
--valid_file_name "${dataset_name}_valid_triples_${kt_model}_${theta}.txt" \
--valid_mlkc_file_name "${dataset_name}_${kt_model}_mlkc_valid.txt" \
--valid_pkc_file_name "${dataset_name}_pkc_valid.txt" \
--valid_efr_file_name "${dataset_name}_efr_${theta}_valid.txt" \
--evaluate_batch_size 8 \

dataset_name="assist2009"
echo "search KG4EX params in ${dataset_name}, kt model: ${kt_model}, theta: ${theta}"
python /root/code/pydemine/examples/exercise_recommendation/train/kg4ex_search_params.py \
--setting_name "ER_offline_setting" --dataset_name "${dataset_name}" \
--user_data_file_name "${dataset_name}_user_data.txt" \
--train_file_name "${dataset_name}_train_triples_${kt_model}_${theta}.txt" \
--valid_file_name "${dataset_name}_valid_triples_${kt_model}_${theta}.txt" \
--valid_mlkc_file_name "${dataset_name}_${kt_model}_mlkc_valid.txt" \
--valid_pkc_file_name "${dataset_name}_pkc_valid.txt" \
--valid_efr_file_name "${dataset_name}_efr_${theta}_valid.txt" \
--evaluate_batch_size 1 \