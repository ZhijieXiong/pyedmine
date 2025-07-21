import argparse
import os

from config import config_roster, current_dir
from utils import *

from edmine.roster.DLKTRoster import DLKTRoster


def total_data2single_data(total_data, span):
    assert span >= 2, "span must greater than 1"
    single_data = []
    span = min(span, total_data["seq_len"])
    for i in range(2, span+2):
        item_data = {}
        for k, v in total_data.items():
            if type(v) is list:
                item_data[k] = v[:i]
            else:
                item_data[k] = v
        item_data["seq_len"] = i
        single_data.append(item_data)
    return single_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir_name", type=str,
                        default=r"qDKT@@pykt_setting@@assist2009_train@@seed_0@@2025-07-18@20-22-14")
    parser.add_argument("--model_file_name", type=str, help="文件名", default="saved.ckt")
    parser.add_argument("--model_name_in_ckt", type=str, help="文件名", default="best_valid")
    parser.add_argument("--dataset_name", type=str, default="assist2009", help="for Q table")
    parser.add_argument("--span", type=int, default=30, help="num of step to plot")
    parser.add_argument("--batch_size", type=int, default=64)
    
    args = parser.parse_args()
    params = vars(args)
    
    global_params, global_objects = config_roster(params)
    
    q2c = global_objects["dataset"]["q2c"]
    user_data = {
        "question_seq": [7383,7364,7391,7342,7359,7356,7362,11974,11998,12089,11987,12121,12108,11995,12043,11932,12144,11936,12063,11903,12023,12149,12787,12387,12013,12018,12064,11909,12120,12088,12691,12600,12323,7148,7095,5954,6171,6157,6725,6485,5949,7738,7759,7706,7760,7709,7701,7763,7743,7745,7744,7773,13035,12965,7779,7761,7700,7752,7753,7735,12934,12999,12920,12919,7891,7860,7883,7892,7795,7794,7921,7821,7481,10318,10305,10372,10441,14394,7837,5754,5475,5624,5520,5495,5721,5430,8802,8791,8809,8035,12572,9214,9151,9098,9317,9326,9243,9669,9636,9756,9679,15069,15563,15660,15288,15047,14681,14748,14978,15325,15326,15327,14778,11435,15768,15769,15752,15937,15788,15789,15870,15938,15939,15940,15827,16123,16124,16125,15742,16113,16114,16115,15821,15822,15724,15913,15914,15915,15897,15936,15929,15946,15783,14792,15483,15484,15485,14717,15217,15218,15219,14798,14799,14800,14801,14713,15269,15270,15271,14750,15291,15292,15293,14807,14902,14903,14904,14926,14927,14928,14929,14796,14873,14709,15531,15052,14698,14732,14733,14734,14696,14747,14737,15764,15835,15731,15734,15735,15736,11439,15833,15878,15879,15880,15729,11440,15798,15868,15869,15770],
        "correctness_seq": [0,0,0,0,1,1,1,1,0,0,0,0,1,0,1,1,0,1,1,0,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,0,0,1,1,1,1,0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,0,1,0,0,0,1,0,1,1,1,1,1,1,1,0,1,1,0,0,1,0,1,0,0,1,0,1,0,0,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,1,0,1,1,0,0,1,1,1,0,0,1,0,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,1,1,1,0,0,1,1,1,1,0,1,0,0,1],
        "mask_seq": [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        "seq_len": 200
    }
    user_data["concept_seq"] = [q2c[q_id][0] for q_id in user_data["question_seq"]]
    target_seq_len = min(user_data["seq_len"], params["span"])
    user_data_ = total_data2single_data(user_data, target_seq_len)
    
    state_seq = []
    cur_idx = 0
    all_batch = data2batches(user_data_, params["batch_size"])
    roster = DLKTRoster(global_params, global_objects)
    for batch in all_batch:
        state = roster.get_knowledge_state(batch).cpu().tolist()
        state_seq.extend(state)
        
    concept_seq = user_data["concept_seq"][:target_seq_len]
    trace_related_cs_change(
        state_seq, 
        concept_seq,
        user_data["correctness_seq"][:target_seq_len],
        figsize=(22, 3)
    ).savefig(os.path.join(current_dir, "trace_related_cs_change.png"))
    
    target_cs = [0, 3, 5, 10]
    trace_selected_cs_change(
        state_seq, 
        user_data["question_seq"][:target_seq_len],
        user_data["correctness_seq"][:target_seq_len],
        target_cs,
        figsize=(22, 3)
    ).savefig(os.path.join(current_dir, "trace_selected_cs_change.png"))
    
    target_concept = 37
    c_state_seq = []
    qc_relation_seq = []
    for state, q_id in zip(state_seq[:target_seq_len], user_data["question_seq"][:target_seq_len]):
        c_state_seq.append(float(state[target_concept]))
        q_realted_cs = q2c[q_id]
        if target_concept in q_realted_cs:
            qc_relation_seq.append(1/len(q_realted_cs))
        else:
            qc_relation_seq.append(0)
    trace_single_concept_change(
        target_concept,
        c_state_seq,
        qc_relation_seq,
        user_data["correctness_seq"][:target_seq_len],
        figsize=(22, 3)
    ).savefig(os.path.join(current_dir, "trace_single_concept_change.png"))
    