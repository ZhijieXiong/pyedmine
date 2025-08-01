import os
import pandas as pd
import networkx as nx

from config import FILE_MANAGER_ROOT
from edmine.data.FileManager import FileManager

file_manager = FileManager(FILE_MANAGER_ROOT)
dataset_raw_path = file_manager.get_dataset_raw_path("junyi2015")
preprocessed_dir = file_manager.get_preprocessed_dir("junyi2015")
print(dataset_raw_path)
print(preprocessed_dir)

exercise_df_path = os.path.join(dataset_raw_path, "junyi_Exercise_table.csv")
topic_id_df_path = os.path.join(preprocessed_dir, "concept_id_map.csv")
target_dir = preprocessed_dir

# 加载习题-知识点映射表
exercise_df = pd.read_csv(exercise_df_path)

# 加载知识点 text → id 映射表
topic_id_df = pd.read_csv(topic_id_df_path)
topic_to_id = {
    str(row['text']).strip(): int(row['mapped_id'])
    for _, row in topic_id_df.iterrows()
    if pd.notna(row['text']) and pd.notna(row['mapped_id'])
}

# 构建 name → topic 映射
name_to_topic = {
    str(name).strip(): str(topic).strip()
    for name, topic in zip(exercise_df['name'], exercise_df['topic'])
    if pd.notna(name) and pd.notna(topic)
}

# 构建 topic 先修对
topic_prereq_pairs = set()
for _, row in exercise_df.iterrows():
    current_name = str(row['name']).strip()
    current_topic = str(row['topic']).strip()
    if pd.isna(row['prerequisites']):
        continue
    prereq_names = [x.strip() for x in str(row['prerequisites']).split(',') if x.strip()]
    for prereq_name in prereq_names:
        prereq_topic = name_to_topic.get(prereq_name)
        if prereq_topic and prereq_topic != current_topic:
            topic_prereq_pairs.add((prereq_topic, current_topic))


# 清洗数据（去除非法topic、自己指向自己的边）
cleaned_pairs = {
    (from_t, to_t)
    for from_t, to_t in topic_prereq_pairs
    if all(isinstance(t, str) and t.lower() != "nan" for t in (from_t, to_t))
    and (from_t != to_t)
}

# 记录已删除的边
G = nx.DiGraph()
G.add_edges_from(cleaned_pairs)
removed_edges = set()
while True:
    cycles = list(nx.simple_cycles(G))
    if not cycles:
        break
    for cycle in cycles:
        # 删除当前环中的一条边，例如第一条
        for i in range(len(cycle)):
            from_t = cycle[i]
            to_t = cycle[(i + 1) % len(cycle)]
            if G.has_edge(from_t, to_t):
                G.remove_edge(from_t, to_t)
                removed_edges.add((from_t, to_t))
                break  # 一次只删除一条边，然后重新检测 cycles
        break  # 外层 break，一次只处理一个 cycle
    
# 更新 cleaned_pairs，去除环边
cleaned_pairs = {
    (from_t, to_t)
    for from_t, to_t in cleaned_pairs
    if (from_t, to_t) not in removed_edges
}
G = nx.DiGraph()
G.add_edges_from(cleaned_pairs)
cycles = list(nx.simple_cycles(G))
if cycles:
    print("检测到闭环！所有环如下：")
    for i, cycle in enumerate(cycles):
        # 将环补成首尾相接的路径
        path = " -> ".join(cycle + [cycle[0]])
        print(f"环 {i}: {path}")
else:
    print("✅ 没有闭环，知识点先修关系是有向无环图（DAG）")

# 映射成ID对
edge_id_pairs = []
for from_topic, to_topic in cleaned_pairs:
    from_id = topic_to_id.get(from_topic)
    to_id = topic_to_id.get(to_topic)
    if from_id is not None and to_id is not None:
        edge_id_pairs.append((from_id, to_id))

# 保存 pre_relation.txt
with open(os.path.join(target_dir, "pre_relation.txt"), "w") as f:
    for from_id, to_id in sorted(edge_id_pairs):
        f.write(f"{from_id},{to_id}\n")

import os
import pandas as pd
import networkx as nx

# 构建有向图
G = nx.DiGraph()
G.add_edges_from(edge_id_pairs)

# 只从根节点（入度为0）出发提取路径
root_nodes = [node for node in G.nodes if G.in_degree(node) == 0]

all_paths = []
for start_node in root_nodes:
    for end_node in G.nodes:
        if nx.has_path(G, start_node, end_node):
            for path in nx.all_simple_paths(G, start_node, end_node):
                all_paths.append(path)

# 加上单节点路径（未出现在任何边的孤立点）
all_nodes = set(topic_to_id.values())
nodes_in_graph = set(G.nodes)
isolated_nodes = all_nodes - nodes_in_graph
for node in isolated_nodes:
    all_paths.append([node])

# 保存 pre_path.txt
with open(os.path.join(target_dir, "pre_path.txt"), "w") as f:
    for path in all_paths:
        f.write(",".join(str(node) for node in path) + "\n")
