import argparse
import inspect
import os
import pandas as pd
from collections import defaultdict


from edmine.utils.data_io import read_json


def remove_cycles(edges):
    # 构建图的邻接表
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)

    # DFS 检测环路
    visited = set()
    on_stack = set()
    cycle_edges = set()

    def dfs(u):
        visited.add(u)
        on_stack.add(u)
        for v in graph[u]:
            if v not in visited:
                dfs(v)
            elif v in on_stack:
                cycle_edges.add((u, v))
        on_stack.remove(u)

    for u in list(graph):
        if u not in visited:
            dfs(u)

    # 去除环路中的边
    if cycle_edges:
        edges = [edge for edge in edges if edge not in cycle_edges]

    return edges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_file_name", type=str, default="concept_directed_graph_xes3g5m_glm-4-air_256_0.75.json")
    args = parser.parse_args()
    params = vars(args)

    current_file_name = inspect.getfile(inspect.currentframe())
    current_dir = os.path.dirname(current_file_name)
    graph_json = read_json(os.path.join(current_dir, "../output", params["graph_file_name"]))

    edge_list = []
    for src, dsts in graph_json.items():
        for dst in dsts:
            edge_list.append((int(src), int(dst)))

    save_path = os.path.join(current_dir, "../output", params["graph_file_name"].replace(".json", ".csv"))
    pd.DataFrame(remove_cycles(edge_list), columns=["from", "to"]).to_csv(save_path , index=False)
