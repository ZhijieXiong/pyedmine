import networkx as nx


def check_cycles(edge_pairs):
    G = nx.DiGraph()
    G.add_edges_from(edge_pairs)
    cycles = list(nx.simple_cycles(G))
    if cycles:
        print("检测到闭环！所有环如下：")
        for i, cycle in enumerate(cycles):
            # 将环补成首尾相接的路径
            path = " -> ".join(cycle + [cycle[0]])
            print(f"环 {i}: {path}")
    else:
        print("✅ 没有闭环，知识点先修关系是有向无环图（DAG）")
        
        