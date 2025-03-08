import dgl


def build_graph4rcd(g_path, node, directed=True):
    g = dgl.DGLGraph()
    g.add_nodes(node)
    edge_list = []
    with open(g_path, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '').split('\t')
            edge_list.append((int(line[0]), int(line[1])))
    if directed:
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    else:
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        g.add_edges(dst, src)
        return g
    