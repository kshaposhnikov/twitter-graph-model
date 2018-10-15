import networkit


def merge_db_line(graph: networkit.Graph, db_line):
    if graph.hasNode(db_line['id']):
        start_node = db_line['id']
    else:
        start_node = graph.addNode()

    for leaf in db_line['associatednodes']:
        if not graph.hasNode(leaf):
            graph.addEdge(start_node, graph.addNode())
        else:
            graph.addEdge(start_node, leaf)
