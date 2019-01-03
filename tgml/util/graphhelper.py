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


def get_giant_component(graph):
    components = networkit.components.ConnectedComponents(graph)
    components.run()

    if components.numberOfComponents() == 1:
        return graph

    giant_id = max(components.getPartition().subsetSizeMap().items(),
                   key=lambda x: x[1])
    giant_comp = components.getPartition().getMembers(giant_id[0])
    for v in graph.nodes():
        if v not in giant_comp:
            for u in graph.neighbors(v):
                graph.removeEdge(v, u)
            graph.removeNode(v)
    name = graph.getName()
    graph = networkit.graph.GraphTools.getCompactedGraph(
        graph,
        networkit.graph.GraphTools.getContinuousNodeIds(graph)
    )
    graph.setName(name)
    return graph


def collection_to_list(collection):
    tmp = []
    for key, value in collection.items():
        if key != '_id':
            tmp.extend(value)
    return tmp
