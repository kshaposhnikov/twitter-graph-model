from networkit import Graph


class MongoDBStorage:

    def storeGraph(self, collection, graph: Graph):
        converted_graph = {"nodes": []}
        for node in graph.nodes():
            associatednodes = []
            graph.forEdgesOf(node, lambda left, right, weight, edge_id:
                (associatednodes.append(right))
            )

            converted_graph["nodes"].append({
                "id": node,
                "associatednodescount": len(associatednodes),
                "associatednodes": associatednodes
            })

        collection.insert_one(converted_graph)

