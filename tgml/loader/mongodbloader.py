import networkit
from pymongo import MongoClient


class MongoDBLoader:

    def __init__(self, connection: MongoClient):
        self.client = connection
        self.db = self.client['twitter-crawler']

    def load(self, depth=0):
        graph = networkit.Graph()
        cur_depth = 0
        for node in self.db['graph2'].find():
            if depth != 0 and cur_depth == depth:
                break

            if graph.hasNode(node['id']):
                start_node = node['id']
            else:
                start_node = graph.addNode()

            for leaf in node['associatednodes']:
                if not graph.hasNode(leaf):
                    graph.addEdge(start_node, graph.addNode())
                else:
                    graph.addEdge(start_node, leaf)
            cur_depth += 1

        return graph
