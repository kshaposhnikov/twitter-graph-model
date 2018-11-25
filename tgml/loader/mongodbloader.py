import networkit
from pymongo import MongoClient

from util.modelconverter import ModelConverter
from util.storage import GraphStorage


class MongoDBLoader:

    def __init__(self, connection: MongoClient):
        self.client = connection
        self.db = self.client['twitter-crawler']

    def load_full_graph(self, depth=0):
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

    def load_as_slice(self, size=10, item_size=1000):
        res = list()
        offset = 0
        converter = ModelConverter()
        for idx, slice_item in enumerate(range(size)):
            source = self.db['graph2'].find().skip(offset).limit(item_size + offset)
            storage = GraphStorage()
            converter.convert(source, storage)
            res.append(storage.graph)
            offset += item_size
            print("\rLoaded {0}".format(idx))

        return res

    def load_one(self, size, number):
        converter = ModelConverter()
        source = self.db['graph2'].find().skip((number * size) - size).limit(number * size)
        storage = GraphStorage()
        converter.convert(source, storage)

        return storage.graph
