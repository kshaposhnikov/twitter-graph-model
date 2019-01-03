import networkit

from loader.MongoGateway import gateway
from util.graphhelper import merge_db_line


class DBStorage:

    def store(self, node):
        gateway.graph2().insert_one(node)
        return


class GraphStorage:

    def __init__(self):
        self._graph = networkit.Graph()

    def store(self, node):
        merge_db_line(self._graph, node)

    @property
    def graph(self):
        return self._graph
