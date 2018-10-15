import networkit

from tgml.util.graphhelper import merge_db_line


class DBStorage:

    def __init__(self, db, target_collection='graph2'):
        self._db = db
        self._target_collection = target_collection

    def store(self, node):
        self._db[self._target_collection].insert_one(node)
        return


class GraphStorage:

    def __init__(self):
        self._graph = networkit.Graph()

    def store(self, node):
        merge_db_line(self._graph, node)

    @property
    def graph(self):
        return self._graph
