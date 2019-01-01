from pymongo import MongoClient


class MongoGateway:

    __shared_state = {}

    def __init__(self, connection: MongoClient):
        self.__dict__ = self.__shared_state
        self._client = connection
        self._db = self._client['twitter-crawler']

    def get_collection(self, name):
        return self._db[name]

    def graph2(self):
        return self.get_collection('graph2')

    def real_features(self):
        return self.get_collection('real_graph_features')

    def close_connection(self):
        self._client.close()


gateway = MongoGateway(MongoClient('localhost', 27017))
