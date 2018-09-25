from matplotlib import pyplot as plt
from networkit import generators, overview, centrality
from pymongo import MongoClient

from src.loader.mongodbloader import MongoDBLoader


def run_test():
    g = generators.ErdosRenyiGenerator(10, 0.5).generate()
    overview(g)

def classify():
    client = MongoClient('localhost', 27017)
    try:
        graph = MongoDBLoader(client).load()
        overview(graph)
        plot_degree_distribution(graph)
    finally:
        client.close()

def plot_degree_distribution(graph):
    dd = sorted(centrality.DegreeCentrality(graph).run().scores(), reverse=True)
    plt.xscale("log")
    plt.xlabel("degree")
    plt.yscale("log")
    plt.ylabel("number of nodes")
    plt.plot(dd)
    plt.show()

if __name__ == '__main__':
    classify()