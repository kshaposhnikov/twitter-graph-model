from matplotlib import pyplot as plt
from networkit import generators, overview, centrality
from pymongo import MongoClient

from tgml.loader.mongodbloader import MongoDBLoader
from tgml.validator.validator import Validator

import powerlaw

def run_test():
    g = generators.ErdosRenyiGenerator(10, 0.5).generate()
    overview(g)

def classify():
    client = MongoClient('localhost', 27017)
    try:
        print("Loading...")
        graph = MongoDBLoader(client).load_as_slice()
        print("Processing...")
        overview(graph)
        dd = plot_degree_distribution(graph)
        print("Calculate alpha")
        alpha = powerlaw.Fit(dd).alpha
        print("Alpha: {0}".format(alpha))
        print("Done")
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
    return dd

def test_generator():
    Validator(10, 10).validate()

if __name__ == '__main__':
    classify()