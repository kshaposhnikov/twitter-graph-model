from matplotlib import pyplot as plt
from networkit import generators, overview, centrality, components
from pymongo import MongoClient

from tgml.loader.mongodbloader import MongoDBLoader
from tgml.util.graphhelper import get_giant_component
from tgml.validator.characteristics import CharacteristicVector
from tgml.validator.validator import Validator

import logging

import powerlaw

def run_test():
    g = generators.ErdosRenyiGenerator(10, 0.5).generate()
    overview(g)

def classify():
    logger = logging.getLogger("tgml")
    logger.setLevel(logging.DEBUG)

    client = MongoClient('localhost', 27017)
    try:
        logger.info("Loading...")
        graph_set = MongoDBLoader(client).load_as_slice(item_size=120)
        characteristics = CharacteristicVector()
        for index, graph in enumerate(graph_set):
            logger.info("Components")
            tmp_graph = get_giant_component(graph)

            logger.info("Processing graph {0} ...".format(index))
            overview(tmp_graph)
            plot_degree_distribution(tmp_graph)
            #logger.info("Calculate alpha")
            #alpha = powerlaw.Fit(dd).alpha
            #logger.info("Alpha: {0}".format(alpha))

            logger.info("Characteristics:")
            logger.info(characteristics.build_vector(tmp_graph))

            logger.info("****************************************")
        logger.info("Done")
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