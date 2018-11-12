from matplotlib import pyplot as plt
from networkit import generators, overview, centrality, components
from pymongo import MongoClient

from loader.mongodbloader import MongoDBLoader
from util.graphhelper import get_giant_component
from validator.characteristics import CharacteristicVector
from validator.classifier import Classifier

import logging

import powerlaw
import sys

from validator.features import FeatureVector


def run_test():
    g = generators.ErdosRenyiGenerator(10, 0.5).generate()
    overview(g)


def classify():
    logger = logging.getLogger("tgml")
    logger.setLevel(logging.DEBUG)

    client = MongoClient('localhost', 27017)
    try:
        logger.info("Loading...")
        graph_set = MongoDBLoader(client).load_as_slice(size=1, item_size=50)
        features = FeatureVector()
        classifier = Classifier(10000, 20000, class_count=1).build_svc_classifier()
        for index, graph in enumerate(graph_set):
            logger.info("Components")
            tmp_graph = get_giant_component(graph)
            print(classifier.predict_proba([features.build_vector_for_graph_as_list(tmp_graph)]))

            logger.info("****************************************")
        logger.info("Done")
    finally:
        client.close()


def is_scale_free(graph):
    dd = centrality.DegreeCentrality(graph).run().scores()
    fit = powerlaw.Fit(dd)
    res_exp, _ = fit.distribution_compare('power_law', 'exponential')
    res_trunc, _ = fit.distribution_compare('power_law', 'truncated_power_law')
    res_long, _ = fit.distribution_compare('power_law', 'lognormal')

    mes = "Not Scale Free"
    assert res_exp > 0, mes
    assert res_trunc > 0, mes
    assert res_long > 0, mes


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
    Classifier(10, 10).build_classifiers()


def test_file():
    res = dict()
    res['a'] = [1, 2, 3]
    res['b'] = [4, 5, 6]
    with open('res.txt', 'w') as f:
        print(res, file=f)


if __name__ == '__main__':
    classify()
    # test_file()
