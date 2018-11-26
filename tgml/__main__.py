from matplotlib import pyplot as plt
from networkit import generators, overview, centrality, components
from pymongo import MongoClient
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from loader.mongodbloader import MongoDBLoader
from util.graphhelper import get_giant_component, collection_to_list
from util.storage import DBStorage
from validator.characteristics import CharacteristicVector
from validator.classifier import Classifier

import logging

import powerlaw
import sys

from validator.features import FeatureVector
from validator.generator.bagenerator import BAGenerator
from validator.generator.chunglugenerator import CLGenerator
from validator.generator.ergenerator import ERGenerator


def run_test():
    g = generators.ErdosRenyiGenerator(10, 0.5).generate()
    overview(g)


def classify_in_flight(class_count=3):
    logger = logging.getLogger("tgml")
    logger.setLevel(logging.DEBUG)

    client = MongoClient('localhost', 27017)
    try:
        logger.info("Loading...")
        graph_set = MongoDBLoader(client).load_as_slice(size=class_count, item_size=60)
        features = FeatureVector()

        node_count = 1010

        gens = [BAGenerator(node_count), ERGenerator(node_count),
                CLGenerator([graph_set[0].degree(v) for v in graph_set[0].nodes()])]
        classifier = Classifier(gens, class_count=class_count).build_svc_classifier()
        res = []
        for index, graph in enumerate(graph_set):
            logger.info("Components")
            tmp_graph = get_giant_component(graph)
            vector = [features.build_vector_for_graph_as_list(tmp_graph)]
            proba = classifier.predict_proba(vector)
            print(proba)
            print(classifier.predict(vector))
            res.append(proba)

            logger.info("****************************************")
        logger.info("Done")

        print(res)
    finally:
        client.close()

def classify(samples_number=10):
    logger = logging.getLogger("tgml")
    logger.setLevel(logging.DEBUG)
    client = MongoClient('localhost', 27017)
    try:
        loader = MongoDBLoader(client)
        er_features = loader.load_features('ER', 10)
        ba_features = loader.load_features('BA', 10)
        cl_features = loader.load_features('CL', 10)

        real_features = loader.load_features('real_graph', 10)
        #classifier = svc_classifier(er_features, cl_features)
        #classifier = svc_classifier(ba_features, cl_features)
        classifier = svc_classifier(ba_features, er_features)
        for real_feature in real_features:
            tmp = [collection_to_list(real_feature)]
            res = classifier.predict_proba(tmp)
            print("Result: {0}".format(res))
    finally:
        client.close()

def svc_classifier(left, right):
    classifier = Pipeline(
        [('scaler', StandardScaler()),
         ('svc', SVC(probability=True))]
    )
    
    x = []
    y = []
    for feature_set in left:
        x.append(collection_to_list(feature_set))
        y.append(1)

    for feature_set in right:
        x.append(collection_to_list(feature_set))
        y.append(2)
    classifier.fit(x, y)
    return classifier

def calculate_characteristics(samples_number=10, item_size=20, type='model'):
    logger = logging.getLogger("tgml")
    logger.setLevel(logging.DEBUG)
    client = MongoClient('localhost', 27017)
    try:
        loader = MongoDBLoader(client)
        features = FeatureVector()
        if type != 'model':
            collection = client['twitter-crawler']['real_graph_features']
            for number in range(6, samples_number):
                print(number)
                tmp_graph = get_giant_component(loader.load_one(item_size, number + 1))
                overview(tmp_graph)
                collection.insert_one(features.build_vector_for_graph(tmp_graph))
        else:
            node_count = 20000
            d = []
            for number in range(samples_number):
                graph = loader.load_one(20, number + 1)
                d.append([graph.degree(v) for v in graph.nodes()])

            gens = [
            	BAGenerator(node_count),
                ERGenerator(node_count)
            ]
            for model in gens:
                collection = client['twitter-crawler'][model.get_name() + '_features']
                for number in range(samples_number):
                    print(number)
                    tmp_graph = get_giant_component(model.generate())
                    overview(tmp_graph)
                    collection.insert_one(features.build_vector_for_graph(tmp_graph))

            # 10
            collection = client['twitter-crawler']['CL_features']
            for number in range(6, samples_number):
                print(number)
                model = CLGenerator(d[number])
                tmp_graph = get_giant_component(model.generate())
                overview(tmp_graph)
                collection.insert_one(features.build_vector_for_graph(tmp_graph))
    finally:
        client.close()

def check_scale_free():
    logger = logging.getLogger("tgml")
    logger.setLevel(logging.DEBUG)

    client = MongoClient('localhost', 27017)

    try:
        graph = MongoDBLoader(client).load_full_graph()
        is_scale_free(graph)
        plot_degree_distribution(graph)
    finally:
        client.close()


def is_scale_free(graph):
    dd = centrality.DegreeCentrality(graph).run().scores()
    fit = powerlaw.Fit(dd)
    res_exp, _ = fit.distribution_compare('power_law', 'exponential')
    res_trunc, _ = fit.distribution_compare('power_law', 'truncated_power_law')
    res_log, _ = fit.distribution_compare('power_law', 'lognormal')

    print(res_exp)
    print(res_trunc)
    print(res_log)

    mes = "Not Scale Free"
    assert res_exp > 0, mes
    assert res_trunc > 0, mes
    assert res_log > 0, mes

    print('Scale Free')


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
    # check_scale_free()
    #calculate_characteristics(samples_number=12)
    # classify_in_flight()
    # test_file()
