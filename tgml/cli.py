import click
from networkit import overview

from graphvisualizer import GraphVisualizer
from loader.MongoGateway import gateway
from loader.mongodbloader import MongoDBLoader
from loader.mongodbstorage import MongoDBStorage
from util.graphhelper import get_giant_component
from validator.features import FeatureVector
from validator.generator.bagenerator import BAGenerator
from validator.generator.chunglugenerator import CLGenerator
from validator.generator.ergenerator import ERGenerator
from validator.classifier import InflightClassifier, PairClassifier
from validator.scalefree import ScaleFreeChecker

feature_collection_suffix = '_features'

@click.group(help='Graph Features')
@click.option('--sample_count', default=10, help='Number of samples to calculate features.')
@click.option('--start_position', default=0, help="Start position in feature's vector")
def model_features():
    pass


@model_features.command(help='Gather Features for Graph craeted by external generator')
@click.option('--prefix', help='Prefix for Mongo Collection')
@click.option('--sample_count', default=10, help='Number of samples to calculate features.')
@click.option('--start_position', default=0, help="Start position in feature's vector")
@click.option('--nodes_count', required=True, default=79999, help='Number of nodes in random graph.')
def gather_external_features(prefix, sample_count, start_position, nodes_count=79999):
    features_collection = gateway.get_collection(prefix + "_features")
    br_collection = gateway.get_collection(prefix + "_graphs")
    loader = MongoDBLoader()
    features = FeatureVector()
    for number in range(start_position, sample_count):
        graph = loader.load_one_from_collection(number, br_collection)
        component = get_giant_component(graph)
        component.removeSelfLoops()
        overview(component)
        features_collection.insert_one(features.build_vector_for_graph(component))


@model_features.command(help='Gather Erdos-Renyi Feature')
@click.option('--node_count', default=20000, required=True, help='Number of nodes in random graph.')
@click.option('--sample_count', default=10, help='Number of samples to calculate features.')
@click.option('--start_position', default=0, help="Start position in feature's vector")
def gather_er_features(sample_count, node_count, start_position):
    gather_features(sample_count, start_position, ERGenerator(node_count))


@model_features.command(help='Gather Barabasi-Albert Features')
@click.option('--node_count', default=20000, required=True, help='Number of nodes in random graph.')
@click.option('--sample_count', default=10, help='Number of samples to calculate features.')
@click.option('--start_position', default=0, help="Start position in feature's vector")
def gather_ba_features(sample_count, node_count, start_position):
    gather_features(sample_count, start_position, BAGenerator(node_count))


@model_features.command(help='Gather Chung-Lu Features')
@click.option('--node_count', default=20, required=True, help='Number of nodes in random graph.')
@click.option('--sample_count', default=10, help='Number of samples to calculate features.')
@click.option('--start_position', default=0, help="Start position in feature's vector")
def gather_cl_features(sample_count, node_count, start_position):
    d = []
    loader = MongoDBLoader()
    for number in range(sample_count):
        graph = loader.load_real_graph_part(node_count, number + 1)
        d.append([graph.degree(v) for v in graph.nodes()])

    feature_vector = FeatureVector()
    for number in range(start_position, sample_count):
        generator = CLGenerator(d[number])
        collection = gateway.get_collection(generator.get_name() + feature_collection_suffix)
        collection_graph = gateway.get_collection(generator.get_name() + '_graphs')
        component = get_giant_component(generator.generate())
        overview(component)
        MongoDBStorage().storeGraph(collection_graph, component)
        collection.insert_one(feature_vector.build_vector_for_graph(component))


@model_features.command(help='Gather Features for Real Graph')
@click.option('--nodes_count', required=True, help='Number of hubs in real graph.')
def gather_real_features(sample_count, node_count, start_position):
    collection = gateway.real_features()
    loader = MongoDBLoader()
    features = FeatureVector()
    for number in range(start_position, sample_count):
        print(number)
        component = get_giant_component(loader.load_real_graph_part(node_count, number + 1))
        overview(component)
        collection.insert_one(features.build_vector_for_graph(component))


def gather_features(sample_count, start_position, generator):
    collection_features = gateway.get_collection(generator.get_name() + feature_collection_suffix)
    collection_graph = gateway.get_collection(generator.get_name() + '_graphs')
    feature_vector = FeatureVector()
    for number in range(start_position, sample_count):
        component = get_giant_component(generator.generate())
        overview(component)
        MongoDBStorage().storeGraph(collection_graph, component)
        collection_features.insert_one(feature_vector.build_vector_for_graph(component))


@model_features.command(help='Calculate specified feature for specified graph')
@click.option('--feature', 'feature_name', required=True, help='Name of feature')
@click.option('--model', required=True, help='Name of model. Currently available: ER, BR, BA, CL, RG')
@click.option('--sample_count', default=10, help='Number of samples to calculate features.')
@click.option('--start_position', default=0, help="Start position in feature's vector")
def gather_specific_feature(feature_name, model, sample_count, start_position):
    vector = FeatureVector()

    if model == 'RG':
        feature_collection = gateway.real_features()
    else:
        feature_collection = gateway.get_collection(model + feature_collection_suffix)

    feature = vector.get_feature(feature_name)
    graph_collection = gateway.get_collection(model + '_graphs')

    loader = MongoDBLoader()
    for number in range(start_position, sample_count):
        if model == 'RG':
            graph = loader.load_real_graph_part(60, number + 1)
        else:
            graph = loader.load_one_from_collection(number, graph_collection)
        component = get_giant_component(graph)
        value = feature.get_value(component)

        document_id = feature_collection.find().skip(number).limit(1)[0]['_id']
        feature_collection.update({"_id": document_id}, {"$set": {feature_name: value}})


@click.group()
def classify():
    pass


@classify.command(help='Classify graphs in-flight')
@click.option('--samples_count', default=10, help='Number of samples to calculate features.')
@click.option('--node_count', required=True, help='Number of node count in random graphs.')
@click.option('--hub_count', default=10, help='Number of hubs in real graph.')
def classify_in_flight(samples_count, node_count, hub_count):
    graph_set = MongoDBLoader().load_as_slice(size=samples_count, item_size=hub_count)
    gens = [BAGenerator(node_count),
            ERGenerator(node_count),
            CLGenerator([graph_set[0].degree(v) for v in graph_set[0].nodes()])]
    res = InflightClassifier(gens, samples_count, hub_count).classify()
    print(res)


@classify.command(help='Match two models with real graph')
@click.option('--sample-count', default=10, help='Number of samples which will be loaded for both models')
@click.option('--left-model', help='Model name like ER, BA, CL')
@click.option('--right-model', help='Model name like ER, BA, CL')
def pair_classify(sample_count, left_model, right_model):
    res = PairClassifier(sample_count, left_model=left_model, right_model=right_model).classify()
    print(res)


@click.group()
def util():
    pass


@util.command(help='Verify Scale Free type for real graph')
def verify_scale_free():
    ScaleFreeChecker().check()


@util.command(help='Verify Scale Free type for real graph')
@click.option('--sample-number', default=0, help='Number of samples which will be loaded for both models')
@click.option('--name', help='Model name like ER, BA, CL')
def visualize_graph(name, sample_number):
    models = {
        'ER': 'Erdos-Renyi',
        'CL': 'Chung-Lu',
        'BA': 'Barabasi-Albert',
        'BR': 'Bollobas-Riordan',
        'BO': 'Buckley-Osthus',
        'YAM': 'Updated-Buckley-Osthus'
    }

    loader = MongoDBLoader()
    graph_collection = gateway.get_collection(name + '_graphs')
    graph = loader.load_one_from_collection(sample_number, graph_collection)
    for key, value in models.items():
        if key in name:
            GraphVisualizer().visualize(value, graph)
            #GraphVisualizer().save_for_gephi(name + '.gexf', graph)
            break


commands = click.CommandCollection(sources=[model_features, classify, util])
