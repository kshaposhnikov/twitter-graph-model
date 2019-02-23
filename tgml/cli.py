import click
from networkit import overview

from loader.MongoGateway import gateway
from loader.mongodbloader import MongoDBLoader
from util.graphhelper import get_giant_component
from validator.features import FeatureVector
from validator.generator.bagenerator import BAGenerator
from validator.generator.chunglugenerator import CLGenerator
from validator.generator.ergenerator import ERGenerator
from validator.classifier import InflightClassifier, PairClassifier
from validator.scalefree import ScaleFreeChecker


@click.group(help='Graph Features')
@click.option('--sample_count', default=10, help='Number of samples to calculate features.')
@click.option('--start_position', default=0, help="Start position in feature's vector")
def model_features():
    pass


@model_features.command(help='Gather Bollobasi-Riordan Feature')
@click.option('--sample_count', default=10, help='Number of samples to calculate features.')
@click.option('--start_position', default=0, help="Start position in feature's vector")
@click.option('--nodes_count', required=True, default=79999, help='Number of nodes in random graph.')
def gather_br_features(sample_count, start_position, nodes_count=79999):
    features_collection = gateway.get_collection("BR_features")
    br_collection = gateway.get_collection("bollobas_riordan_30000")
    loader = MongoDBLoader()
    features = FeatureVector()
    for number in range(start_position, sample_count):
        graph = loader.load_one_from_collection(number, br_collection)
        component = get_giant_component(graph)
        component.removeSelfLoops()
        overview(component)
        #features.get_features[9].get_value(component)
        features_collection.insert_one(features.build_vector_for_graph(component))


@model_features.command(help='Gather Erdos-Renyi Feature')
@click.option('--nodes_count', required=True, help='Number of nodes in random graph.')
def gather_er_features(sample_count, node_count, start_position):
    gather_features(sample_count, start_position, ERGenerator(node_count))


@model_features.command(help='Gather Barabasi-Albert Features')
@click.option('--nodes_count', required=True, help='Number of nodes in random graph.')
def gather_ba_features(sample_count, node_count, start_position):
    gather_features(sample_count, start_position, BAGenerator(node_count))


@model_features.command(help='Gather Chung-Lu Features')
@click.option('--nodes_count', required=True, help='Number of nodes in random graph.')
def gather_cl_features(sample_count, node_count, start_position):
    d = []
    loader = MongoDBLoader()
    for number in range(sample_count):
        graph = loader.load_real_graph_part(node_count, number + 1)
        d.append([graph.degree(v) for v in graph.nodes()])

    feature_vector = FeatureVector()
    for number in range(start_position, sample_count):
        generator = CLGenerator(d[number])
        collection = gateway.get_collection(generator.get_name() + '_features')
        component = get_giant_component(generator.generate())
        overview(component)
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
    collection = gateway.get_collection(generator.get_name() + '_features')
    feature_vector = FeatureVector()
    for number in range(start_position, sample_count):
        component = get_giant_component(generator.generate())
        overview(component)
        collection.insert_one(feature_vector.build_vector_for_graph(component))


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


commands = click.CommandCollection(sources=[model_features, classify, util])
