import logging
from _NetworKit import Graph, Diameter, EffectiveDiameter

import networkit
import numpy
import powerlaw
from scipy import stats


class FeatureVector:

    logger = logging.getLogger("tgml.validator.features.FeatureVector")

    def __init__(self):
        self._features = [
            NodesCountFeature(),
            VertexesCountFeature(),
            DiameterFeature(),
            EffectiveDiameterFeature(),
            PowerlawAlphaFeature(),
            ClusteringCentralityFeature(),
            BetweennessCentralityFeature(),
            ClosenessCentralityFeature(),
            KatzCentralityFeature(),
            PageRankCentralityFeature()
        ]

    @property
    def get_features(self):
        return self._features

    def build_vector_for_graph(self, graph: Graph):
        vector = dict()
        for feature in self.get_features:
            vector[feature.get_name] = feature.get_value(graph)
        return vector

    def build_vector_for_graph_as_list(self, graph: Graph):
        vector = list()
        self.logger.debug('Start vector building:')
        for feature in self.get_features:
            vector.extend(feature.get_value(graph))
        self.logger.debug('Done.')
        return vector


class AbstractFeature:

    logger = logging.getLogger("tgml.validator.features")

    def __init__(self, characteristic_name):
        self._name = characteristic_name

    @property
    def get_name(self):
        return self._name

    def process_centrality(self, centrality):
        # 1st and 3rd quartiles
        quartiles = numpy.percentile(centrality, [25, 75])
        return [
            # коэффициент эксцесса
            stats.kurtosis(centrality),
            # коэффициент ассиметрии (skewness)
            stats.skew(centrality),
            numpy.max(centrality),
            numpy.min(centrality),
            numpy.median(centrality),
            numpy.average(centrality),
            quartiles[0],
            quartiles[1],
            numpy.std(centrality),
            numpy.var(centrality)
        ]

    def get_value(self, graph: Graph):
        pass


class NodesCountFeature(AbstractFeature):

    def __init__(self):
        super(NodesCountFeature, self).__init__('n')

    def get_value(self, graph: Graph):
        self.logger.debug('Calculate node count')
        return [graph.numberOfNodes()]


class VertexesCountFeature(AbstractFeature):

    def __init__(self):
        super(VertexesCountFeature, self).__init__('vertex_count')

    def get_value(self, graph: Graph):
        self.logger.debug('Calculate vertex count')
        return [graph.numberOfEdges()]


class DiameterFeature(AbstractFeature):

    def __init__(self):
        super(DiameterFeature, self).__init__('diameter')

    def get_value(self, graph: Graph):
        self.logger.debug('Calculate diameter')
        return Diameter(graph).getDiameter()


class EffectiveDiameterFeature(AbstractFeature):

    def __init__(self):
        super(EffectiveDiameterFeature, self).__init__('diameter')

    def get_value(self, graph: Graph):
        self.logger.debug('Calculate effective diameter')
        return [EffectiveDiameter(graph).run().getEffectiveDiameter()]


class PowerlawAlphaFeature(AbstractFeature):

    def __init__(self):
        super(PowerlawAlphaFeature, self).__init__('diameter')

    def get_value(self, graph: Graph):
        self.logger.debug('Calculate powerlaw')
        degrees = networkit.centrality.DegreeCentrality(graph).run().scores()
        return [powerlaw.Fit(degrees).alpha]


class ClusteringCentralityFeature(AbstractFeature):

    def __init__(self):
        super(ClusteringCentralityFeature, self).__init__('clustering_centrality')

    def get_value(self, graph: Graph):
        self.logger.debug('calculate clustering centrality')
        return self.process_centrality(
            networkit.centrality.LocalClusteringCoefficient(graph).run().scores()
        )


class BetweennessCentralityFeature(AbstractFeature):

    def __init__(self):
        super(BetweennessCentralityFeature, self).__init__('betweenness_centrality')

    def get_value(self, graph: Graph):
        self.logger.debug('calculate betweeness centrality')
        return self.process_centrality(
            networkit.centrality.Betweenness(graph).run().scores()
        )


class ClosenessCentralityFeature(AbstractFeature):

    def __init__(self):
        super(ClosenessCentralityFeature, self).__init__('closeness_centrality')

    def get_value(self, graph: Graph):
        self.logger.debug('calculate closeness centrality')
        return self.process_centrality(
            networkit.centrality.Closeness(graph).run().scores()
        )


class KatzCentralityFeature(AbstractFeature):

    def __init__(self):
        super(KatzCentralityFeature, self).__init__('katz_centrality')

    def get_value(self, graph: Graph):
        self.logger.debug('calculate katz centrality')
        return self.process_centrality(
            networkit.centrality.KatzCentrality(graph).run().scores()
        )


class PageRankCentralityFeature(AbstractFeature):

    def __init__(self):
        super(PageRankCentralityFeature, self).__init__('page_rank_centrality')

    def get_value(self, graph: Graph):
        self.logger.debug('calculate page rank centrality')
        return self.process_centrality(
            networkit.centrality.PageRank(graph).run().scores()
        )
