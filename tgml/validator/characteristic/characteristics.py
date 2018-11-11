from _NetworKit import Graph, Diameter, EffectiveDiameter
import powerlaw
import networkit
import logging
from scipy import stats
import numpy


class CharacteristicVector:

    def __init__(self):
        self._characteristics = [
            NodesCountCharacteristic,
            VertexesCountCharacteristic,
            DiameterCharacteristic,
            EffectiveDiameterCharacteristic,
            PowerlawAlphaCharacteristic,
            ClusteringCentralityCharacteristic,
            BetweennessCentralityCharacteristic,
            ClosenessCentralityCharacteristic,
            KatzCentralityCharacteristic,
            PageRankCentralityCharacteristic
        ]

    @property
    def get_characteristics(self):
        return self._characteristics

    def build_vector_for_graph(self, graph: Graph):
        vector = dict()
        for characteristic in self.get_characteristics:
            vector[characteristic.get_name] = characteristic.get_value(graph)
        return vector


class AbstractCharacteristic:

    logger = logging.getLogger("tgml.validator.characteristic.characteristics")

    def __init__(self, characteristic_name):
        self._name = characteristic_name

    @property
    def get_name(self):
        return self._name

    def process_centrality(self, centrality):
        res = []
        # коэффициент эксцесса
        res.append(stats.kurtosis(centrality))
        # коэффициент ассиметрии (skewness)
        res.append(stats.skew(centrality))
        res.append(numpy.max(centrality))
        res.append(numpy.min(centrality))
        res.append(numpy.median(centrality))
        res.append(numpy.average(centrality))
        #1st and 3rd quartiles
        quartiles = numpy.percentile(centrality, [25, 75])
        res.append(quartiles[0])
        res.append(quartiles[1])
        res.append(numpy.std(centrality))
        res.append(numpy.var(centrality))
        return res

class NodesCountCharacteristic(AbstractCharacteristic):

    def __init__(self):
        super(NodesCountCharacteristic, self).__init__('n')

    def get_value(self, graph: Graph):
        super(NodesCountCharacteristic, self).logger.debug('Calculate node count')
        return graph.numberOfNodes()


class VertexesCountCharacteristic(AbstractCharacteristic):

    def __init__(self):
        super(VertexesCountCharacteristic, self).__init__('vertex_count')

    def get_value(self, graph: Graph):
        super(VertexesCountCharacteristic, self).logger.debug('Calculate vertex count')
        return graph.numberOfEdges()


class DiameterCharacteristic(AbstractCharacteristic):

    def __init__(self):
        super(DiameterCharacteristic, self).__init__('diameter')

    def get_value(self, graph: Graph):
        super(DiameterCharacteristic, self).logger.debug('Calculate diameter')
        return Diameter(graph).getDiameter()


class EffectiveDiameterCharacteristic(AbstractCharacteristic):

    def __init__(self):
        super(EffectiveDiameterCharacteristic, self).__init__('diameter')

    def get_value(self, graph: Graph):
        super(EffectiveDiameterCharacteristic, self).logger.debug('Calculate effective diameter')
        return EffectiveDiameter(graph).run().getEffectiveDiameter()


class PowerlawAlphaCharacteristic(AbstractCharacteristic):

    def __init__(self):
        super(PowerlawAlphaCharacteristic, self).__init__('diameter')

    def get_value(self, graph: Graph):
        super(PowerlawAlphaCharacteristic, self).logger.debug('Calculate effective diameter')
        degrees = networkit.centrality.DegreeCentrality(graph).run().scores()
        return powerlaw.Fit(degrees).alpha


class ClusteringCentralityCharacteristic(AbstractCharacteristic):

    def __init__(self):
        super(ClusteringCentralityCharacteristic, self).__init__('clustering_centrality')

    def get_value(self, graph: Graph):
        return super(ClusteringCentralityCharacteristic, self)\
            .process_centrality(networkit.centrality.LocalClusteringCoefficient(graph).run().scores())


class BetweennessCentralityCharacteristic(AbstractCharacteristic):

    def __init__(self):
        super(BetweennessCentralityCharacteristic, self).__init__('betweenness_centrality')

    def get_value(self, graph: Graph):
        return super(BetweennessCentralityCharacteristic, self) \
            .process_centrality(networkit.centrality.Betweenness(graph).run().scores())


class ClosenessCentralityCharacteristic(AbstractCharacteristic):

    def __init__(self):
        super(ClosenessCentralityCharacteristic, self).__init__('closeness_centrality')

    def get_value(self, graph: Graph):
        return super(ClosenessCentralityCharacteristic, self) \
            .process_centrality(networkit.centrality.Closeness(graph).run().scores())


class KatzCentralityCharacteristic(AbstractCharacteristic):

    def __init__(self):
        super(KatzCentralityCharacteristic, self).__init__('katz_centrality')

    def get_value(self, graph: Graph):
        return super(KatzCentralityCharacteristic, self) \
            .process_centrality(networkit.centrality.KatzCentrality(graph).run().scores())


class PageRankCentralityCharacteristic(AbstractCharacteristic):

    def __init__(self):
        super(PageRankCentralityCharacteristic, self).__init__('page_rank_centrality')

    def get_value(self, graph: Graph):
        return super(PageRankCentralityCharacteristic, self) \
            .process_centrality(networkit.centrality.PageRank(graph).run().scores())