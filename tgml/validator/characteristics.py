from _NetworKit import Graph, Diameter, EffectiveDiameter
import powerlaw
import networkit
import logging

class CharacteristicVector:

    def build_vector(self, graph: Graph):
        logger = logging.getLogger("tgml.characteristics.build_vector")
        
        degrees = networkit.centrality.DegreeCentrality(graph).run().scores()

        vector = dict()

        logger.debug("Size...")
        vector['n'] = graph.numberOfNodes()
        vector['m'] = graph.numberOfEdges()

        logger.debug("Diameter...")
        vector['d'] = Diameter(graph).getDiameter()

        logger.debug("Effective Diameter...")
        vector['ed'] = EffectiveDiameter(graph).run().getEffectiveDiameter()

        logger.debug("Alpha...")
        vector['a'] = powerlaw.Fit(degrees).alpha

        logger.debug("Local Clustering Centrality...")
        vector['clust_centrality'] = networkit.centrality.LocalClusteringCoefficient(graph).run().scores()

        logger.debug("Betweenness Centrality..")
        vector['bet_centrality'] = networkit.centrality.Betweenness(graph).run().scores()

        logger.debug("Closenness Centrality...")
        vector['clos_centrality'] = networkit.centrality.Closeness(graph).run().scores()

        logger.debug("Katz Centrality...")
        vector['katz_centrality'] = networkit.centrality.KatzCentrality(graph).run().scores()

        logger.debug("Page Rank...")
        vector['page_rank'] = networkit.centrality.PageRank(graph).run().scores()
        return vector
