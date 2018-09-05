from _NetworKit import Graph, Diameter, EffectiveDiameter
import powerlaw
import networkit


class CharacterisitcVector:

    def build_vector(self, graph: Graph):
        degrees = networkit.centrality.DegreeCentrality(graph).run().scores()

        return [graph.numberOfNodes(),
                graph.numberOfEdges(),
                Diameter(graph).getDiameter(),
                EffectiveDiameter(graph).getEffectiveDiameter(),
                powerlaw.Fit(degrees).alpha
                ]
