import networkit


class ERGenerator:

    def generate(self, node_count, edge_count):
        return networkit.generators.ErdosRenyiGenerator(node_count, 0.1).generate()
