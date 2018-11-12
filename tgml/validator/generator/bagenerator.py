import networkit


class BAGenerator:

    def generate(self, node_count, edge_count):
        return networkit.generators.BarabasiAlbertGenerator(1, node_count, 1).generate()

    def get_name(self):
    	return 'BA'
