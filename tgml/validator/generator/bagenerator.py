import time

import networkit


class BAGenerator:

    def __init__(self, node_count):
        self.node_count = node_count

    def generate(self):
        #networkit.setSeed(seed=time.clock(), useThreadId=True)
        return networkit.generators.BarabasiAlbertGenerator(1, self.node_count, 1).generate()

    def get_name(self):
    	return 'BA'
