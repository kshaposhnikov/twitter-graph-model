import time

import networkit


class ERGenerator:

    def __init__(self, node_count):
        self.node_count = node_count

    def generate(self):
        networkit.setSeed(seed=time.clock(), useThreadId=False)
        return networkit.generators.ErdosRenyiGenerator(self.node_count, 0.0002).generate()


    def get_name(self):
    	return 'ER'
