import time

import networkit


class CLGenerator:

    def __init__(self, deg_sequence):
        self.deg_sequence = deg_sequence

    def generate(self):
        #networkit.setSeed(seed=time.clock(), useThreadId=True)
        return networkit.generators.ChungLuGenerator(self.deg_sequence).generate()

    def get_name(self):
        return 'CL'