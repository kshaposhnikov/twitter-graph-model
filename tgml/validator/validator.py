import networkit

from generator.bagenerator import BAGenerator
from src.validator.generator.ergenerator import ERGenerator


class Validator:

    generators = [BAGenerator, ERGenerator]

    def __init__(self, node_count, edge_count):
        self.node_count = node_count
        self.edge_count = edge_count

    def validate(self):
        for generator in self.generators:
            tmp_g = generator.generate(self.node_count, self.edge_count)


