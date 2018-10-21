import networkit

from tgml.validator.characteristics import CharacteristicVector
from tgml.validator.generator.bagenerator import BAGenerator
from tgml.validator.generator.ergenerator import ERGenerator


class Validator:

    generators = [BAGenerator(), ERGenerator()]

    def __init__(self, node_count, edge_count):
        self.node_count = node_count
        self.edge_count = edge_count

    def validate(self):
        builder = CharacteristicVector()
        for generator in self.generators:
            tmp_g = generator.generate(self.node_count, self.edge_count)
            res = builder.build_vector(tmp_g)
            print(res)



