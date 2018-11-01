from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from tgml.validator.characteristics import CharacteristicVector
from tgml.validator.generator.bagenerator import BAGenerator
from tgml.validator.generator.ergenerator import ERGenerator


class Classifier:
    generators = [BAGenerator(), ERGenerator()]

    def __init__(self, node_count, edge_count, class_count=5):
        self.node_count = node_count
        self.edge_count = edge_count
        self.class_count = class_count

    def build_classifier(self):
        builder = CharacteristicVector()
        target_vectors = list()
        classifier = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
        current_class = 0
        classes = list()
        for generator in self.generators:
            for i in range(self.class_count):
                tmp_g = generator.generate(self.node_count, self.edge_count)
                target_vectors.append(builder.build_vector(tmp_g))
                classes.append(current_class)
            current_class += 1
        classifier.fit(target_vectors, current_class)
        return classifier
