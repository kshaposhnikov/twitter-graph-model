from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

from util.graphhelper import get_giant_component
from validator.characteristics import CharacteristicVector
from validator.features import FeatureVector
from validator.generator.bagenerator import BAGenerator
from validator.generator.ergenerator import ERGenerator
from networkit import overview

import logging

class Classifier:

    logger = logging.getLogger("tgml.validator.features.FeatureVector")

    generators = [BAGenerator(), ERGenerator()]

    def __init__(self, node_count, edge_count, class_count=5):
        self.node_count = node_count
        self.edge_count = edge_count
        self.class_count = class_count

    def build_classifiers(self):
        vector = FeatureVector()
        target_vectors = list()
        # classifier = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
        classifier = SVC(probability=True)
        current_class = 0
        classes = list()
        for generator in self.generators:
            for i in range(self.class_count):
                self.logger.debug('Current class number: #{0}'.format(i))
                tmp_g = get_giant_component(generator.generate(self.node_count, self.edge_count))
                self.logger.debug(overview(tmp_g))
                target_vectors.append(vector.build_vector_for_graph_as_list(tmp_g))
                classes.append(current_class)
            current_class += 1
        classifier.fit(target_vectors, classes)
        return classifier
