from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict

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

    def build_svc_classifier(self):
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
                self.logger.debug(tmp_g.toString())
                target_vectors.append(vector.build_vector_for_graph_as_list(tmp_g))
                classes.append(current_class)
            current_class += 1
        classifier.fit(target_vectors, classes)
        return classifier


    def cross_vlidate(self, real_graphs):
        vector = FeatureVector()
        target_vectors = list()
        classifier = SVC(probability=True)
        classes = list()
        for generator in self.generators:
            for i in range(self.class_count):
                self.logger.debug('Current class number: #{0}'.format(i))
                tmp_g = get_giant_component(generator.generate(self.node_count, self.edge_count))
                self.logger.debug(tmp_g.toString())
                target_vectors.append(vector.build_vector_for_graph_as_list(tmp_g))
                classes.append(generator.get_name())
        
        assert len(real_graphs) != self.class_count, 'Size of real_graphs list should be equal to class_count. ' \ 
                                                     'Now real_graphs {0} and class_count {1}'.format(len(real_graphs), self.class_count)

        for real_graph in real_graphs:
            tmp_g = get_giant_component(real_graph)
            target_vectors.append(vector.build_vector_for_graph_as_list(tmp_g))
            classes.append('real_graph')

        cross_val_predict(classifier, target_vectors, classes, cv=3)