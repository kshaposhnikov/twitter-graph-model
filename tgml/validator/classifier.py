import logging

from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from loader.mongodbloader import MongoDBLoader
from util.graphhelper import get_giant_component, collection_to_list
from validator.features import FeatureVector


class InflightClassifier:
    logger = logging.getLogger("tgml.validator.classifier.InflightClassifier")

    def __init__(self, generators, class_count, hub_count):
        self.class_count = class_count
        self.generators = generators
        self.hub_count = hub_count

    def classify(self):
        self.logger.info("Loading...")
        graph_set = MongoDBLoader().load_as_slice(size=self.class_count, item_size=self.hub_count)
        features = FeatureVector()

        classifier = self.build_svc_classifier()
        res = []
        for index, graph in enumerate(graph_set):
            self.logger.info("Components")
            component = get_giant_component(graph)
            vector = [features.build_vector_for_graph_as_list(component)]
            proba = classifier.predict_proba(vector)
            self.logger.debug("Result of classification for sample {0}: {1}", index, proba)
            res.append(proba)

            self.logger.info("****************************************")
        self.logger.info("Done")

        return res

    def build_svc_classifier(self):
        vector = FeatureVector()
        target_vectors = list()
        classifier = SVC(probability=True)
        classes = list()
        for generator in self.generators:
            for i in range(self.class_count):
                self.logger.debug('Current class number: #{0}'.format(i))
                tmp_g = get_giant_component(generator.generate())
                self.logger.debug(tmp_g.toString())
                target_vectors.append(vector.build_vector_for_graph_as_list(tmp_g))
                classes.append(generator.get_name())
        classifier.fit(target_vectors, classes)
        return classifier

    def cross_validate(self, real_graphs):
        """
        Deprecated
        :param real_graphs:
        :return:
        """
        vector = FeatureVector()
        target_vectors = list()
        classifier = SVC(probability=True)
        classes = list()
        for generator in self.generators:
            for i in range(self.class_count):
                self.logger.debug('Current class number: #{0}'.format(i))
                tmp_g = get_giant_component(generator.generate())
                self.logger.debug(tmp_g.toString())
                target_vectors.append(vector.build_vector_for_graph_as_list(tmp_g))
                classes.append(generator.get_name())

        assert len(real_graphs) != self.class_count, 'Size of real_graphs list should be equal to class_count. ' \
                                                     'Now real_graphs {0} and class_count {1}'.format(len(real_graphs),
                                                                                                      self.class_count)

        for real_graph in real_graphs:
            tmp_g = get_giant_component(real_graph)
            target_vectors.append(vector.build_vector_for_graph_as_list(tmp_g))
            classes.append('real_graph')

        cross_val_predict(classifier, target_vectors, classes, cv=3)


class PairClassifier:

    logger = logging.getLogger("tgml.validator.classifier.PairClassifier")

    def __init__(self, samples_number, left_model, right_model):
        self.samples_number = samples_number
        self.left_model = left_model
        self.right_model = right_model

    def classify(self):
        loader = MongoDBLoader()
        left_features = loader.load_features(self.left_model, self.samples_number)
        right_features = loader.load_features(self.right_model, self.samples_number)

        real_features = loader.load_features('real_graph', self.samples_number)
        classifier = self._svc_classifier(left_features, right_features)

        y = [collection_to_list(real_feature) for real_feature in real_features]
        res = classifier.predict_proba(y)
        self.logger.debug("Result: {0}".format(res))
        return res

    def _svc_classifier(self, left, right):
        classifier = Pipeline(
            [('scaler', StandardScaler()),
             ('svc', SVC(probability=True))]
        )

        x = []
        y = []
        for feature_set in left:
            x.append(collection_to_list(feature_set))
            y.append(1)

        for feature_set in right:
            x.append(collection_to_list(feature_set))
            y.append(2)
        classifier.fit(x, y)
        return classifier
