import powerlaw
from matplotlib import pyplot as plt
from networkit import centrality

from loader.mongodbloader import MongoDBLoader


class ScaleFreeChecker:

    def __init__(self):
        pass

    def check(self):
        graph = MongoDBLoader().load_full_graph()
        self._is_scale_free(graph)
        self._plot_degree_distribution(graph)

    def _is_scale_free(self, graph):
        dd = centrality.DegreeCentrality(graph).run().scores()
        fit = powerlaw.Fit(dd)
        res_exp, _ = fit.distribution_compare('power_law', 'exponential')
        res_trunc, _ = fit.distribution_compare('power_law', 'truncated_power_law')
        res_log, _ = fit.distribution_compare('power_law', 'lognormal')

        print(res_exp)
        print(res_trunc)
        print(res_log)

        mes = "Not Scale Free"
        assert res_exp > 0, mes
        assert res_trunc > 0, mes
        assert res_log > 0, mes

        print('Scale Free')

    def _plot_degree_distribution(self, graph):
        dd = sorted(centrality.DegreeCentrality(graph).run().scores(), reverse=True)
        plt.xscale("log")
        plt.xlabel("degree")
        plt.yscale("log")
        plt.ylabel("number of nodes")
        plt.plot(dd)
        plt.show()
        return dd
