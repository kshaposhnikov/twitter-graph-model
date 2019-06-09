import logging

import plotly.offline as py
import networkit
import networkx as kx
from networkit import nxadapter
import plotly.graph_objs as go


class GraphVisualizer:

    logger = logging.getLogger("tgml.GraphVisualizer")

    def __init__(self):
        pass

    def save_for_gephi(self, file_path, graph: networkit.Graph):
        nk_graph = nxadapter.nk2nx(graph)
        kx.write_gexf(nk_graph, file_path)

    def visualize(self, title, graph: networkit.Graph):
        self.logger.debug("Convert networkit to networkx...")
        nk_graph = nxadapter.nk2nx(graph)
        self.logger.debug("Calculate positions.....")
        pos = kx.spring_layout(nk_graph)
        self.logger.debug("Build edge vector .........")
        edges_data = self._edge_vector(nk_graph, pos)
        self.logger.debug("Build node vector ................")
        nodes_data = self._node_vector(nk_graph, pos)

        fig = go.Figure(data=[edges_data, nodes_data],
                        layout=go.Layout(
                            title='<br>' + title,
                            titlefont=dict(size=16),
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            annotations=[dict(
                                #text="Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> https://plot.ly/ipython-notebooks/network-graphs/</a>",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002)],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

        self.logger.debug("Draw ........ .........")
        py.plot(fig, filename=title + '.html')

    def _edge_vector(self, graph, pos):
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        i = 0
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])
            if i % 100 == 0:
                self.logger.debug(i)
            i+=1

        return edge_trace

    def _node_vector(self, graph, pos):
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                # colorscale options
                # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2)))

        for node in graph.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])

        for node, adjacencies in enumerate(graph.adjacency()):
            node_trace['marker']['color'] += tuple([len(adjacencies[1])])
            node_info = '# of connections: ' + str(len(adjacencies[1]))
            node_trace['text'] += tuple([node_info])

        return node_trace
