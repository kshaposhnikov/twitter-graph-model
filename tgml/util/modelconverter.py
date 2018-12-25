from pymongo.cursor import Cursor


class ModelConverter:

    def convert(self, source: Cursor, storage):
        existing_nodes = dict()
        for node in source:
            if node['id'] in existing_nodes:
                start_number = existing_nodes[node['id']]
            else:
                start_number = len(existing_nodes)
                existing_nodes[node['id']] = start_number

            leafs = list()
            for leaf in node['associatednodes']:
                if leaf in existing_nodes:
                    leafs.append(existing_nodes[leaf])
                else:
                    tmp = len(existing_nodes)
                    existing_nodes[leaf] = tmp
                    leafs.append(tmp)

            new_node = {
                'id': start_number,
                'associatednodescount': node['associatednodescount'],
                'associatednodes': leafs
            }

            storage.store(new_node)
