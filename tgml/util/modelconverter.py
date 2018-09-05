from pymongo import MongoClient


def convert():
    client = MongoClient('localhost', 27017)
    db = client['twitter-crawler']

    try:
        existing_nodes = dict()
        for node in db['graph'].find():
            if node['id'] in existing_nodes:
                start_number = existing_nodes[node['id']]
            else:
                start_number = len(existing_nodes)
                existing_nodes[node['id']] = start_number

            new_node = {'id': start_number, 'associatednodescount': node['associatednodescount']}

            leafs = list()
            for leaf in node['associatednodes']:
                if leaf in existing_nodes:
                    leafs.append(existing_nodes[leaf])
                else:
                    tmp = len(existing_nodes)
                    existing_nodes[leaf] = tmp
                    leafs.append(tmp)

            new_node['associatednodes'] = leafs

            db['graph2'].insert_one(new_node)
    finally:
        client.close()
