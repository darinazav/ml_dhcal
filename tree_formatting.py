import uproot
import numpy as np
import sys
import awkward as ak



def reformat_tree(filepath):
    f = uproot.open(filepath)
    tree = f['tvec']

    scalar_branches = ['eBeam', 'pBeam']
    vector_branches = ['layer', 'xpos', 'ypos', 'digit']
    branches = ['eventId'] + scalar_branches + vector_branches

    # read the tree
    data = {}
    for br in branches:
        data[br] = tree[br].array(library='np')

    # create the dictionaries for the new tree
    data_new_vec = {}
    for br in vector_branches:
        data_new_vec[br] = []

    data_new = {}
    for br in scalar_branches:
            data_new[br] = []

    # not optimized, but okay !!
    ev_ids = np.unique(data['eventId'])
    for ev_id in ev_ids:
        ev_mask = data['eventId'] == ev_id
        for br in vector_branches:
            data_new_vec[br].append(data[br][ev_mask].tolist())
        for br in scalar_branches:
            data_new[br].append(data[br][ev_mask][0])

    # add the vector branches to data_new
    data_new['hit'] = ak.zip(data_new_vec)

    new_filepath = filepath.replace('.root', '_formatted.root')
    with uproot.recreate(new_filepath) as file:
        file['digi_tree'] = data_new
        file['digi_tree'].show()


if __name__ == '__main__':
    filepath = sys.argv[1]
    reformat_tree(filepath)