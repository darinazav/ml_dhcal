import uproot
import numpy as np
import torch
import dgl

from torch.utils.data import Dataset


def collate_fn(samples):
    batched_graphs = dgl.batch([x[0] for x in samples])
    targets_e = torch.stack([x[1] for x in samples])
    return batched_graphs, targets_e


class DHCalDataset(Dataset):
    def __init__(self, filename, config=None, reduce_ds=-1):

        # reading the dataset
        self.config = config
        self.var_transform = self.config['var_transform']

        # open the file and access the tree
        self.f = uproot.open(filename)
        self.tree = self.f['digi_tree']

        # check the number of events; update it if we want to use less events
        self.nevents = self.tree.num_entries
        if reduce_ds != -1:
            if reduce_ds < 1:
                self.nevents = int(self.nevents * reduce_ds)
            else:
                self.nevents = reduce_ds

        # read the branches and store them in a dictionary. It's easier to work this way
        self.data_dict = {}
        var_list_scalar = ['nhit', 'eBeam']
        var_list_vec = ['hit_xpos', 'hit_ypos', 'hit_layer']

        for var in var_list_scalar + var_list_vec:
            self.data_dict[var] = self.tree[var].array(library='np',entry_stop=self.nevents)
            
            # let's covert them to torch tensors and also flatten them
            if var in var_list_vec:
                self.data_dict[var] = torch.FloatTensor(np.hstack(self.data_dict[var]))
            else:
                self.data_dict[var] = torch.FloatTensor(self.data_dict[var])

            # scale the variables that we want to scale
            if var in self.var_transform.keys():
                if var == "eBeam":
                    self.data_dict[var] = (self.data_dict[var] - self.var_transform[var]['min']) /\
                                            (self.var_transform[var]['max'] - self.var_transform[var]['min'])
                else:
                    self.data_dict[var] = (self.data_dict[var] - self.var_transform[var]['mean'])/self.var_transform[var]['std']


        # since we are falttenting above, we need to cumsum to get the start and end indices for each event
        self.cumsum  = np.cumsum([0] + self.data_dict['nhit'].int().tolist())

        print(f'done loading data ({self.nevents} events)')



    def __getitem__(self, idx):

        start,  end  = self.cumsum[idx],  self.cumsum[idx+1]

        n_hit = int(self.data_dict['nhit'][idx].item())
        nodes = torch.arange(n_hit)

        # create the graph (No edges at this point. we'll add them later)
        g = dgl.graph((nodes, nodes))

        # add the node features (first two are just technical stuff)
        g.ndata['_ID']   = nodes
        g.ndata['_TYPE'] = torch.zeros(n_hit, dtype=torch.long)

        # the physics features
        g.ndata['x'] = self.data_dict['hit_xpos'][start:end]
        g.ndata['y'] = self.data_dict['hit_ypos'][start:end]
        g.ndata['z'] = self.data_dict['hit_layer'][start:end]

        g.ndata['feat'] = torch.stack([
            g.ndata['x'], g.ndata['y'], g.ndata['z']
        ], dim=1)

        return g, self.data_dict['eBeam'][idx]


    def __len__(self):
        return self.nevents 

    