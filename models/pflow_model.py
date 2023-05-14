import torch.nn as nn
import dgl
import torch



def build_net(dims, activation=None):
    layers = []
    for i in range(len(dims)-2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(nn.LeakyReLU())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if activation is not None:
        layers.append(activation)
    return nn.Sequential(*layers)



class PflowModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.epsilon = 1e-8
        self.config = config

        h_dim = config['output_model']['h_dim']

        # will be used to update the nodes
        self.net1 = build_net([3] + config['output_model']['node_net_layers'], activation=nn.LeakyReLU())
        self.net2 = build_net(config['output_model']['node_net_layers'], activation=nn.LeakyReLU())
        self.net3 = build_net(config['output_model']['node_net_layers'], activation=nn.LeakyReLU())
        self.net4 = build_net(config['output_model']['node_net_layers'], activation=nn.LeakyReLU())
        self.net5 = build_net(config['output_model']['node_net_layers'], activation=nn.LeakyReLU())

        # will be used to predict the energy
        self.outputnet = build_net(config['output_model']['energy_net_layers'])

    def forward(self, g,):

        # update the node features
        g.ndata['feat'] = self.net1(g.ndata['feat'])
        global_feat = dgl.mean_nodes(g, 'feat')
        global_feat_bc = dgl.broadcast_nodes(g, global_feat)
        g.ndata['feat'] = g.ndata['feat'] + global_feat_bc

        g.ndata['feat'] = self.net2(g.ndata['feat'])
        global_feat = dgl.mean_nodes(g, 'feat')
        global_feat_bc = dgl.broadcast_nodes(g, global_feat)
        g.ndata['feat'] = g.ndata['feat'] + global_feat_bc

        g.ndata['feat'] = self.net3(g.ndata['feat'])
        global_feat = dgl.mean_nodes(g, 'feat')
        global_feat_bc = dgl.broadcast_nodes(g, global_feat)
        g.ndata['feat'] = g.ndata['feat'] + global_feat_bc

        g.ndata['feat'] = self.net4(g.ndata['feat'])
        global_feat = dgl.mean_nodes(g, 'feat')
        global_feat_bc = dgl.broadcast_nodes(g, global_feat)
        g.ndata['feat'] = g.ndata['feat'] + global_feat_bc

        g.ndata['feat'] = self.net5(g.ndata['feat'])
        global_feat = dgl.mean_nodes(g, 'feat')

        # predict the energy
        energy = self.outputnet(global_feat)

        return energy.squeeze(-1)
