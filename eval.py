import sys
paths = sys.path
for p in paths:
     if '.local' in p:
             paths.remove(p)

import uproot
import awkward as ak

sys.path.append('./models/')
sys.path.append('./utility/')

import glob
import tqdm
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasetloader import DHCalDataset, collate_fn
from lightning import PflowLightning

import matplotlib.pyplot as plt
from pathlib import Path


# Global feat
REDUCE_DS  = -1
BATCH_SIZE = 200




config_path = sys.argv[1]
with open(config_path, 'r') as fp:
     config = json.load(fp)
exp_key = config_path.split('/')[-2]



# find the checkpoint path
checkpoint_for_pred = config['checkpoint_for_pred']
if 'ckpt' in checkpoint_for_pred:
    checkpoint_path = checkpoint_for_pred
elif checkpoint_for_pred == 'last':
    checkpoint_path = os.path.join(
        config['base_root_dir'], config['name'], f"{config['comet_settings']['project_name']}", exp_key, 'checkpoints', 'last.ckpt')
elif checkpoint_for_pred == 'best':
    checkpoints = glob.glob(os.path.join(
        config['base_root_dir'], config['name'], f"{config['comet_settings']['project_name']}", exp_key, 'checkpoints', 'epoch*ckpt'))
    checkpoint_path = sorted(checkpoints, key=lambda x: float(x.split('=')[-1].split('.')[0]))[0]


net = PflowLightning(config)
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
net.load_state_dict(checkpoint['state_dict'])


torch.set_grad_enabled(False)
net.eval(); net.cuda()
device = torch.device('cuda')




for file_to_pred_on in [config['test_path'], config['train_path']]:

    ds = DHCalDataset(file_to_pred_on, reduce_ds=REDUCE_DS, config=config)
    loader = DataLoader(ds, batch_size=BATCH_SIZE,
        num_workers=config['num_workers'], shuffle=False, collate_fn=collate_fn)


    pred_es = []; target_es = []

    for batch in tqdm.tqdm(loader):

        batched_g, target_e = batch
        batched_g = batched_g.to(device)
        pred_e = net(batched_g)

        # undo scaling
        pred_e = pred_e * (config['var_transform']['eBeam']['max'] - config['var_transform']['eBeam']['min']) + config['var_transform']['eBeam']['min']
        target_e = target_e * (config['var_transform']['eBeam']['max'] - config['var_transform']['eBeam']['min']) + config['var_transform']['eBeam']['min']

        pred_es.extend(pred_e.cpu().detach().numpy().tolist())
        target_es.extend(target_e.cpu().detach().numpy().tolist())



    # writing to file
    #-------------------------

    fname = file_to_pred_on.split('/')[-1].replace('.root','_') + checkpoint_path.split("/")[-1].replace("ckpt","root")
    eval_path = f'{config["eval_root_dir"]}/{config["name"]}/{exp_key}/{fname}'
    Path(eval_path).parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(eval_path) as file:
        file["ML_Tree"] = {
            "eBeam_target": np.array(target_es),
            "eBeam_pred": np.array(pred_es)
        }
        file["ML_Tree"].show()

    print(f"Predictions saved to {eval_path}")

