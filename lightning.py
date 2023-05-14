
from pytorch_lightning.core.module import LightningModule

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import sys
sys.path.append('./models/')
from models.pflow_model import PflowModel
from datasetloader import DHCalDataset, collate_fn

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class PflowLightning(LightningModule):

    def __init__(self, config, comet_logger=None):
        super().__init__()

        self.save_hyperparameters()

        self.config = config
        self.net = PflowModel(self.config)

        self.comet_logger = comet_logger        


    def set_comet_logger(self, comet_logger):
        self.comet_logger = comet_logger


    def forward(self, batch):
        return self.net(batch)


    def train_dataloader(self):
        reduce_ds = self.config['reduce_ds_train']

        ds = DHCalDataset(self.config['train_path'], reduce_ds=reduce_ds, config=self.config)
        loader = DataLoader(ds, batch_size=self.config['batch_size'], 
            num_workers=self.config['num_workers'], shuffle=True, collate_fn=collate_fn)

        return loader

    
    def val_dataloader(self):
        reduce_ds = self.config['reduce_ds_val']

        ds = DHCalDataset(self.config['val_path'], reduce_ds=reduce_ds, config=self.config)
        loader = DataLoader(ds, batch_size=self.config['batch_size'], 
            num_workers=self.config['num_workers'], shuffle=False, collate_fn=collate_fn)

        return loader


    def training_step(self, batch, batch_idx):
        g, target_e  = batch

        pred_e = self.net(g)
        loss = F.mse_loss(pred_e, target_e)

        self.log('train/loss', loss.item(), batch_size=g.batch_size)
        return loss.mean()


    def validation_step(self, batch, batch_idx):

        g, target_e  = batch

        pred_e = self.net(g)
        loss = F.mse_loss(pred_e, target_e)

        return_dict = {'loss': loss, 'pred_e': pred_e, 'target_e': target_e}
        return return_dict


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config['learningrate'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.config['lr_scheduler']['T_max'], 
            eta_min=self.config['lr_scheduler']['eta_min'],
            last_epoch=self.config['lr_scheduler']['last_epoch'],
            verbose=True)

        # return optimizer
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


    def validation_epoch_end(self, outputs):
        
        val_loss = torch.hstack([x['loss'] for x in outputs]).mean()
        self.log('val_loss', val_loss.item())
        self.log('lr', self.lr_schedulers().get_lr()[0])

        if plt.get_fignums():
            plt.clf()
            fig = plt.gcf()
        else:
            fig = plt.figure(figsize=(5, 5), dpi=100, tight_layout=True)
        canvas = FigureCanvas(fig) 

        e_targets = torch.hstack([x['target_e'] for x in outputs]).detach().cpu().numpy()
        e_preds = torch.hstack([x['pred_e'] for x in outputs]).detach().cpu().numpy()
        e_targets = e_targets * (self.config['var_transform']['eBeam']['max'] - \
            self.config['var_transform']['eBeam']['min']) + self.config['var_transform']['eBeam']['min']
        e_preds = e_preds * (self.config['var_transform']['eBeam']['max'] - \
            self.config['var_transform']['eBeam']['min']) + self.config['var_transform']['eBeam']['min']

        ax = fig.add_subplot(1,1,1)
        ax.scatter(e_targets, e_preds, s=1)
        ax.set_xlabel('Target Energy')
        ax.set_ylabel('Predicted Energy')

        canvas.draw()
        w, h = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(h), int(w), 3)

        if self.comet_logger is not None:
            self.comet_logger.experiment.log_image(
                image_data=image,
                name='ED',
                overwrite=False, 
                image_format="png",
            )
        else:
            plt.savefig('ED.png')



