import comet_ml
import glob

import sys
paths = sys.path
for p in paths:
     if '.local' in p:
             paths.remove(p)

import json

from lightning import PflowLightning
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


if __name__ == "__main__":
    config_path = sys.argv[1]

    if len(sys.argv) == 3:
        debug_mode = sys.argv[2]
    else:
        debug_mode = '0'

    with open(config_path, 'r') as fp:
         config = json.load(fp)

    net = PflowLightning(config)


    # for saving checkpoints for best 3 models (according to val loss) and last epoch
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        every_n_train_steps=0,
        every_n_epochs=1,
        train_time_interval=None,
        save_top_k=3,
        save_last= True,
        filename='{epoch}-{val_loss:.4f}')


    if debug_mode == '1':
        trainer = Trainer(
            max_epochs=config['num_epochs'],
            accelerator='gpu',
            devices = 1,
            default_root_dir=config["base_root_dir"],
            callbacks=[checkpoint_callback],
            resume_from_checkpoint = config['resume_from_checkpoint'],
        )

    else:
        exp_key = sys.argv[2]
        comet_logger = CometLogger(
            api_key=config["comet_settings"]["api_key"],
            project_name=config["comet_settings"]["project_name"], # super_res
            workspace=config["comet_settings"]["workspace"], # nilotpal_09
            experiment_name=config['name']+'_v'+config["version"], # scd_single_e_v_test
            experiment_key=exp_key
        )

        net.set_comet_logger(comet_logger)
        comet_logger.experiment.log_asset(config_path,file_name='config')

        all_files = glob.glob('./*.py')+glob.glob('models/*.py')+glob.glob('utility/*.py')
        for fpath in all_files:
            comet_logger.experiment.log_asset(fpath)

        trainer = Trainer(
            max_epochs = config['num_epochs'],
            accelerator = 'gpu',
            devices = 1,
            default_root_dir=f'{config["base_root_dir"]}/{config["name"]}', # experimet/scd_single_e
            logger = comet_logger,
            resume_from_checkpoint = config['resume_from_checkpoint'],
            callbacks=[checkpoint_callback],
            log_every_n_steps=1
        )
    
    trainer.fit(net, ckpt_path=config['resume_from_checkpoint'])

    # val_out = trainer.validate(model=net, ckpt_path='best')

    # print(val_out)
    # print(trainer.current_epoch)
