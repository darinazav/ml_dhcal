import os
import sys
import json
import shutil
from pathlib import Path
import random, string

ncpus = '3'
ngpus = '1'
mem   = '10gb'
walltime = '72:00:00'


# copy the config file to the output directory
config_path = sys.argv[1]
with open(config_path, 'r') as fp:
    config = json.load(fp)

mode = sys.argv[2]

if mode=='train':

    exp_key  = f'v{config["version"]}xxx'
    exp_key += ''.join(random.choices(string.ascii_lowercase + string.digits, k=32-len(exp_key)))

    dst = f'{config["base_root_dir"]}/{config["name"]}/{config["comet_settings"]["project_name"]}/{exp_key}'
    Path(dst).mkdir(parents=True, exist_ok=True)

    new_config_path = os.path.join(dst, 'config.json')
    shutil.copyfile(config_path, new_config_path)


    command  = f'qsub -o {dst}/output.log'
    command += f' -e {dst}/error.log'
    command += f' -q gpu -N sup_train -l walltime={walltime},mem={mem},ncpus={ncpus},ngpus={ngpus}'
    command += f' -v MODE="{mode}",CONFIG_PATH="{new_config_path}",EXP_KEY="{exp_key}"'
    command += f' /srv01/agrp/darinaza/01_phd_workspace/17_dhcal_ml/ml_dhcal/run_on_node.sh'

elif mode=='pred':

    if "/storage" not in config_path:
        raise ValueError("Please submit the config file from /storage")

    exp_key = sys.argv[3]
    dst = f'{config["eval_root_dir"]}/{config["name"]}/v{config["version"]}'
    Path(dst).mkdir(parents=True, exist_ok=True)

    command  = f'qsub -o {dst}/output.log'
    command += f' -e {dst}/error.log'
    command += f' -q gpu -N sup_pred -l walltime={walltime},mem={mem},ncpus={ncpus},ngpus={ngpus}'
    command += f' -v MODE="{mode}",CONFIG_PATH="{config_path}",EXP_KEY="{exp_key}"'
    command += f' /srv01/agrp/darinaza/01_phd_workspace/17_dhcal_ml/ml_dhcal/run_on_node.sh'


print(command)
os.system(command)

