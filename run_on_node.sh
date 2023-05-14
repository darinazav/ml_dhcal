cd /srv01/agrp/darinaza/01_phd_workspace/17_dhcal_ml/ml_dhcal

source /usr/local/anaconda/3.8u/etc/profile.d/conda.sh
conda activate sup_res

if [ "$MODE" == "train" ]; then
    python train.py $CONFIG_PATH $EXP_KEY
elif [ "$MODE" == "pred" ]; then
    python eval.py $CONFIG_PATH $EXP_KEY
else
    echo "Invalid mode"
fi
