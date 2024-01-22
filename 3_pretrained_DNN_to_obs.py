import numpy as np
import sys
import pandas as pd
import joblib
import multiprocessing as mp
import psutil
import datetime
import warnings
from Models import data_extraction,rsquared
warnings.filterwarnings('ignore')


def pretrained_ANN_to_obs(RTM,tr,tr_obs,refl_obs,type):
    model_path = f'/scratch/fji7/transfer_learning_paper/2_saved_ML_model/{RTM}_{tr}_pre-trained.pkl'
    model = joblib.load(model_path)
    x = refl_obs.values
    y = tr_obs[tr].values
    pred = model.predict(x)
    print(f'{RTM} pretrained ANN~{tr} R2=',round(rsquared(y, pred),4))
    tr_obs['predicted'] = pred
    tr_obs.to_csv(f'/scratch/fji7/transfer_learning_paper/3_results/{tr}/2_pretrained {RTM} ANN to obs_{tr}_{type}.csv',index = False)
    return tr_obs

"""
1. Original pretrained ANN model apply to observations (overall)
"""

RTMs = ['Leaf-SIP','PROSPECT']
tr_name = ["Chla+b", 'Ccar', 'EWT','LMA']

start_t = datetime.datetime.now()
print('start:', start_t)

for RTM in RTMs:
    for tr in tr_name:
        if tr == 'EWT':
            data_types = ['sites','PFT']
        else:
            data_types = ['sites','PFT','temporal']
        for type in data_types:
            refl = pd.read_csv(f'/scratch/fji7/transfer_learning_paper/0_datasets/{tr}_reflectance_{type}.csv')
            traits = pd.read_csv(f'/scratch/fji7/transfer_learning_paper/0_datasets/{tr}_traits_{type}.csv', dtype = {'Sample date': object})
            tr_obs, refl_obs = data_extraction(traits, refl, tr)
            print(tr, type, len(tr_obs))
            pretrained_ANN_to_obs(RTM,tr,tr_obs,refl_obs,type)

end_t = datetime.datetime.now()
elapsed_sec = (end_t - start_t).total_seconds()
print('end:', end_t)
print('total:',elapsed_sec/60, 'min')