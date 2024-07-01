import os
import random
import pickle
import warnings
import gc

import xarray as xr
import numpy as np
import torch
from tqdm import tqdm
from sklearn.linear_model import Ridge
from dotenv import load_dotenv

from ..regression.regression_tools import regression_shared_unshared
from ..regression.torch_cv import TorchRidgeGCV
from code.encoding_score.benchmarks.benchmarks_configs import PREDS_PATH, ALPHA_RANGE, NSD_NEURAL_DATA

load_dotenv()
warnings.filterwarnings('ignore')
random.seed(0)

CACHE = os.getenv("CACHE")

def normalize(X, X_min=None, X_max=None, use_min_max=False):
    if use_min_max:
        X_normalized = (X - X_min) / (X_max - X_min)
        return X_normalized
    else:
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        X_normalized = (X - X_min) / (X_max - X_min)
        return X_normalized, X_min, X_max


def nsd_scorer(activations_identifier: str, 
               region: str,
              device: str):
        
        for subject in tqdm(range(8)):

            file_path = os.path.join(PREDS_PATH,f'{activations_identifier}_{region}_{subject}.pkl')
            if not os.path.exists(file_path):
            
                activations_data = xr.open_dataarray(os.path.join(CACHE,'activations',activations_identifier), engine='netcdf4')  
                ids_train, neural_data_train = load_nsd_data(mode ='unshared',
                                                    subject = subject,
                                                    region = region)
                X_train = filter_activations(data = activations_data, ids = ids_train)  
                X_train, X_min, X_max = normalize(X_train) # normalize
                X_train = np.nan_to_num(X_train) # omit nans
                y_train = neural_data_train['beta'].values
            
                regression = TorchRidgeGCV(
                    alphas=ALPHA_RANGE,
                    fit_intercept=True,
                    scale_X=False,
                    scoring='pearsonr',
                    store_cv_values=False,
                    alpha_per_target=False,
                    device=device)
                
                regression.fit(X_train, y_train)
                best_alpha = float(regression.alpha_)
            
            
                ids_test, neural_data_test = load_nsd_data(mode ='shared',
                                                        subject = subject,
                                                        region = region)           
                X_test = filter_activations(data = activations_data, ids = ids_test)                
                X_test = normalize(X_test, X_min, X_max, use_min_max=True) # normalize
                X_test = np.nan_to_num(X_test) # omit nans
                y_test = neural_data_test['beta'].values                   
                
                y_true, y_predicted = regression_shared_unshared(x_train=X_train,
                                                                x_test=X_test,
                                                                y_train=y_train,
                                                                y_test=y_test,
                                                                model= Ridge(alpha=best_alpha))
                with open(file_path,'wb') as f:
                    pickle.dump(y_predicted, f,  protocol=4)
                del y_true, y_predicted
            else:
                pass
        return         
            

def get_best_model_layer(activations_identifier, region, device):
    
        best_score = 0
        
        for iden in activations_identifier:
            
            print('getting scores for:',iden)
            activations_data = xr.open_dataarray(os.path.join(CACHE,'activations',iden), engine='netcdf4')  
        
            scores = []
            alphas = [] 
            
            for subject in tqdm(range(8)):

                regression = fit_model_for_subject_roi(subject, region, activations_data, device)                
                scores.append(regression.score_.mean())
                alphas.append(float(regression.alpha_))
            
            mean_layer_score = sum(scores)/len(scores)
            
            if  mean_layer_score > best_score:
                best_score = mean_layer_score
                best_layer = iden
                best_alphas = alphas
            print('best_layer:', best_layer, 'best_alphas:', best_alphas)
            
            gc.collect()            
        
        return best_layer, best_alphas
    
            

def nsd_get_best_layer_scores(activations_identifier: list, region: str, device:str):

        best_layer, best_alphas = get_best_model_layer(activations_identifier, region, device)            
        
        for subject in tqdm(range(8)):

            file_path = os.path.join(PREDS_PATH,f'{best_layer}_{region}_{subject}.pkl')
            if not os.path.exists(file_path):
                activations_data = xr.open_dataarray(os.path.join(CACHE,'activations',best_layer), engine='netcdf4')  
                ids_train, neural_data_train = load_nsd_data(mode ='unshared',
                                                    subject = subject,
                                                    region = region)
                X_train = filter_activations(data = activations_data, ids = ids_train)       
                y_train = neural_data_train['beta'].values

                ids_test, neural_data_test = load_nsd_data(mode ='shared',
                                                        subject = subject,
                                                        region = region)           

                X_test = filter_activations(data = activations_data, ids = ids_test)               
                y_test = neural_data_test['beta'].values                    
                
                _, y_predicted = regression_shared_unshared(x_train=X_train,
                                                                x_test=X_test,
                                                                y_train=y_train,
                                                                y_test=y_test,
                                                                model= Ridge(alpha=best_alphas[subject]))
                with open(file_path,'wb') as f:
                    pickle.dump(y_predicted, f,  protocol=4)
                del _, y_predicted
            else:
                pass
        return
        
    
    
def fit_model_for_subject_roi(subject:int, region:str, activations_data:xr.DataArray(), device:str):
            
            ids_train, neural_data_train = load_nsd_data(mode ='unshared',
                                                         subject = subject,
                                                         region = region)

            X_train = filter_activations(data = activations_data, ids = ids_train)       
            y_train = neural_data_train['beta'].values

            regression = TorchRidgeGCV(
                alphas=ALPHA_RANGE,
                fit_intercept=True,
                scale_X=False,
                scoring='pearsonr',
                store_cv_values=False,
                alpha_per_target=False,
                device=device)
            
            regression.fit(X_train, y_train)
            return regression    
    
            
            
def load_nsd_data(mode: str, subject: int, region: str, return_data=True) -> torch.Tensor:
        
        SHARED_IDS = pickle.load(open(os.path.join(NSD_NEURAL_DATA, 'nsd_ids_shared'), 'rb'))
        SHARED_IDS = [image_id.strip('.png') for image_id in SHARED_IDS]
        # IDS_TRUNCATED = pickle.load(open(os.path.join(NSD_NEURAL_DATA, 'ids_truncated'), 'rb'))
        # SHARED_IDS = list(set(SHARED_IDS) & set(IDS_TRUNCATED))

        ds = xr.open_dataset(os.path.join(NSD_NEURAL_DATA,region,'preprocessed',f'subject={subject}.nc'),engine='netcdf4')
        # mask = ds.presentation.stimulus.isin(IDS_TRUNCATED)
        # ds = ds.sel(presentation=ds['presentation'][mask])
        
        if mode == 'unshared':
                mask = ~ds.presentation.stimulus.isin(SHARED_IDS)
                ds = ds.sel(presentation=ds['presentation'][mask])

        elif mode == 'shared':
                mask = ds.presentation.stimulus.isin(SHARED_IDS)
                ds = ds.sel(presentation=ds['presentation'][mask])
            
        ids = list(ds.presentation.stimulus.values.astype(str))
            
        if return_data:
            return ids, ds
        else:
            return ids
        
            
def filter_activations(data: xr.DataArray, ids: list) -> torch.Tensor:
        
        data = data.where(data['stimulus_id'].isin(ids),drop=True)
        data = data.sortby('presentation', ascending=True)

        return data.values




