import sys
import xarray as xr
import numpy as np
import torch
import os
import random 
from tqdm import tqdm 
import pickle 
import warnings
warnings.filterwarnings('ignore')
import random    
random.seed(0)
import scipy.stats as st
import gc
from scipy.sparse import csr_matrix
from pathlib import Path
from sklearn.linear_model import Ridge

from config import CACHE, DATA    
from ..regression.regression_tools import regression_shared_unshared
from ..regression.torch_cv import TorchRidgeGCV

NSD_NEURAL_DATA = os.path.join(DATA,'naturalscenes')
SHARED_IDS = pickle.load(open(os.path.join(NSD_NEURAL_DATA, 'nsd_ids_shared'), 'rb'))
SHARED_IDS = [image_id.strip('.png') for image_id in SHARED_IDS]

ALPHA_RANGE = [10**i for i in range(10)]
      
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
    

        activations_data = xr.open_dataarray(os.path.join(CACHE,'activations',activations_identifier), engine='h5netcdf')  

        for subject in tqdm(range(8)):

            ids_train, neural_data_train = load_nsd_data(mode ='unshared',
                                                    subject = subject,
                                                    region = region)
        
            X_train = filter_activations(data = activations_data, ids = ids_train)  
            
            X_train, X_min, X_max = normalize(X_train) # normalize
            X_train = np.nan_to_num(X_train) # omit nans
            
            y_train = neural_data_train['beta'].values
            print('neural data shape:',y_train.shape)
            
            
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
            print('best alpha:',best_alpha)
            
            
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
        return y_predicted           
            




def get_best_model_layer(activations_identifier, region, device):
    
        best_score = 0
        
        for iden in activations_identifier:
            
            print('getting scores for:',iden)
            activations_data = xr.open_dataarray(os.path.join(CACHE,'activations',iden), engine='h5netcdf')  
        
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
        activations_data = xr.open_dataarray(os.path.join(CACHE,'activations',best_layer), engine='h5netcdf')  
        
        for subject in tqdm(range(8)):

            #file_path = Path(PREDS_PATH) / f'alexnet_gpool=False_dataset=naturalscenes_{region}_{subject}.pkl'
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
            
            y_true, y_predicted = regression_shared_unshared(x_train=X_train,
                                                             x_test=X_test,
                                                             y_train=y_train,
                                                             y_test=y_test,
                                                             model= Ridge(alpha=best_alphas[subject]))
        return y_predicted      
        
    
    
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
        
        """
        
        Loads the neural data from disk for a particular subject and region.

        Parameters
        ----------
        mode:
            The type of neural data to load ('shared' or 'unshared')
            
        subject:
            The subject number 
        
        region:
            The region name
            
        return_ids: 
            Whether the image ids are returned 
        

        Returns
        -------
        A Tensor of Neural data, or Tensor of Neural data and stimulus ids
        
        """
        ds = xr.open_dataset(os.path.join(NSD_NEURAL_DATA,region,'preprocessed',f'subject={subject}.nc'),engine='h5netcdf')
        
        if mode == 'unshared':
                mask = ~ds.presentation.stimulus.isin(SHARED_IDS)
                ds = ds.sel(presentation=ds['presentation'][mask])

        elif mode == 'shared':
                mask = ds.presentation.stimulus.isin(SHARED_IDS)
                ds = ds.sel(presentation=ds['presentation'][mask])
            
        ids = list(ds.presentation.stimulus.values)
            
        if return_data:
            return ids, ds
        else:
            return ids
        
        
            
def filter_activations(data: xr.DataArray, ids: list) -> torch.Tensor:
            
        """
    
        Filters model activations using image ids.


        Parameters
        ----------
        data:
            Model activation data
            
        ids:
            image ids
        

        Returns
        -------
        A Tensor of model activations filtered by image ids
        
        """
        
       
        #data = data.set_index({'presentation':'stimulus'})
        #formatted_ids = [f"image{num:05d}" for num in ids]
        #print('neural ids:',ids[:10])
        #print('activation ids:',data.stimulus_id.values[:10])
        data = data.where(data['stimulus_id'].isin(ids),drop=True)

        #activations = data.sel(presentation=ids)
        data = data.sortby('presentation', ascending=True)

        return data.values
            
        
        
        
def set_new_coord(ds):
    
    new_coord_data = np.core.defchararray.add(np.core.defchararray.add(ds.x.values.astype(str), ds.y.values.astype(str)), ds.z.values.astype(str))
    new_coord = xr.DataArray(new_coord_data, dims=['neuroid'])
    ds = ds.assign_coords(xyz=new_coord)
    
    return ds



def filter_roi(subject,roi):

    ds_source = xr.open_dataset(f'/data/rgautha1/cache/bonner-caching/neural-dimensionality/data/dataset=allen2021.natural_scenes/betas/resolution=1pt8mm/preprocessing=fithrf/z_score=True/roi={roi}/subject={subject}.nc',engine='h5netcdf')

    ds_target = xr.open_dataset(os.path.join(NSD_NEURAL_DATA,f'roi=general/preprocessed/z_score=session.average_across_reps=True/subject={subject}.nc'), engine='h5netcdf')

    ds_source = set_new_coord(ds_source)
    ds_target = set_new_coord(ds_target)

    source_ids = ds_source['xyz'].values
    mask = ds_target['xyz'].isin(source_ids)
    ds_target = ds_target.sel(neuroid=ds_target['neuroid'][mask])
    
    return ds_target


