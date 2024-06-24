import os
import random
import pickle
from pathlib import Path

import xarray as xr
import torch
from tqdm import tqdm
from sklearn.linear_model import Ridge

from config import CACHE, DATA
from ..regression.regression_tools import regression_shared_unshared
from ..regression.torch_cv import TorchRidgeGCV


random.seed(0)

DATASET = 'majajhong'
SUBJECTS = ['Chabo','Tito']

MAJAJ_DATA = os.path.join(DATA,'majajhong')
TRAIN_IDS =  pickle.load(open(os.path.join(MAJAJ_DATA,'majaj_train_ids'), "rb"))
TEST_IDS =  pickle.load(open(os.path.join(MAJAJ_DATA,'majaj_test_ids'), "rb"))
PREDS_PATH = os.path.join(CACHE,'neural_preds')

ALPHA_RANGE = [10**i for i in range(10)]

def majajhong_scorer(activations_identifier: str, 
                       region: str,
                       device:str):
    
    pbar = tqdm(total = 2)
    
    for subject in tqdm(SUBJECTS):  
        file_path =  Path(PREDS_PATH) / f'{activations_identifier}_{region}_{subject}.pkl'
        
        if not os.path.exists(file_path):
            
            X_train = load_activations(activations_identifier, mode = 'train')
            X_test = load_activations(activations_identifier, mode = 'test')

            y_train = load_majaj_data(subject= subject, region= region, mode = 'train')
            y_test = load_majaj_data(subject= subject, region= region, mode = 'test')
    
            regression = TorchRidgeGCV(
                alphas=ALPHA_RANGE,
                fit_intercept=True,
                scale_X=False,
                scoring='pearsonr',
                store_cv_values=False,
                alpha_per_target=False,
                device = device)
            
            regression.fit(X_train, y_train)
            best_alpha = float(regression.alpha_)
    
            y_true, y_predicted = regression_shared_unshared(x_train=X_train,
                                                         x_test=X_test,
                                                         y_train=y_train,
                                                         y_test=y_test,
                                                         model= Ridge(alpha=best_alpha))
            pbar.update(1)
            with open(file_path,'wb') as f:
                pickle.dump(y_predicted, f,  protocol=4)
        else:
            pass
    return
        
    
def get_best_model_layer(activations_identifier, region, device):
        t = 0
        for iden in activations_identifier:
            activations_data = load_activations(activations_identifier = iden, mode = 'train') 
            scores = []
            alphas = [] 
            
            for subject in tqdm(range(len(SUBJECTS))):
                regression = fit_model_for_subject_roi(SUBJECTS[subject], region, activations_data, device)                
                scores.append(regression.score_.mean())
                alphas.append(float(regression.alpha_))
            
            mean_layer_score = sum(scores)/len(scores)
            if t == 0:
                best_score = mean_layer_score 
                t += 1
            if  mean_layer_score >= best_score:
                best_score = mean_layer_score
                best_layer = iden
                best_alphas = alphas
            print('best_layer:', best_layer)
        return best_layer, best_alphas
    
    
def majajhong_get_best_layer_scores(activations_identifier: list, region: str, device:str):

        best_layer, best_alphas = get_best_model_layer(activations_identifier, region, device)            

        for subject in tqdm(range(len(SUBJECTS))):
            file_path = Path(PREDS_PATH) / f'{best_layer}_{region}_{SUBJECTS[subject]}.pkl'
            
            if not os.path.exists(file_path):
                X_train = load_activations(best_layer, mode = 'train')
                X_test = load_activations(best_layer, mode = 'test')

                y_train = load_majaj_data(subject= SUBJECTS[subject], region= region, mode = 'train')
                y_test = load_majaj_data(subject= SUBJECTS[subject], region= region, mode = 'test')
                y_true, y_predicted = regression_shared_unshared(x_train=X_train,
                                                             x_test=X_test,
                                                             y_train=y_train,
                                                             y_test=y_test,
                                                             model= Ridge(alpha=best_alphas[subject]))
                with open(file_path,'wb') as f:
                    pickle.dump(y_predicted, f,  protocol=4)
            else:
                pass
        return
    
    
def fit_model_for_subject_roi(subject:int, region:str, activations_data:xr.DataArray(), device:str):
            
            X_train = activations_data
            y_train = load_majaj_data(subject, region, 'train')
    
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
            
            
def load_majaj_data(subject: str, region: str, mode: bool = None) -> torch.Tensor:
        
        """
        Loads the neural data from disk for a particular subject and region.

        Parameters
        ----------
            
        subject:
            The subject number 
        
        region:
            The region name
            
        mode:
            The type of neural data to load ('train' or 'test')

        Returns
        -------
        A Tensor of Neural data
        """
        
        file_name = f'SUBJECT_{subject}_REGION_{region}'
        file_path = os.path.join(MAJAJ_DATA,file_name)
        neural_data = xr.open_dataset(file_path, engine='netcdf4')
        
        if mode == 'train':
            neural_data = neural_data.where(neural_data.stimulus_id.isin(TRAIN_IDS),drop=True)
        elif mode == 'test':
            neural_data = neural_data.where(neural_data.stimulus_id.isin(TEST_IDS),drop=True)
        else:
            raise ValueError("mode should be one of 'train' or 'test'")
            
        neural_data = neural_data.sortby('stimulus_id', ascending=True)
        neural_data = torch.Tensor(neural_data['x'].values.squeeze())
        return neural_data

            
def load_activations(activations_identifier: str, mode: bool = None) -> torch.Tensor:
            
        """
        Loads model activations.

        Parameters
        ----------
            
        activations_identifier:
            model activations identifier
            
        mode:
            The type of neural data to load ('train' or 'test')
        
        Returns
        -------
        A Tensor of activation data
        """
        
        activations_data = xr.open_dataarray(os.path.join(CACHE,'activations',activations_identifier), engine='netcdf4')
        activations_data = activations_data.set_index({'presentation':'stimulus_id'})

        if mode == 'train':
            activations_data = activations_data.sel(presentation=TRAIN_IDS)
        elif mode == 'test':            
            activations_data = activations_data.sel(presentation=TEST_IDS)           
        else:
            raise ValueError("mode should be one of 'train' or 'test'")
        activations_data = activations_data.sortby('presentation', ascending=True)
        return torch.Tensor(activations_data.values)
    
    
    

    
    
    
    
