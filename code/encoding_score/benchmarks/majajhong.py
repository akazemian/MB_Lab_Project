import os
import random
import pickle

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
PREDS_PATH = os.path.join(CACHE,'neural_preds')

TRAIN_IDS =  pickle.load(open(os.path.join(MAJAJ_DATA,'majaj_train_ids'), "rb"))
TEST_IDS =  pickle.load(open(os.path.join(MAJAJ_DATA,'majaj_test_ids'), "rb"))
TRAIN_IDS_DEMO =  pickle.load(open(os.path.join(MAJAJ_DATA,'majaj_train_ids_demo'), "rb"))
TEST_IDS_DEMO =  pickle.load(open(os.path.join(MAJAJ_DATA,'majaj_test_ids_demo'), "rb"))
ALPHA_RANGE = [10**i for i in range(10)]

def majajhong_scorer(activations_identifier: str, 
                       region: str,
                       device:str,
                       demo:bool=False):
    
    pbar = tqdm(total = 2)
    
    for subject in tqdm(SUBJECTS):  
        file_path = os.path.join(PREDS_PATH,f'{activations_identifier}_{region}_{subject}.pkl')
        
        if not os.path.exists(file_path):
            
            X_train, X_test, y_train, y_test = load_train_test(activations_identifier, subject, region, demo)
        
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
            del y_true, y_predicted
        else:
            pass
            
    return
        
    
def get_best_model_layer(activations_identifier, region, device, demo):
        t = 0
        for iden in activations_identifier:
            if demo:
                 activations_data = load_activations(activations_identifier = iden, mode = 'train_demo') 
            else:
                 activations_data = load_activations(activations_identifier = iden, mode = 'train') 
            scores = []
            alphas = [] 
            
            for subject in tqdm(range(len(SUBJECTS))):
                regression = fit_model_for_subject_roi(SUBJECTS[subject], region, activations_data, device, demo)                
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
    
    
def majajhong_get_best_layer_scores(activations_identifier: list, region: str, device:str, demo:bool=False):

        best_layer, best_alphas = get_best_model_layer(activations_identifier, region, device, demo)            

        for subject in tqdm(range(len(SUBJECTS))):
            file_path = os.path.join(PREDS_PATH,f'{best_layer}_{region}_{SUBJECTS[subject]}.pkl')
            
            if not os.path.exists(file_path):
                X_train, X_test, y_train, y_test = load_train_test(best_layer, SUBJECTS[subject], region, demo)
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
    
    
def fit_model_for_subject_roi(subject:int, region:str, activations_data:xr.DataArray(), device:str, demo:bool):
            
            X_train = activations_data
            if demo:
                 y_train = load_majaj_data(subject, region, 'train_demo')
            else:
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
            
            
def load_train_test(activations_identifier, subject, region, demo):

        if demo:
            X_train = load_activations(activations_identifier, mode = 'train_demo')
            X_test = load_activations(activations_identifier, mode = 'test_demo')
            y_train = load_majaj_data(subject= subject, region= region, mode = 'train_demo')
            y_test = load_majaj_data(subject= subject, region= region, mode = 'test_demo')
        else:
            X_train = load_activations(activations_identifier, mode = 'train')
            X_test = load_activations(activations_identifier, mode = 'test')
            y_train = load_majaj_data(subject= subject, region= region, mode = 'train')
            y_test = load_majaj_data(subject= subject, region= region, mode = 'test')             

        return X_train, X_test, y_train, y_test


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
        
        match mode:
            
            case 'train':
                neural_data = neural_data.where(neural_data.stimulus_id.isin(TRAIN_IDS),drop=True)
            case 'test':
                neural_data = neural_data.where(neural_data.stimulus_id.isin(TEST_IDS),drop=True)
            case 'train_demo':
                  neural_data = neural_data.where(neural_data.stimulus_id.isin(TRAIN_IDS_DEMO),drop=True)
            case 'test_demo':
                  neural_data = neural_data.where(neural_data.stimulus_id.isin(TEST_IDS_DEMO),drop=True)
            case _:
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
        print(CACHE, activations_identifier)
        activations_data = xr.open_dataarray(os.path.join(CACHE,'activations',activations_identifier), engine='netcdf4')
        activations_data = activations_data.set_index({'presentation':'stimulus_id'})

        match mode:
            case 'train':
                activations_data = activations_data.sel(presentation=TRAIN_IDS)
            case 'test':            
                activations_data = activations_data.sel(presentation=TEST_IDS)           
            case 'train_demo':
                activations_data = activations_data.sel(presentation=TRAIN_IDS_DEMO)
            case 'test_demo':            
                activations_data = activations_data.sel(presentation=TEST_IDS_DEMO)   
            case _:
                raise ValueError("mode should be one of 'train' or 'test'")
        activations_data = activations_data.sortby('presentation', ascending=True)
        return torch.Tensor(activations_data.values)
    
    
    

    
    
    
    
