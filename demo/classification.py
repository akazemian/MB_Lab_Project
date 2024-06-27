import os
import logging
import pickle
import argparse
import time

import xarray as xr

from config import CACHE, RESULTS_PATH, setup_logging
from model_configs import analysis_cfg as cfg
from code.model_activations.models.utils import load_model, load_full_identifier
from code.model_activations.activation_extractor import Activations
from code.image_classification.classification_tools import get_Xy, cv_performance
from code.eigen_analysis.compute_pcs import compute_model_pcs

setup_logging()

if not os.path.exists(CACHE):
    os.makedirs(CACHE)
    
if not os.path.exists(RESULTS_PATH):
    os.mkdir(RESULTS_PATH)      

MODEL_NAMES = ['expansion','alexnet']
PCA_DATASET = 'places_train_demo'
DATASET = 'places_val_demo'
NUM_COMPONENTS = 100

def main(batch_size:int, device:str):    
    '''
    The output accuracy score for each model is saved as a pkl file in the results folder
    '''
    start_time = time.perf_counter()
    for model_name in MODEL_NAMES:
        # get model activation iden
        activations_identifier = load_full_identifier(model_name=model_name, 
                                                features=cfg[DATASET]['models'][model_name]['features'], 
                                                layers=cfg[DATASET]['models'][model_name]['layers'], 
                                                dataset=DATASET,
                                                principal_components = NUM_COMPONENTS)
        
        if not os.path.exists(os.path.join(RESULTS_PATH,f'classification_{model_name}')):
            logging.info(f"Getting classification results for: {activations_identifier}")
            
            
            pca_identifier = load_full_identifier(model_name=model_name, 
                                                        features=cfg[PCA_DATASET]['models'][model_name]['features'], 
                                                        layers=cfg[PCA_DATASET]['models'][model_name]['layers'], 
                                                        dataset=PCA_DATASET,
                                                        principal_components = NUM_COMPONENTS)
            
            logging.info(f"Computing model principal components using a subset of the Places train set")
            if not os.path.exists(os.path.join(CACHE,'pca',pca_identifier)):
                compute_model_pcs(model_name = model_name, 
                                features=cfg[PCA_DATASET]['models'][model_name]['features'],  
                                layers=cfg[PCA_DATASET]['models'][model_name]['layers'], 
                                dataset = PCA_DATASET, 
                                components = NUM_COMPONENTS, 
                                device = device,
                                batch_size=batch_size)
                
            # load model
            model = load_model(model_name=model_name, 
                            features=cfg[DATASET]['models'][model_name]['features'], 
                            layers=cfg[DATASET]['models'][model_name]['layers'],
                            device=device)
            
            logging.info(f"Extracting activations from the Places val set and projecting them onto the learned PCs")
            # compute activations and project onto PCs
            Activations(model=model, 
                        dataset=DATASET, 
                        pca_iden = pca_identifier,
                        n_components = NUM_COMPONENTS, 
                        batch_size = batch_size,
                        device= device).get_array(activations_identifier)  
            
            logging.info(f"Extracting activations from the Places val set and projecting them onto the learned PCs")
            X, y = get_Xy(data=xr.open_dataset(os.path.join(CACHE,'activations',activations_identifier), 
                                engine='netcdf4').set_xindex('stimulus_id'))
            score = cv_performance(X, y, class_balance=False)

            with open(os.path.join(RESULTS_PATH,f'classification_{model_name}'),'wb') as f:
                pickle.dump(score,f)
        else:
            print(f'results for model: {activations_identifier} are already saved in cache')

    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.3f} seconds") 
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', 
                        default = 64, 
                        help="Specify the batch size to use",
                        type=int)
    parser.add_argument('--device', 
                        required=False, 
                        help="Specify device name",
                        type=str, 
                        default='cuda',
                        choices=['cpu', 'cuda'])
    args = parser.parse_args()
    main(args.batchsize, args.device)



# 2.4 minutes