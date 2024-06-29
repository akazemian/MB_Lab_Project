import gc
import logging

import numpy as np 

from code.model_activations.activation_extractor import Activations
from code.model_activations.models.utils import load_model, load_full_identifier
from .regression.get_betas import NeuralRegression
from .regression.scores_tools import get_bootstrap_rvalues
from config import setup_logging

setup_logging()
MODELS = ['expansion', 'fully_connected', 'vit']

def untrained_models_(dataset, cfg, batch_size, device):
       
    N_BOOTSTRAPS = 1000
    N_ROWS = cfg[dataset]['test_data_size']
    ALL_SAMPLED_INDICES = np.random.choice(N_ROWS, (N_BOOTSTRAPS, N_ROWS), replace=True) 

    for region in cfg[dataset]['regions']:
        for model_name in MODELS:
            for features in cfg[dataset]['models'][model_name]['features']:
                
                logging.info(f"Model: {model_name}, Features: {features}, Region: {cfg[dataset]['regions']}")
                # get model identifier
                activations_identifier = load_full_identifier(model_name=model_name, 
                                                              features=features, 
                                                              layers=cfg[dataset]['models'][model_name]['layers'], 
                                                              dataset=dataset
                                                              )
                    
                model = load_model(model_name=model_name, 
                                   features=features, 
                                   layers=cfg[dataset]['models'][model_name]['layers'],
                                   device=device)
    
                # extract activations 
                _ = Activations(model=model, 
                            dataset=dataset, 
                            device= device,
                            batch_size=batch_size).get_array(activations_identifier)
                del _ 
    
    
                logging.info(f"Predicting neural data from model activations")
                # predict neural data in a cross validated manner
                NeuralRegression(activations_identifier=activations_identifier, 
                                 dataset=dataset,
                                 region=region, 
                                 device= device).predict_data()
                

            logging.info(f"Getting a bootstrap distribution of scores")
            # get a bootstrap distribution of r-values between predicted and actual neural responses
            get_bootstrap_rvalues(model_name = model_name,
                    features=cfg[dataset]['models'][model_name]['features'],
                    layers=cfg[dataset]['models'][model_name]['layers'],
                    dataset=dataset, 
                    subjects = cfg[dataset]['subjects'],
                    region=region,
                    all_sampled_indices=ALL_SAMPLED_INDICES,
                    device=device)
            gc.collect()
    return
                
