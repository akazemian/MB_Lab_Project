import gc
import logging

import numpy as np

from model_activations.models.utils import load_model, load_full_identifier
from model_activations.models.configs import analysis_cfg as cfg
from model_activations.activation_extractor import Activations
from encoding_score.regression.get_betas import NeuralRegression
from encoding_score.regression.scores_tools import get_bootstrap_rvalues
from config import setup_logging

setup_logging()
MODEL_NAME = 'expansion'
N_BOOTSTRAPS = 1000

def local_connectivity_(dataset, device):

    dataset += '_shuffled'
    N_ROWS = cfg[dataset]['test_data_size']
    ALL_SAMPLED_INDICES = np.random.choice(N_ROWS, (N_BOOTSTRAPS, N_ROWS), replace=True) # Sample indices for all 

    for features in cfg[dataset]['models'][MODEL_NAME]['features']:
        
        # get model identifier
        activations_identifier = load_full_identifier(model_name=MODEL_NAME, features=features, 
                                                      layers=cfg[dataset]['models'][MODEL_NAME]['layers'], 
                                                      dataset=dataset)
        logging.info(f"Model: {activations_identifier}, Region: {cfg[dataset]['regions']}")
        
        # load model
        model = load_model(model_name=MODEL_NAME, 
                           features=features, 
                           layers=cfg[dataset]['models'][MODEL_NAME]['layers'],
                           device=device)

        # extract activations 
        Activations(model=model, 
                    dataset=dataset, 
                    device=device).get_array(activations_identifier) 

        # predict neural data in a cross validated manner
        NeuralRegression(activations_identifier=activations_identifier,
                   dataset=dataset,
                   region=cfg[dataset]['regions'],
                   device= device).predict_data()

    # get a bootstrap distribution of r-values between predicted and actual neural responses
    get_bootstrap_rvalues(model_name = MODEL_NAME,
            features=cfg[dataset]['models'][MODEL_NAME]['features'],
            layers=cfg[dataset]['models'][MODEL_NAME]['layers'],
            dataset=dataset, 
            subjects = cfg[dataset]['subjects'],
            region=cfg[dataset]['regions'],
            all_sampled_indices=ALL_SAMPLED_INDICES,
            device=device,
            file_name = 'shuffled_data')
    gc.collect()
            
