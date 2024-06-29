import gc
import logging

import numpy as np 

from code.model_activations.models.utils import load_full_identifier
from code.model_activations.models.expansion import Expansion5L
from code.model_activations.activation_extractor import Activations
from code.encoding_score.regression.get_betas import NeuralRegression
from code.encoding_score.regression.scores_tools import get_bootstrap_rvalues
from config import setup_logging

setup_logging()
MODEL_NAME = 'expansion'
ANALYSIS = 'non_linearities'
N_BOOTSTRAPS = 1000

def non_linearity_(dataset, cfg, batch_size, device):
    N_ROWS = cfg[dataset]['test_data_size']
    ALL_SAMPLED_INDICES = np.random.choice(N_ROWS, (N_BOOTSTRAPS, N_ROWS), replace=True) # Sample indices for all 

    for non_linearity in cfg[dataset]['analysis'][ANALYSIS]['variations']:
    
        for features in cfg[dataset]['analysis'][ANALYSIS]['features']:
                    
            logging.info(f"Non linearity type: {non_linearity}")

            # get model identifier
            activations_identifier = load_full_identifier(model_name=MODEL_NAME, features=features, 
                                                          layers=cfg[dataset]['analysis'][ANALYSIS]['layers'], 
                                                          dataset=dataset,
                                                          non_linearity = non_linearity)
                
            model = Expansion5L(filters_5=features, 
                                non_linearity=non_linearity,
                                device=device).build()
    
    
            # extract activations 
            data = Activations(model=model, 
                               dataset=dataset, 
                               device=device,
                               batch_size=batch_size).get_array(activations_identifier) 
            del data
    
            logging.info(f"Predicting neural data from model activations")
            # predict neural data in a cross validated manner
            NeuralRegression(activations_identifier=activations_identifier,
                       dataset=dataset,
                       region=cfg[dataset]['regions'],
                       device= device).predict_data()

            gc.collect()
            
    logging.info(f"Getting a bootstrap distribution of scores")
    # get a bootstrap distribution of r-values between predicted and actual neural responses
    get_bootstrap_rvalues(model_name = MODEL_NAME,
            features=cfg[dataset]['analysis'][ANALYSIS]['features'],
            layers=cfg[dataset]['analysis'][ANALYSIS]['layers'],
            dataset=dataset, 
            subjects = cfg[dataset]['subjects'],
            region=cfg[dataset]['regions'],
            all_sampled_indices=ALL_SAMPLED_INDICES,
            device=device,
            non_linearity=cfg[dataset]['analysis'][ANALYSIS]['variations'],
            file_name = 'non_linearity')
    gc.collect()
            

