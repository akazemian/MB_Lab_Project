import time
import os
import gc
import argparse
import logging
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
sys.path.insert(0, ROOT)

import numpy as np 
from dotenv import load_dotenv

from code.model_activations.models.utils import load_model, load_full_identifier
from code.model_activations.activation_extractor import Activations
from code.encoding_score.regression.get_betas import NeuralRegression
from code.encoding_score.regression.scores_tools import get_bootstrap_rvalues
from code.eigen_analysis.compute_pcs import compute_model_pcs
from model_configs import analysis_cfg as cfg
from config import setup_logging

setup_logging()
load_dotenv()

CACHE = os.getenv("CACHE")
MODEL_NAME = 'expansion'
DATASET = 'majajhong_demo'

def main(device, batch_size):
    '''
    The output encoding score for each model is saved as a pandas dataframe in the results folder
    '''   
    start_time = time.perf_counter()
    
    N_BOOTSTRAPS = 1000
    N_ROWS = cfg[DATASET]['test_data_size']
    ALL_SAMPLED_INDICES = np.random.choice(N_ROWS, (N_BOOTSTRAPS, N_ROWS), replace=True) 

    for features in cfg[DATASET]['analysis']['pca']['features']:

        TOTAL_COMPONENTS = 10 
        N_COMPONENTS = list(np.logspace(0, np.log10(TOTAL_COMPONENTS), num=int(np.log10(TOTAL_COMPONENTS)) + 1, base=10).astype(int))
        
        pca_identifier = load_full_identifier(model_name=MODEL_NAME, 
                                                    features=features, 
                                                    layers=cfg[DATASET]['analysis']['pca']['layers'], 
                                                    dataset=DATASET,
                                                    principal_components = TOTAL_COMPONENTS)
    
        # compute model PCs using the train set
        if not os.path.exists(os.path.join(CACHE,'pca',pca_identifier)):
            compute_model_pcs(model_name = MODEL_NAME, 
                              features = features, 
                              layers = cfg[DATASET]['analysis']['pca']['layers'], 
                              dataset = DATASET, 
                              components = TOTAL_COMPONENTS, 
                              device = device,
                              batch_size=batch_size)
            
        # project activations onto the computed PCs 
        for n_components in N_COMPONENTS:
            
            activations_identifier = load_full_identifier(model_name=MODEL_NAME, 
                                                    features=features, 
                                                    layers=cfg[DATASET]['analysis']['pca']['layers'], 
                                                    dataset=DATASET,
                                                    principal_components = n_components)            
            
            logging.info(f"Model: {activations_identifier}, Components = {n_components}, Region: {cfg[DATASET]['regions']}")
            #load model
            model = load_model(model_name=MODEL_NAME, 
                               features=features, 
                                   layers=cfg[DATASET]['analysis']['pca']['layers'],
                                   device=device)

            # compute activations and project onto PCs
            Activations(model=model, 
                        dataset=DATASET, 
                        pca_iden = pca_identifier,
                        n_components = n_components, 
                        batch_size = batch_size,
                        device= device).get_array(activations_identifier)  


            # predict neural data in a cross validated manner using model PCs
            NeuralRegression(activations_identifier=activations_identifier,
                             dataset=DATASET,
                             region=cfg[DATASET]['regions'],
                             device= device).predict_data()

            gc.collect()


    # get a bootstrap distribution of r-values between predicted and actual neural responses
    get_bootstrap_rvalues(model_name= MODEL_NAME,
                    features=cfg[DATASET]['analysis']['pca']['features'],
                    layers = cfg[DATASET]['analysis']['pca']['layers'],
                    principal_components=[1,10],
                    dataset=DATASET, 
                    subjects=cfg[DATASET]['subjects'],
                    region=cfg[DATASET]['regions'],
                    all_sampled_indices=ALL_SAMPLED_INDICES,
                    device=device,
                    file_name= 'pca')
    gc.collect()

    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.3f} seconds") 
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run scripts with dataset selection.")
    parser.add_argument('--device', default="cuda", help="Specify device name",
                        type=str, choices=['cpu', 'cuda'])
    parser.add_argument('--batchsize', default=50, help="Specify the batch size to use",
                        type=int)
    args = parser.parse_args()
    main(args.device, args.batchsize)

# 14-15 seconds