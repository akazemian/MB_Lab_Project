import os
import gc
import argparse
import logging

import numpy as np 

from model_configs import analysis_cfg as cfg
from model_activations.models.utils import load_model, load_full_identifier
from model_activations.activation_extractor import Activations
from encoding_score.regression.get_betas import NeuralRegression
from encoding_score.regression.scores_tools import get_bootstrap_rvalues
from eigen_analysis.compute_pcs import compute_model_pcs
from config import CACHE, setup_logging

setup_logging()
MODEL_NAME = 'expansion'

def main(dataset, device, batch_size):
       
    N_BOOTSTRAPS = 1000
    N_ROWS = cfg[dataset]['test_data_size']
    ALL_SAMPLED_INDICES = np.random.choice(N_ROWS, (N_BOOTSTRAPS, N_ROWS), replace=True) 

    for features in cfg[dataset]['analysis']['pca']['features']:

        logging.info(f"Model: {MODEL_NAME}, Features: {features}, Region: {cfg[dataset]['regions']}")
        
        TOTAL_COMPONENTS = 100 if features == 3 else 1000 # model with 10^2 features has 100 components, the rest 1000
        N_COMPONENTS = list(np.logspace(0, np.log10(TOTAL_COMPONENTS), num=int(np.log10(TOTAL_COMPONENTS)) + 1, base=10).astype(int))
        
        pca_identifier = load_full_identifier(model_name=MODEL_NAME, 
                                                    features=features, 
                                                    layers=cfg[dataset]['analysis']['pca']['layers'], 
                                                    dataset=dataset,
                                                    principal_components = TOTAL_COMPONENTS)
    
        # compute model PCs using the train set
        if not os.path.exists(os.path.join(CACHE,'pca',pca_identifier)):
            logging.info(f"Computing PCs for model")
            
            compute_model_pcs(model_name = MODEL_NAME, 
                              features = features, 
                              layers = cfg[dataset]['analysis']['pca']['layers'], 
                              dataset = dataset, 
                              components = TOTAL_COMPONENTS, 
                              device = device,
                              batch_size=batch_size)
            
        # project activations onto the computed PCs 
        for n_components in N_COMPONENTS:
            
            activations_identifier = load_full_identifier(model_name=MODEL_NAME, 
                                                    features=features, 
                                                    layers=cfg[dataset]['analysis']['pca']['layers'], 
                                                    dataset=dataset,
                                                    principal_components = n_components)            
            
            
            logging.info(f"Extracting activations and projecting onto the first {n_components} PCs ")
            
            #load model
            model = load_model(model_name=MODEL_NAME, 
                               features=features, 
                                   layers=cfg[dataset]['analysis']['pca']['layers'],
                                   device=device)

            # compute activations and project onto PCs
            Activations(model=model, 
                        dataset=dataset, 
                        pca_iden = pca_identifier,
                        n_components = n_components, 
                        batch_size = 100,
                        device= device).get_array(activations_identifier)  


            logging.info(f"Predicting neural data using the first {n_components} PCs ")
            # predict neural data in a cross validated manner using model PCs
            NeuralRegression(activations_identifier=activations_identifier,
                             dataset=dataset,
                             region=cfg[dataset]['regions'],
                             device= device).predict_data()

            gc.collect()


    # get a bootstrap distribution of r-values between predicted and actual neural responses
    get_bootstrap_rvalues(model_name= MODEL_NAME,
                    features=cfg[dataset]['analysis']['pca']['features'],
                    layers = cfg[dataset]['analysis']['pca']['layers'],
                    principal_components=[1,10,100,1000],
                    dataset=dataset, 
                    subjects=cfg[dataset]['subjects'],
                    region=cfg[dataset]['regions'],
                    all_sampled_indices=ALL_SAMPLED_INDICES,
                    device=device,
                    file_name= 'pca')
    gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run scripts with dataset selection.")
    parser.add_argument('--dataset', required=True, help="Specify the dataset name",
                        type=str, choices=['naturalscenes', 'majajhong'])
    parser.add_argument('--device', default="cuda", help="Specify device name",
                        type=str, choices=['cpu', 'cuda'])
    parser.add_argument('--batchsize', default=50, help="Specify the batch size to use",
                        type=int)
    args = parser.parse_args()
    main(args.dataset, args.device, args.batchsize)
