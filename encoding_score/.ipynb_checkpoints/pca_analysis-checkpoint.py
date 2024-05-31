import os
import numpy as np 
import gc
import argparse
import logging

from model_activations.models.utils import load_model, load_full_identifier
from model_activations.models.configs import analysis_cfg as cfg
from model_activations.activation_extractor import Activations
from encoding_score.regression.get_betas import NeuralRegression
from encoding_score.regression.scores_tools import get_bootstrap_rvalues
from config import CACHE, DATA, setup_logging
from eigen_analysis.compute_pcs import compute_model_pcs
setup_logging()

MODEL_NAME = 'expansion'

def main(dataset):
       
    N_BOOTSTRAPS = 1000
    N_ROWS = cfg[dataset]['test_data_size']
    ALL_SAMPLED_INDICES = np.random.choice(N_ROWS, (N_BOOTSTRAPS, N_ROWS), replace=True) 

    for features in cfg[dataset]['analysis']['pca']['features']:

        TOTAL_COMPONENTS = 100 if features == 3 else 1000 # model with 10^2 features has 100 components, the rest 1000
        N_COMPONENTS = list(np.logspace(0, np.log10(TOTAL_COMPONENTS), num=int(np.log10(TOTAL_COMPONENTS)) + 1, base=10).astype(int))
        
        pca_identifier = load_full_identifier(model_name=MODEL_NAME, 
                                                    features=features, 
                                                    layers=cfg[dataset]['analysis']['pca']['layers'], 
                                                    dataset=dataset,
                                                    principal_components = TOTAL_COMPONENTS)
    
        # compute model PCs using the train set
        if not os.path.exists(os.path.join(CACHE,'pca',pca_identifier)):
            compute_model_pcs(model_name = MODEL_NAME, features = features, 
                              layers = cfg[dataset]['analysis']['pca']['layers'], 
                              dataset = dataset, components = TOTAL_COMPONENTS, device = 'cuda')
            
        # project activations onto the computed PCs 
        for n_components in N_COMPONENTS:
            
            activations_identifier = load_full_identifier(model_name=MODEL_NAME, 
                                                    features=features, 
                                                    layers=cfg[dataset]['analysis']['pca']['layers'], 
                                                    dataset=dataset,
                                                    principal_components = n_components)            
            
            logging.info(f"Model: {activations_identifier}, Components = {n_components}, Region: {cfg[dataset]['regions']}")
            #load model
            model = load_model(model_name=MODEL_NAME, features=features, 
                                   layers=cfg[dataset]['analysis']['pca']['layers'])

            # compute activations and project onto PCs
            Activations(model=model, dataset=dataset, pca_iden = pca_identifier,
                        n_components = n_components, batch_size = 100,
                        device= 'cuda').get_array(activations_identifier)  


            # predict neural data in a cross validated manner using model PCs
            NeuralRegression(activations_identifier=activations_identifier,
                             dataset=dataset,
                             region=cfg[dataset]['regions'],
                             device= 'cuda').predict_data()

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
                    device='cpu',
                    file_name= 'pca')
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='majajhong', help='name of neural dataset')
    args = parser.parse_args()
    main(args.dataset)