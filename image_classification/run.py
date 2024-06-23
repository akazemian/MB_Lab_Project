import os
import logging
import pickle
import xarray as xr 

from config import CACHE, setup_logging
from tools.loading import load_places_cat_labels
from model_activations.models.utils import load_model, load_full_identifier
from model_activations.models.configs import analysis_cfg as cfg
from model_activations.activation_extractor import Activations
from image_classification.config_ import MODEL_NAMES, PCA_DATASET, DATASET, NUM_COMPONENTS
from image_classification.classification_tools import get_Xy, cv_performance

from eigen_analysis.compute_pcs import compute_model_pcs
setup_logging()

if not os.path.exists(CACHE):
    os.makedirs(CACHE)
    
if not os.path.exists(os.path.join(CACHE,'classification')):
    os.mkdir(os.path.join(CACHE,'classification'))    

for model_name in MODEL_NAMES:
    
    pca_identifier = load_full_identifier(model_name=model_name, 
                                                    features=cfg[PCA_DATASET]['models'][model_name]['features'], 
                                                    layers=cfg[PCA_DATASET]['models'][model_name]['layers'], 
                                                    dataset=PCA_DATASET,
                                                    principal_components = NUM_COMPONENTS)
    # compute model PCs using the train set
    if not os.path.exists(os.path.join(CACHE,'pca',pca_identifier)):
        compute_model_pcs(model_name = model_name, features=cfg[PCA_DATASET]['models'][model_name]['features'],  
                          layers=cfg[PCA_DATASET]['models'][model_name]['layers'], 
                          dataset = PCA_DATASET, components = NUM_COMPONENTS, device = 'cuda')

    activations_identifier = load_full_identifier(model_name=model_name, 
                                            features=cfg[DATASET]['models'][model_name]['features'], 
                                            layers=cfg[DATASET]['models'][model_name]['layers'], 
                                            dataset=DATASET,
                                            principal_components = NUM_COMPONENTS)
    
    if not os.path.exists(os.path.join(CACHE,'classification',activations_identifier)):
        logging.info(f"Model: {activations_identifier}")
        
        # load model
        model = load_model(model_name=model_name, features=cfg[DATASET]['models'][model_name]['features'], 
                               layers=cfg[DATASET]['models'][model_name]['layers'])
        
        # compute activations and project onto PCs
        Activations(model=model, dataset=DATASET, pca_iden = pca_identifier,
                    n_components = NUM_COMPONENTS, batch_size = 50,
                    device= 'cuda').get_array(activations_identifier)  
        
        data = xr.open_dataset(os.path.join(CACHE,'activations',activations_identifier), engine='h5netcdf').set_xindex('stimulus_id')
        cat_labels = load_places_cat_labels()
        X, y = get_Xy(data)
        score = cv_performance(X, y, cat_labels)
        print(activations_identifier, ':', score)
                
        with open(os.path.join(CACHE,'classification',activations_identifier),'wb') as f:
            pickle.dump(score,f)