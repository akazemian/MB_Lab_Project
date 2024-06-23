import xarray as xr
import os
import pickle

from eigen_analysis.utils import _PCA
from model_activations.models.utils import load_model, load_full_identifier
from model_activations.activation_extractor import Activations
from config import CACHE, DATA, setup_logging
setup_logging()

def compute_model_pcs(model_name:str, features:int, layers:int, 
                      dataset:str, components:int, device:str):
    
    activations_identifier = load_full_identifier(model_name=model_name, features=features, layers=layers, dataset=dataset)
    model = load_model(model_name=model_name, 
                       features=features, 
                       layers=layers,
                       device=device)
    
    Activations(model=model, dataset=dataset, device= device).get_array(activations_identifier)   
    
    data = xr.open_dataarray(os.path.join(CACHE,'activations',activations_identifier),
                             engine='netcdf4')
    
    pca_iden = load_full_identifier(model_name=model_name, features=features,
                                    layers=5, dataset=dataset, principal_components = components)
    
    if dataset == 'naturalscenes':
        from encoding_score.benchmarks.nsd import filter_activations
        IDS_PATH = pickle.load(open(os.path.join(DATA,'naturalscenes', 'nsd_ids_unshared_sample=30000'), 'rb')) 
        NSD_UNSHARED_SAMPLE = [image_id.strip('.png') for image_id in IDS_PATH]
        data = filter_activations(data, NSD_UNSHARED_SAMPLE)
        _PCA(n_components = components)._fit(pca_iden, data)
    
    elif dataset == 'majajhong':
        from encoding_score.benchmarks.majajhong import load_activations
        data = load_activations(activations_identifier, mode = 'train')
        _PCA(n_components = components)._fit(pca_iden, data)
    
    else: 
        data = data.values
        _PCA(n_components = components)._fit(pca_iden, data)
