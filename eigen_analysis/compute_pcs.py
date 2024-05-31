import xarray as xr
import os
import pickle

from eigen_analysis.utils import _PCA
from model_activations.models.utils import load_model, load_full_identifier
from model_activations.activation_extractor import Activations
from encoding_score.regression.get_betas import NeuralRegression
from encoding_score.regression.scores_tools import get_bootstrap_rvalues
from encoding_score.benchmarks.majajhong import load_activations
from encoding_score.benchmarks.nsd import filter_activations
from config import CACHE, DATA, setup_logging
setup_logging()

IDS_PATH = pickle.load(open(os.path.join(DATA,'naturalscenes', 'nsd_ids_unshared_sample=30000'), 'rb')) 
NSD_UNSHARED_SAMPLE = [image_id.strip('.png') for image_id in IDS_PATH]
SHARED_IDS = pickle.load(open(os.path.join(DATA,'naturalscenes','nsd_ids_shared'), 'rb'))

def compute_model_pcs(model_name, features, layers, dataset, components, device):
    
    activations_identifier = load_full_identifier(model_name=model_name, features=features, layers=layers, dataset=dataset)
    model = load_model(model_name=model_name, features=features, layers=layers)
    
    Activations(model=model, dataset=dataset, device= device).get_array(activations_identifier)   
    
    data = xr.open_dataarray(os.path.join(CACHE,'activations',activations_identifier),
                             engine='netcdf4')
    
    pca_iden = load_full_identifier(model_name=model_name, features=features,
                                    layers=5, dataset=dataset, principal_components = components)
    
    if dataset == 'naturalscenes':
        data = filter_activations(data, NSD_UNSHARED_SAMPLE)
        _PCA(n_components = components)._fit(pca_iden, data)
    
    elif dataset == 'majajhong':
        data = load_activations(activations_identifier, mode = 'train')
        _PCA(n_components = components)._fit(pca_iden, data)
    
    else: 
        data = data.values
        _PCA(n_components = components)._fit(pca_iden, data)
