import os
import pickle
import functools
import gc
import logging

import torch

from config import CACHE, setup_logging

setup_logging()

def register_pca_hook(x: torch.Tensor, pca_file_name: str, n_components, 
                      device) -> torch.Tensor:
    """
    Applies a PCA transformation to the tensor x using precomputed PCA parameters.

    Args:
        x (torch.Tensor): The input tensor for which PCA should be applied.
        pca_file_name (str): The file name where PCA parameters are stored.
        n_components (int, optional): Number of principal components to keep. If None, all components are used.
        device (str): Device to perform the computations on ('cuda' or 'cpu').

    Returns:
        torch.Tensor: The transformed tensor after applying PCA.
    """
    pca_path = os.path.join(CACHE, 'pca', pca_file_name)

    with open(pca_path, 'rb') as file:
        _pca = pickle.load(file)
    
    _mean = torch.Tensor(_pca.mean_).to(device)
    _eig_vec = torch.Tensor(_pca.components_.transpose()).to(device)
    
    x = x.squeeze()
    x -= _mean
    
    if n_components is not None:
        return x @ _eig_vec[:, :n_components]
    else:
        return x @ _eig_vec


def cache(file_name_func):
    if not os.path.exists(CACHE):
        os.mkdir(CACHE)
    
    if not os.path.exists(os.path.join(CACHE, 'activations')):
        os.mkdir(os.path.join(CACHE, 'activations'))
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            file_name = file_name_func(*args, **kwargs)
            cache_path = os.path.join(CACHE, file_name)

            if os.path.exists(cache_path):
                logging.info('Activations already exist')
                return

            result = func(self, *args, **kwargs)
            result.to_netcdf(cache_path, engine='netcdf4')
            gc.collect()

        return wrapper
    return decorator