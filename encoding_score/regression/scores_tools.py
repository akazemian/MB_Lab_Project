import gc
import os
import logging

import pandas as pd
import torch
import pickle
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from model_activations.models.utils import load_full_identifier, find_best_layer_iden
from config import CACHE, setup_logging

setup_logging()

PREDS_PATH = os.path.join(CACHE,'neural_preds')
BOOTSTRAP_RESULTS_PATH = os.path.join(CACHE,'bootstrap_r_values')

if not os.path.exists(BOOTSTRAP_RESULTS_PATH):
    os.mkdir(BOOTSTRAP_RESULTS_PATH)

def compute_similarity_matrix(features):
    """
    Compute the similarity matrix (using Pearson correlation) for a set of features.
    """
    # Compute the pairwise distances (using correlation) and convert to similarity
    return 1 - squareform(pdist(features, 'correlation'))


def rsa(features1, features2):
    """
    Perform Representational Similarity Analysis between two sets of features.
    """
    # Compute similarity matrices for both sets of features
    sim_matrix_1 = compute_similarity_matrix(features1)
    sim_matrix_2 = compute_similarity_matrix(features2)

    # Flatten the upper triangular part of the matrices
    upper_tri_indices = np.triu_indices_from(sim_matrix_1, k=1)
    sim_matrix_1_flat = sim_matrix_1[upper_tri_indices]
    sim_matrix_2_flat = sim_matrix_2[upper_tri_indices]

    # Compute the Spearman correlation between the flattened matrices
    correlation, p_value = spearmanr(sim_matrix_1_flat, sim_matrix_2_flat)

    return correlation, p_value


def pearson_r_(x, y):
    """
    Compute Pearson correlation coefficients for batches of bootstrap samples.

    Parameters:
    x (torch.Tensor): A 3D tensor of shape (n_bootstraps, n_samples, n_features).
    y (torch.Tensor): A 3D tensor of shape (n_bootstraps, n_samples, n_features).

    Returns:
    torch.Tensor: 1D tensor of Pearson correlation coefficients for each bootstrap.
    """
    # Ensure the input tensors are of the same shape
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape")

    # Mean-centering the data
    x_mean = torch.mean(x, dim=2, keepdim=True)
    y_mean = torch.mean(y, dim=2, keepdim=True)
    x = x - x_mean
    y = y - y_mean

    # Calculating Pearson Correlation Coefficient
    sum_sq_x = torch.sum(x ** 2, axis=2)
    sum_sq_y = torch.sum(y ** 2, axis=2)
    sum_coproduct = torch.sum(x * y, axis=2)
    denominator = torch.sqrt(sum_sq_x * sum_sq_y)

    # Avoid division by zero
    denominator = torch.where(denominator != 0, denominator, torch.ones_like(denominator))

    r_values = sum_coproduct / denominator

    # Average across the samples in each bootstrap
    mean_r_values = torch.mean(r_values, axis=1)
    return mean_r_values


def percentile(t, q):
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result


def get_bootstrap_rvalues(model_name, features, layers, subjects, dataset, region, all_sampled_indices, 
                       init_type=['kaiming_uniform'], non_linearity=['relu'], principal_components=[None],
                       batch_size=50, n_bootstraps=1000, device='cuda',file_name = None):
    

    if file_name == None:
        file_path = os.path.join(BOOTSTRAP_RESULTS_PATH, model_name + '_' + region + '.pkl')
    else:
        file_path = os.path.join(BOOTSTRAP_RESULTS_PATH, file_name + '_' + region + '.pkl')
        
    if not os.path.exists(file_path):
        logging.info('Computing bootstrap distribution of r-values...')
        data_dict = {'model': [], 'features': [], 'pcs': [], 'init_type': [], 'nl_type': [],
                     'score': [], 'lower': [], 'upper': []}
        
        for features_ in features:
            for principal_components_ in principal_components:
                for non_linearity_ in non_linearity:
                    for init_type_ in init_type:

                        try:
                            if model_name == 'alexnet':
                                activations_identifier = find_best_layer_iden(dataset, region)
                            else:
                                activations_identifier = load_full_identifier(model_name=model_name, 
                                                                          features=features_, 
                                                                          layers = layers,
                                                                          dataset=dataset,
                                                                          init_type = init_type_,
                                                                          non_linearity = non_linearity_,
                                                                          principal_components = principal_components_)
        
                            bootstrap_dist = compute_bootstrap_distribution(activations_identifier, subjects, region,
                                                                    all_sampled_indices, batch_size, 
                                                                    n_bootstraps, dataset, device)
                            
                            update_data_dict(data_dict, activations_identifier, features_, principal_components_,
                                     init_type_, non_linearity_, bootstrap_dist.cpu())
                            
                        except FileNotFoundError:
                            logging.warning(f'file with identifier: {activations_identifier} does not exist')
                            pass
                        
        df = pd.DataFrame.from_dict(data_dict)

        with open((file_path), 'wb') as file:
            pickle.dump(df, file)
        logging.info('Bootstrap r-values are now saved in cache')
        
        del bootstrap_dist, data_dict, df
        gc.collect()
        return 
    else:
        logging.info('Bootstrap r-values already exist')
        return
        

def compute_bootstrap_distribution(identifier, subjects, region, all_sampled_indices, batch_size, n_bootstraps, dataset, device):
    score_sum = torch.zeros(n_bootstraps).to(device)
    for subject in tqdm(subjects):
        preds, test = load_data(identifier, region, subject, dataset)
        all_sampled_preds = preds[all_sampled_indices]
        all_sampled_tests = test[all_sampled_indices]
        score_sum += batch_pearson_r(all_sampled_tests, all_sampled_preds, batch_size, n_bootstraps, device)
        
        del preds, test, all_sampled_preds, all_sampled_tests
        gc.collect()
        
    return score_sum / len(subjects)


def batch_pearson_r(all_sampled_tests, all_sampled_preds, batch_size, n_bootstraps, device):
    #r_values = []
    r_values = torch.Tensor([])
    i = 0
    while i < n_bootstraps:
        # Compute Pearson r for all bootstraps at once
        mean_r_values = pearson_r_(all_sampled_tests[i:i + batch_size, :, :].to(device),
                                   all_sampled_preds[i:i + batch_size, :, :].to(device))
        r_values = torch.concat((r_values.to(device), mean_r_values))
        #r_values.extend(mean_r_values.tolist())
        i += batch_size
        
        del mean_r_values
        gc.collect()
        
    return r_values


def load_data(identifier, region, subject, dataset):
    
    with open(os.path.join(PREDS_PATH, f'{identifier}_{region}_{subject}.pkl'), 'rb') as file:
            preds = torch.Tensor(pickle.load(file))
    if 'naturalscenes' in dataset:
        from encoding_score.benchmarks.nsd import load_nsd_data
        _, neural_data_test = load_nsd_data(mode='shared', subject=subject, region=region)
        test = torch.Tensor(neural_data_test['beta'].values)
    elif 'majajhong' in dataset:
        from encoding_score.benchmarks.majajhong import load_majaj_data
        test = load_majaj_data(subject, region, 'test')
    else:
        raise ValueError('Invalid dataset name')
    
    return preds, test


def update_data_dict(data_dict, model_name, feature, component, init_type, non_linearity, bootstrap_dist):
    data_dict['model'].append(model_name)
    data_dict['features'].append(str(feature))
    data_dict['pcs'].append(str(component))
    data_dict['init_type'].append(init_type)
    data_dict['nl_type'].append(non_linearity)
    data_dict['score'].append(torch.mean(bootstrap_dist))
    data_dict['lower'].append(percentile(bootstrap_dist, 2.5))
    data_dict['upper'].append(percentile(bootstrap_dist, 97.5))

