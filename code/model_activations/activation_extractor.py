import os
import warnings
from collections import OrderedDict
from typing import Optional
import logging

from tqdm import tqdm
import torch
from torch import nn
from torch.autograd import Variable
import xarray as xr
import numpy as np
from dotenv import load_dotenv

from code.tools.loading import load_image_paths, get_image_labels
from code.tools.processing import ImageProcessor
from .utils import cache, register_pca_hook
from config import setup_logging

load_dotenv()
setup_logging()

warnings.warn('my warning')
SUBMODULE_SEPARATOR = '.'
CACHE = os.getenv("CACHE")
PATH_TO_PCA = os.path.join(CACHE, 'pca')

class PytorchWrapper:
    """
    A wrapper class for handling PyTorch models with enhanced functionality such as
    layer-specific activation extraction and applying PCA on activations.
    """
    def __init__(self, model: nn.Module, identifier: str, device: str, forward_kwargs: dict = None) -> None:
        self._device = device
        self._model = model
        self._model.to(self._device)
        self._forward_kwargs = forward_kwargs or {}
        self.identifier = identifier


    def get_activations(self, images: list, layer_name: str, *args, **kwargs) -> OrderedDict:
        """
        Extracts activations from the specified layers.

        Args:
        images (list): A list of images to process.
        layer_name (str): Name of the layer from which to extract activations.

        Returns:
        OrderedDict: A dictionary containing layer activations.
        """
        images = [torch.from_numpy(image) if not isinstance(image, torch.Tensor) else image for image in images]
        images = Variable(torch.stack(images))
        images = images.to(self._device)
        self._model.eval()

        layer_results = OrderedDict()
        layer = self.get_layer(layer_name)
        hook = self.register_hook(layer, layer_name, target_dict=layer_results, *args, **kwargs)

        with torch.no_grad():
            self._model(images, **self._forward_kwargs)
        hook.remove()
        return layer_results

    def get_layer(self, layer_name: str) -> nn.Module:
        """
        Retrieves a submodule by its name from the model.

        Args:
        layer_name (str): The name of the layer to retrieve.

        Returns:
        nn.Module: The specified submodule.
        """
        module = self._model
        for part in layer_name.split(SUBMODULE_SEPARATOR):
            module = module._modules.get(part)
            assert module is not None, f"No submodule found for layer {layer_name}, at part {part}"
        return module

    @classmethod
    def _tensor_to_numpy(cls, output:torch.Tensor) -> np.ndarray: 
        try:
            return output.cpu().data.numpy()
        except AttributeError:
            return output

    def register_hook(self, layer: nn.Module, layer_name: str, target_dict: OrderedDict, 
                      n_components: Optional[int], pca_iden: Optional[str]) -> torch.utils.hooks.RemovableHandle:
        """
        Registers a forward hook on a layer to capture or transform its outputs during the forward pass.

        Args:
        layer (nn.Module): The layer to which the hook will be attached.
        layer_name (str): The name of the layer.
        target_dict (OrderedDict): Dictionary to store layer outputs.
        n_components (Optional[int]): Number of PCA components.
        pca_iden (Optional[str]): Identifier for where to get the PCs from.

        Returns:
        torch.utils.hooks.RemovableHandle: A handle that can be used to remove the hook.
        """
        def hook_function(_layer: nn.Module, _input: torch.Tensor, output: torch.Tensor, name: str = layer_name):    
            if pca_iden is not None:
                target_dict[name] = register_pca_hook(x=output, pca_file_name=pca_iden,
                                                      n_components=n_components, device=self._device)
            else:
                target_dict[name] = output

        hook = layer.register_forward_hook(hook_function)
        return hook 

import os
import logging
from typing import Optional
import torch
import xarray as xr
from tqdm import tqdm
import numpy as np
from torch import nn

class Activations:
    """
    Manages the extraction and caching of model activations for a specified model
    and dataset, with optional hooks for transformation like PCA.
    
    Attributes:
        device (str): The device on which computations are to be performed ('cuda' or 'cpu').
    """
    def __init__(self, model: nn.Module, dataset: str, 
                 layer_name: str = 'last', pca_iden: Optional[str] = None, 
                 n_components: Optional[int] = None, batch_size: int = 64, device: str = 'cuda') -> None:
        self.model = model
        self.dataset = dataset
        self.layer_name = layer_name
        self.pca_iden = pca_iden
        self.n_components = n_components
        self.batch_size = batch_size
        self.device = device

    @staticmethod
    def cache_file(iden: str) -> str:
        return os.path.join('activations', iden)

    @cache(cache_file)
    def get_array(self, iden: str) -> xr.Dataset:
        """
        Gets activations for a batch of images, caching the results.

        Args:
            iden (str): A unique identifier used to cache and retrieve the dataset.
            model (nn.Module): The neural network model from which to extract activations.
            dataset (str): The name of the dataset from which image paths and labels will be fetched.
            layer_names (list[str]): The list of model layer names from which activations will be extracted.
            batch_size (int): The number of images to process in each batch.
            pca_iden (str, optional): Identifier for file where the PCs are stored.
            n_components (int, optional): The number of components to retain if PCA is applied.

        Returns:
            xr.Dataset: An xarray Dataset containing activations, potentially transformed by PCA, and indexed by image labels.
        """
        
        pytorch_model = PytorchWrapper(model=self.model, identifier=iden, device=self.device)
        images, labels = load_image_data(dataset_name=self.dataset, device=self.device)

        # Preallocate the entire array
        num_samples = len(images)
        first_batch_activations = get_batch_activations(images[:self.batch_size], labels[:self.batch_size], pytorch_model, self.layer_name, self.pca_iden, self.n_components)
        num_features = first_batch_activations.dims['features']

        all_activations = torch.zeros((num_samples, num_features), device='cpu')
        all_labels = []

        logging.info('Batched activations...')
        pbar = tqdm(total=len(images) // self.batch_size)
        i = 0
        while i < len(images):
            batch_images = images[i:i + self.batch_size]
            batch_labels = labels[i:i + self.batch_size]
            batch_data_final = get_batch_activations(batch_images, batch_labels, pytorch_model, self.layer_name, self.pca_iden, self.n_components)

            # Fill the preallocated array
            batch_activations = batch_data_final['x'].values
            all_activations[i:i + len(batch_images)] = torch.tensor(batch_activations, device= "cpu")
            all_labels.extend(batch_labels)

            i += self.batch_size
            pbar.update(1)

        pbar.close()

        # Create the xarray Dataset
        ds = xr.Dataset(
            data_vars=dict(x=(["presentation", "features"], all_activations.cpu())),
            coords={'stimulus_id': (['presentation'], all_labels)}
        )

        # Save to disk
        logging.info('Model activations are saved in cache')
        return ds

def get_batch_activations(images: torch.Tensor,
                          labels: list[str],
                          model: PytorchWrapper,
                          layer_name: str,
                          pca_iden: Optional[str],
                          n_components: Optional[int]) -> xr.Dataset:
    """
    Processes batches of images or paths to extract neural network activations, applying hooks if specified.

    Args:
    dataset (str): The name of the image dataset.
    image_paths (list[str], optional): Paths to the images.
    image_labels (list[str]): Labels for each image, used for indexing in the output dataset.
    model (PytorchWrapper): The model wrapper from which activations will be extracted.
    layer_name (list[str]): Name of the layer from which activations are to be extracted.
    pca_iden (Optional[str]): Identifier for the PCA components if PCA is applied.
    n_components (Optional[int]): Number of PCA components to keep.
    device (str): The device on which to process the images.
    batch_size (int, optional): Size of the batch to process at one time.

    Returns:
    xr.Dataset: An xarray Dataset containing the activations indexed by image labels.
    """
    
    activations_dict = model.get_activations(images=images,
                                             layer_name=layer_name,
                                             n_components=n_components,
                                             pca_iden=pca_iden)
    activations_b = activations_dict[layer_name]
    activations_b = torch.tensor(activations_b.reshape(activations_dict[layer_name].shape[0], -1))
    ds = xr.Dataset(
        data_vars=dict(x=(["presentation", "features"], activations_b.cpu())),
        coords={'stimulus_id': (['presentation'], labels)}
    )
    return ds



# class Activations:
#     """
#     Manages the extraction and caching of model activations for a specified model
#     and dataset, with optional hooks for transformation like PCA.
    
#     Attributes:
#         device (str): The device on which computations are to be performed ('cuda' or 'cpu').
#     """
#     def __init__(self, model: nn.Module, dataset: str, 
#                   layer_name: str = 'last', pca_iden: Optional[str] = None, 
#                   n_components: Optional[int] = None, batch_size: int = 64, device: str = 'cuda') -> None:
#         self.model = model
#         self.dataset = dataset
#         self.layer_name = layer_name
#         self.pca_iden = pca_iden
#         self.n_components = n_components
#         self.batch_size = batch_size
#         self.device = device

#     @staticmethod
#     def cache_file(iden: str) -> str:
#         return os.path.join('activations', iden)

#     @cache(cache_file)
#     def get_array(self, iden: str) -> xr.Dataset:
#         """
#         Gets activations for a batch of images, caching the results.

#         Args:
#             iden (str): A unique identifier used to cache and retrieve the dataset.
#             model (nn.Module): The neural network model from which to extract activations.
#             dataset (str): The name of the dataset from which image paths and labels will be fetched.
#             layer_names (list[str]): The list of model layer names from which activations will be extracted.
#             batch_size (int): The number of images to process in each batch.
#             pca_iden (str, optional): Identifier for file where the PCs are stored.
#             n_components (int, optional): The number of components to retain if PCA is applied.

#         Returns:
#             xr.Dataset: An xarray Dataset containing activations, potentially transformed by PCA, and indexed by image labels.
#         """
        
#         pytorch_model=PytorchWrapper(model=self.model, identifier=iden, device=self.device)
#         images, labels = load_image_data(dataset_name=self.dataset, device=self.device)
        
        
#         logging.info('Batched activations...')
#         ds_list = []
#         pbar = tqdm(total=len(images) // self.batch_size)
#         i = 0
#         while i < len(images):
#             batch_data_final = get_batch_activations(images=images[i:i + self.batch_size, :],
#                                                      labels=labels[i:i + self.batch_size],
#                                                      model=pytorch_model,
#                                                      layer_name=self.layer_name,
#                                                      pca_iden=self.pca_iden,
#                                                      n_components=self.n_components,
#                                                     )

#             ds_list.append(batch_data_final)
#             i += self.batch_size
#             pbar.update(1)

#         pbar.close()

#         data = xr.concat(ds_list, dim='presentation')
#         logging.info('Model activations are saved in cache')
#         return data

# def get_batch_activations(images: torch.Tensor,
#                           labels: list[str],
#                           model: PytorchWrapper,
#                           layer_name: str,
#                           pca_iden: Optional[str],
#                           n_components: Optional[int]) -> xr.Dataset:
#     """
#     Processes batches of images or paths to extract neural network activations, applying hooks if specified.

#     Args:
#     dataset (str): The name of the image dataset.
#     image_paths (list[str], optional): Paths to the images.
#     image_labels (list[str]): Labels for each image, used for indexing in the output dataset.
#     model (PytorchWrapper): The model wrapper from which activations will be extracted.
#     layer_name (list[str]): Name of the layer from which activations are to be extracted.
#     pca_iden (Optional[str]): Identifier for the PCA components if PCA is applied.
#     n_components (Optional[int]): Number of PCA components to keep.
#     device (str): The device on which to process the images.
#     batch_size (int, optional): Size of the batch to process at one time.

#     Returns:
#     xr.Dataset: An xarray Dataset containing the activations indexed by image labels.
#     """
    
#     activations_dict = model.get_activations(images=images,
#                                              layer_name=layer_name,
#                                              n_components=n_components,
#                                              pca_iden=pca_iden)
#     activations_b = activations_dict[layer_name]
#     activations_b = torch.Tensor(activations_b.reshape(activations_dict[layer_name].shape[0], -1))
#     ds = xr.Dataset(
#             data_vars=dict(x=(["presentation", "features"], activations_b.cpu())),
#             coords={'stimulus_id': (['presentation'], labels)})
#     return ds

   
    
def load_image_data(dataset_name:str, device:str):
    """
    Loads image paths and their corresponding labels for a given dataset.

    Args:
    dataset_name (str): The name of the dataset from which to load images. 

    Returns:
    tuple: A tuple containing:
           - image_paths (list[str]): A list of file paths corresponding to images.
           - image_labels (list[str]): A list of labels associated with the images. 
    """
    image_paths = load_image_paths(dataset_name=dataset_name)
    images = ImageProcessor(device=device).process(image_paths=image_paths, dataset=dataset_name)
    labels = get_image_labels(dataset_name = dataset_name, image_paths=image_paths)

    return images, labels

