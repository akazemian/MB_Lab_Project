import os
import logging

from config import CACHE, setup_logging

from ..benchmarks.majajhong import majajhong_scorer, majajhong_get_best_layer_scores
from ..benchmarks.nsd import nsd_scorer, nsd_get_best_layer_scores

setup_logging()

class NeuralRegression():
    def __init__(self,
                 activations_identifier: str|list,
                 dataset: str,
                 region:str,
                 device:str):
        
        self.activations_identifier = activations_identifier
        self.dataset = dataset
        self.region = region
        self.device = device

        if not os.path.exists(os.path.join(CACHE,'neural_preds')):
            os.mkdir(os.path.join(CACHE,'neural_preds'))
        
    def predict_data(self):       
        """
    
        Obtain and save the encoding score (unit-wise pearson r values) of a particular model for a particular dataset 

        Parameters
        ----------
        
        model_name:
                Name of model for which the encoding score is being obtained
        
        activations_identifier:
                Name of the file containing the model activations  
        
        dataset:
                Name of neural dataset (majajhong, naturalscenes)
        
        """

        logging.info('Predicting neural data from model activations...')        

        match self.dataset:
            
            case 'naturalscenes' | 'naturalscenes_shuffled':

                if type(self.activations_identifier) == list:
                    nsd_get_best_layer_scores(activations_identifier= self.activations_identifier, 
                                                   region= self.region,
                                                  device = self.device)                  
                else:
                    nsd_scorer(activations_identifier = self.activations_identifier, 
                                    region = self.region,
                                    device = self.device)

            case 'majajhong' | 'majajhong_shuffled':
                                
                if type(self.activations_identifier) == list:
                    majajhong_get_best_layer_scores(activations_identifier= self.activations_identifier, 
                                                         region= self.region,
                                                         device = self.device)                  
                
                else:
                    majajhong_scorer(activations_identifier = self.activations_identifier, 
                                        region = self.region,
                                        device = self.device)   

            case 'majajhong_demo' | 'majajhong_demo_shuffled':     
                                
                if type(self.activations_identifier) == list:
                    majajhong_get_best_layer_scores(activations_identifier= self.activations_identifier, 
                                                         region= self.region,
                                                         device = self.device,
                                                         demo = True)                  
                
                else:
                    majajhong_scorer(activations_identifier = self.activations_identifier, 
                                        region = self.region,
                                        device = self.device,
                                        demo = True)       
        return 


    
        
    