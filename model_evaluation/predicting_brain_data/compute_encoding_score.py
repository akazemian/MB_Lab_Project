import os 
import sys
sys.path.append(os.getenv('BONNER_ROOT_PATH'))
import warnings
warnings.filterwarnings('ignore')

from image_tools.processing import *
from model_evaluation.predicting_brain_data.regression.scorer import EncodingScore
from model_evaluation.utils import get_activations_iden
from model_features.activation_extractor import Activations
from model_features.models.models import load_model_dict
import gc

# define local variables

# DATASET = 'majajhong'
# REGIONS = ['V4','IT']

DATASET = 'naturalscenes'
REGIONS = ['V1','V2','V3','V4','general']
HOOK = None
DEVICE = 'cuda' 
    
models = ['expansion_10','expansion_100','expansion_1000','expansion_10000','expansion_first_256_pcs',
          'expansion_linear','fully_random','fully_connected_10','fully_connected_100','fully_connected_1000',
          'fully_connected_10000','fully_connected_3_layers_10','fully_connected_3_layers_100',
          'fully_connected_3_layers_1000','fully_connected_3_layers_10000','alexnet_conv1','alexnet_conv2',
          'alexnet_conv3','alexnet_conv4','alexnet_conv5','alexnet_test','alexnet_untrained_conv1',
          'alexnet_untrained_conv2','alexnet_untrained_conv3','alexnet_untrained_conv4',
          'alexnet_untrained_conv5']     
 


for model_name in models:
    
    print(model_name)
    model_info = load_model_dict(model_name)
    
    for region in REGIONS:
        
        activations_identifier = get_activations_iden(model_info, DATASET)

        Activations(model=model_info['model'],
                    layer_names=model_info['layers'],
                    dataset=DATASET,
                    hook = HOOK,
                    device= DEVICE,
                    batch_size = 50,
                    compute_mode = 'slow').get_array(activations_identifier)   

        scores_identifier = activations_identifier + '_' + region

        EncodingScore(model_name=model_info['iden'],
                       activations_identifier=activations_identifier,
                       dataset=DATASET,
                       region=region).get_scores(scores_identifier=scores_identifier)
        gc.collect()


