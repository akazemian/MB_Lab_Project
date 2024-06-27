import time
import logging
import argparse

from model_configs import analysis_cfg as cfg
from config import setup_logging
from code.encoding_score.init_type import init_type_
from code.encoding_score.linear_models import linear_models_
from code.encoding_score.local_connectivity import local_connectivity_
from code.encoding_score.non_linearity import non_linearity_
from code.encoding_score.random_models import random_models_

DATASET = "majajhong_demo"

    
def main(batch_size, device):
    '''
    The output encoding score for each type of analysis is saved as a pandas dataframe in the results folder
    '''
    setup_logging()
    
    logging.info(f'\033[1m Running script for initialization types\033[0m')
    init_type_(DATASET, cfg, batch_size, device)

    logging.info(f'\033[1m Running script for linear models\033[0m')
    linear_models_(DATASET, cfg, batch_size, device)   

    logging.info(f'\033[1m Running script for local connectivity\033[0m')
    local_connectivity_(DATASET, cfg, batch_size, device)   

    logging.info(f'\033[1m Running script for non linearity\033[0m')
    local_connectivity_(DATASET, cfg, batch_size, device)       

    logging.info(f'\033[1m Running script for random models\033[0m')
    random_models_(DATASET, cfg, 3, device) 

    logging.info(f'\033[1m Running script for non linearities\033[0m')
    non_linearity_(DATASET, cfg, batch_size, device)  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run scripts with dataset selection.")
    parser.add_argument('--dataset', required=True, help="Specify the dataset name",
                        type=str, choices=['naturalscenes', 'majajhong'])
    parser.add_argument('--device', help="Specify device name",
                        type=str, default = "cuda", choices=['cpu', 'cuda'])
    parser.add_argument('--batchsize', help="Specify batch size",
                        type=int, default=50)
    args = parser.parse_args()

    main(args.dataset, args.batchsize, args.device)
