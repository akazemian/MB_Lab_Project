
import logging
import argparse
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
sys.path.insert(0, ROOT)

from code_.model_configs import analysis_cfg as cfg
from config import setup_logging
from encoding_score.init_type import init_type_
from encoding_score.linear_models import linear_models_
from encoding_score.local_connectivity import local_connectivity_
from encoding_score.non_linearity import non_linearity_
from encoding_score.random_models import random_models_

def main(dataset_name, batch_size, device):
    setup_logging()
    
    logging.info(f'\033[1m Running script for initialization types\033[0m')
    init_type_(dataset_name, cfg, batch_size, device)

    logging.info(f'\033[1m Running script for linear models\033[0m')
    linear_models_(dataset_name, cfg, batch_size, device)   

    logging.info(f'\033[1m Running script for local connectivity\033[0m')
    local_connectivity_(dataset_name, cfg, batch_size, device)   

    logging.info(f'\033[1m Running script for non linearity\033[0m')
    local_connectivity_(dataset_name, cfg, batch_size, device)       

    logging.info(f'\033[1m Running script for random models\033[0m')
    random_models_(dataset_name, cfg, 3, device) 

    logging.info(f'\033[1m Running script for non linearities\033[0m')
    non_linearity_(dataset_name, cfg, batch_size, device)  

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
