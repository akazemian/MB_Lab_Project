
import logging
import argparse

from config import setup_logging
from helpers.init_type import init_type_
from helpers.linear_models import linear_models_
from helpers.local_connectivity import local_connectivity_
from helpers.non_linearity import non_linearity_
from helpers.random_models import random_models_

def main(dataset_name, device):
    setup_logging()
    
    logging.info(f'\033[1m Running script for initialization types\033[0m')
    init_type_(dataset_name, device)

    logging.info(f'\033[1m Running script for linear models\033[0m')
    linear_models_(dataset_name, device)   

    logging.info(f'\033[1m Running script for local connectivity\033[0m')
    local_connectivity_(dataset_name, device)   

    logging.info(f'\033[1m Running script for non linearity\033[0m')
    local_connectivity_(dataset_name, device)       

    logging.info(f'\033[1m Running script for random models\033[0m')
    random_models_(dataset_name, device)        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run scripts with dataset selection.")
    parser.add_argument('--dataset', required=True, help="Specify the dataset name",
                        type=str, choices=['naturalscenes', 'majajhong'])
    parser.add_argument('--device', required=False, help="Specify device name",
                        type=str, choices=['cpu', 'cuda'])
    args = parser.parse_args()

    main(args.dataset, args.device)
