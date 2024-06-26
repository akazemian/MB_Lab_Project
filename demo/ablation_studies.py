import time
import logging
import argparse
import sys
sys.path.append('./Code')

from model_configs import analysis_cfg as cfg
from config import setup_logging
from code.encoding_score.init_type import init_type_
from code.encoding_score.linear_models import linear_models_
from code.encoding_score.local_connectivity import local_connectivity_
from code.encoding_score.non_linearity import non_linearity_
from code.encoding_score.random_models import random_models_

DATASET = "majajhong_demo"

def main(device):
    setup_logging()
    start_time = time.perf_counter()
    
    logging.info(f'\033[1m Running script for initialization types\033[0m')
    init_type_(dataset=DATASET, cfg=cfg, device=device)

    logging.info(f'\033[1m Running script for linear models\033[0m')
    linear_models_(dataset=DATASET, cfg=cfg, device=device)   

    logging.info(f'\033[1m Running script for local connectivity\033[0m')
    local_connectivity_(dataset=DATASET, cfg=cfg, device=device)   

    logging.info(f'\033[1m Running script for non linearity\033[0m')
    local_connectivity_(dataset=DATASET, cfg=cfg, device=device)       

    logging.info(f'\033[1m Running script for random models\033[0m')
    random_models_(dataset=DATASET, cfg=cfg, device=device) 

    logging.info(f'\033[1m Running script for non linearities\033[0m')
    non_linearity_(dataset=DATASET, cfg=cfg, device=device)  

    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.3f} seconds") 
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run scripts with dataset selection.")
    parser.add_argument('--device', help="Specify device name",
                        type=str, default = "cuda", choices=['cpu', 'cuda'])
    args = parser.parse_args()

    main(args.device)


    # 80 seconds
