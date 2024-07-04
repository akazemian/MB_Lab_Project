import logging
import argparse
import time
import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
sys.path.insert(0, ROOT)

from model_configs import model_cfg as cfg
from config import setup_logging
from code_.encoding_score.pretrained_alexnet import pretrained_alexnet_
from code_.encoding_score.untrained_models import untrained_models_

DATASET = "majajhong_demo"

def main(device, batch_size):
    '''
    The output encoding score for each model is saved as a pandas dataframe in the results folder
    '''
    setup_logging()
    start_time = time.perf_counter()

    logging.info(f'\033[1m Running script for pretrained alexnet\033[0m')
    pretrained_alexnet_(dataset=DATASET, cfg =cfg, batch_size=batch_size, device=device)

    logging.info(f'\033[1m Running script for untrained_models\033[0m')
    untrained_models_(dataset=DATASET, cfg = cfg, batch_size=batch_size, device=device)

    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.3f} seconds") 
    
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run scripts with dataset selection.")
    parser.add_argument('--device', required=False, help="Specify device name",
                        type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument('--batchsize', required=False, help="Specify batch size",
                        type=int, default=50)
    args = parser.parse_args()

    main(args.device, args.batchsize)


# 7-8 seconds
