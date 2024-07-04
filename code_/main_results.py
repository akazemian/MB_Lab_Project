import logging
import argparse
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
sys.path.insert(0, ROOT)

from config import setup_logging
from model_configs import model_cfg as cfg
from encoding_score.pretrained_alexnet import pretrained_alexnet_
from encoding_score.untrained_models import untrained_models_

def main(dataset_name, batch_size, device):
    setup_logging()

    logging.info(f'\033[1m Running script for pretrained alexnet\033[0m')
    pretrained_alexnet_(dataset_name, cfg, batch_size, device)

    logging.info(f'\033[1m Running script for untrained_models\033[0m')
    untrained_models_(dataset_name, cfg, batch_size, device)  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run scripts with dataset selection.")
    parser.add_argument('--dataset', required=True, help="Specify the dataset name",
                        type=str, choices=['naturalscenes', 'majajhong'])
    parser.add_argument('--device', required=False, help="Specify device name",
                        type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument('--batchsize', help="Specify batch size",
                        type=int, default=50)
    args = parser.parse_args()

    main(args.dataset, args.batchsize, args.device)
