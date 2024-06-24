import logging
import argparse
from config import setup_logging

from helpers.pretrained_alexnet import pretrained_alexnet_
from helpers.untrained_models import untrained_models_

def main(dataset_name, device):
    setup_logging()

    logging.info(f'\033[1m Running script for pretrained alexnet\033[0m')
    pretrained_alexnet_(dataset_name, device)

    logging.info(f'\033[1m Running script for untrained_models\033[0m')
    untrained_models_(dataset_name, device)  




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run scripts with dataset selection.")
    parser.add_argument('--dataset', required=True, help="Specify the dataset name",
                        type=str, choices=['naturalscenes', 'majajhong'])
    parser.add_argument('--device', required=False, help="Specify device name",
                        type=str, default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()

    main(args.dataset, args.device)
