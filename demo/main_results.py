import logging
import argparse
import time

from model_configs import model_cfg as cfg
from config import setup_logging
from code.encoding_score.pretrained_alexnet import pretrained_alexnet_
from code.encoding_score.untrained_models import untrained_models_

DATASET = "majajhong_demo"

def main(device):
    setup_logging()
    start_time = time.perf_counter()

    logging.info(f'\033[1m Running script for pretrained alexnet\033[0m')
    pretrained_alexnet_(dataset=DATASET, cfg =cfg, device=device)

    logging.info(f'\033[1m Running script for untrained_models\033[0m')
    untrained_models_(dataset=DATASET, cfg = cfg, device=device)

    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.3f} seconds") 
    
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run scripts with dataset selection.")
    parser.add_argument('--device', required=False, help="Specify device name",
                        type=str, default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()

    main(args.device)


# 7-8 seconds
