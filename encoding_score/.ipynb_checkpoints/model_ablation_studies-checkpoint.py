import os
import logging
import subprocess
import argparse

from config import CACHE, setup_logging

def main(dataset_name):
    setup_logging()
    script_list = ['random_models','init_type', 'non_linearity','local_connectrivity','linear_models']

    for i, script in enumerate(script_list):
        logging.info(f'\033[1m Running script for {script} (total scripts left = {len(script_list) - i}) \033[0m')
        script_path = os.path.join(os.getcwd(), 'encoding_score', f'{script}.py')
        command = ['python', script_path, f'--dataset={dataset_name}']
        result = subprocess.run(command, text=True, capture_output=True)

        # Log the output and errors
        logging.info(result.stdout)
        if result.stderr:
            logging.error(result.stderr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run scripts with dataset selection.")
    parser.add_argument('--dataset', required=True, help="Specify the dataset name")
    args = parser.parse_args()

    main(args.dataset)
