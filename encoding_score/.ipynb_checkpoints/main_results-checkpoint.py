import os
import runpy
import logging

from config import CACHE, setup_logging
setup_logging()

script_list = ['alexnet', 'untrained_models']
i = 0

for script in script_list: 
    logging.info(f'\033[1m Running script for {script} (total scripts left = {len(script_list) - i}) \033[0m')
    script_path = os.path.join(os.getcwd(),f'{script}_score.py')
    runpy.run_path(script_path)
    i+=1
