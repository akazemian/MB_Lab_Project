import os
import logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )


# path to the data folder containing all data files
DATA = '/home/atlask/data/atlas/data/'

# cache path
CACHE = '/home/atlask/data/atlas/.cache'

# encoding scores results
current_dir = os.getcwd()
if os.path.basename(current_dir) == 'demo':
    os.chdir(os.path.dirname(current_dir))
    current_dir = os.getcwd()
RESULTS_PATH = os.path.join(current_dir, 'results')


