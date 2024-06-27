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

current_dir = os.getcwd()
if os.path.basename(current_dir) == 'demo_notebooks':
    parent_dir = os.path.dirname(current_dir)  
    grandparent_dir = os.path.dirname(parent_dir)  
    os.chdir(grandparent_dir)  
    current_dir = os.getcwd() 
RESULTS_PATH = os.path.join(current_dir, 'results')


