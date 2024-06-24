from pathlib import Path

# path to the data folder containing all data files
DATA = '/home/atlask/data/atlas/data/'

# cache path
CACHE = './.cache'

# logging 
import logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
