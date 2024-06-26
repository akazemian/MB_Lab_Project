# path to the data folder containing all data files
DATA = '/home/atlask/data/atlas/data/'

# cache path
CACHE = '/home/atlask/data/atlas/.cache'

# logging 
import logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
