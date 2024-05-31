# set paths
DATA = '/data/atlas/data' # path to datasets
CACHE = '/data/atlas/.cache_new' # path to everything that will be cached


# logging 
import logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )