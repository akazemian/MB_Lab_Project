# set paths
DATA = '/home/atlask/data/atlas/repo_data' # path to datasets
CACHE = '/home/atlask/data/atlas/.repo_cache' # path to everything that will be cached


# logging 
import logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
