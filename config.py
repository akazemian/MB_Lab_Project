# set paths
DATA = '/home/atlask/data/atlas/repo_data' # path to datasets
CACHE = '/home/atlask/data/atlas/.repo_cache' # path to everything that will be cached
DROPBOX_ACCESS_TOKEN = 'sl.B24WLf-BkKFSJVmE_Cv0KGUGbJKdgeLtXO3irOyr9cZoTS8J4lpU6BLdm3f5YAt5RSN5mcXMjl5LzaREG2nFgchyA1Bk7kg0j8wXQAhd8pwJ715U8INKiJOCjPhEVsuCk14X46Q1Twiy3ceR1UA4LgI'

# logging 
import logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
