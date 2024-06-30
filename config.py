import logging
from pathlib import Path

ROOT = Path.cwd()
RESULTS_PATH = ROOT / 'results'

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')



