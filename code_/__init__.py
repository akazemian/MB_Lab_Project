import os
from config import RESULTS_PATH

if not os.path.exists(RESULTS_PATH):
    os.mkdir(RESULTS_PATH)