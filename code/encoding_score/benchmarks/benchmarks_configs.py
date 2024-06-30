import os
import pickle
from dotenv import load_dotenv

load_dotenv()

DATA = os.getenv("DATA")
CACHE = os.getenv("CACHE")

PREDS_PATH = os.path.join(CACHE,'neural_preds')

MAJAJ_DATA = os.path.join(DATA,'majajhong')
MAJAJ_TRAIN_IDS =  pickle.load(open(os.path.join(MAJAJ_DATA,'majaj_train_ids'), "rb"))
MAJAJ_TEST_IDS =  pickle.load(open(os.path.join(MAJAJ_DATA,'majaj_test_ids'), "rb"))
TRAIN_IDS_DEMO =  pickle.load(open(os.path.join(MAJAJ_DATA,'majaj_train_ids_demo'), "rb"))
TEST_IDS_DEMO =  pickle.load(open(os.path.join(MAJAJ_DATA,'majaj_test_ids_demo'), "rb"))

NSD_NEURAL_DATA = os.path.join(DATA,'naturalscenes')

ALPHA_RANGE = [10**i for i in range(10)]