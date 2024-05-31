import os
import runpy

PATH = f'/home/akazemi3/Desktop/untrained_models_of_visual_cortex/model_evaluation/predicting_brain_data'


for score_type in ['shuffled_pixels_score','non_linearity_score', 'init_type_score', 
                  'random_model_score', 'linear_model_score']: 

    script_path = os.path.join(PATH,f'{score_type}_score.py')
    runpy.run_path(script_path)
