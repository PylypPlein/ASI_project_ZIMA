import os

from autogluon.tabular import TabularPredictor

MODEL_PATH = os.environ["MODEL_PATH"]


predictor = TabularPredictor.load(MODEL_PATH,require_version_match=False)


print("PREDICTOR PATH:", predictor.path)

print("LEARNER PATH:", predictor._learner.path)

