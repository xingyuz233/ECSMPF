from typing import List
import pandas as pd
import yaml
from src.models.base import BaseModel
from src.models.xgb import XGBOOST
from src.models.im_ensemble import IM_ENSEMBLE

config = yaml.load(open('config/config.yaml'), Loader=yaml.FullLoader)
model_constructor_dict = {
    'xgboost': XGBOOST,
    'im_ensemble': IM_ENSEMBLE
}

class ModelAPI(BaseModel):
    def __init__(self, model_name) -> None:
        super(ModelAPI, self).__init__()
        self.model = model_constructor_dict[model_name]()

    def fit(self, x_train: List[List[float]], y_train: List[int]) -> None:
        return self.model.fit(x_train, y_train)
    
    def predict(self, x_train: List[List[float]]) -> List[int]:
        return self.model.predict(x_train)
    
    def predict_proba(self, x_train: List[List[float]]) -> List[List[float]]:
        return self.model.predict_proba(x_train)

    def get_feature_importance(self):
        return self.model.get_feature_importance()

