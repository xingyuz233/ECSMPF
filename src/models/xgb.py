from typing import List
import yaml
from src.models.base import BaseModel
from xgboost.sklearn import XGBClassifier
config = yaml.load(open('config/config.yaml'), Loader=yaml.FullLoader)

class XGBOOST(BaseModel):
    def __init__(self) -> None:
        super(XGBOOST, self).__init__()
        self.model = XGBClassifier(
            **config['model']['hyperameters']['xgboost']
        )
    def fit(self, x_train: List[List[float]], y_train: List[int]) -> None:
        return self.model.fit(x_train, y_train)

    def predict(self, x_train: List[List[float]]) -> List[int]:
        return self.model.predict(x_train)
    
    def predict_proba(self, x_train: List[List[float]]) -> List[List[float]]:
        return self.model.predict_proba(x_train)
    
    def get_feature_importance(self):
        return self.model.feature_importances_
