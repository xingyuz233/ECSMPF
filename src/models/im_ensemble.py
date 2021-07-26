from typing import List
import yaml
from src.models.base import BaseModel
from xgboost.sklearn import XGBClassifier
from imblearn.ensemble import EasyEnsembleClassifier  # adaboost

config = yaml.load(open('config/config.yaml'), Loader=yaml.FullLoader)

class IM_ENSEMBLE(BaseModel):
    def __init__(self) -> None:
        super(IM_ENSEMBLE, self).__init__()
        self.xgb = XGBClassifier(
            **config['model']['hyperameters']['xgboost'],
            random_state=config['model']['random_state']
        )
        self.model = EasyEnsembleClassifier(
            **config['model']['hyperameters']['im_ensemble'],
            base_estimator=self.xgb, 
            random_state=config['model']['random_state']
        )
    def fit(self, x_train: List[List[float]], y_train: List[int]) -> None:
        return self.model.fit(x_train, y_train)

    def predict(self, x_train: List[List[float]]) -> List[int]:
        return self.model.predict(x_train)
    
    def predict_proba(self, x_train: List[List[float]]) -> List[List[float]]:
        return self.model.predict_proba(x_train)
    
    def get_feature_importance(self):
        return self.model.feature_importances_
