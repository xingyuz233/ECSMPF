from typing import List
class BaseModel(object):
    def __init__(self) -> None:
        pass
    def fit(self, x_train: List[List[float]], y_train: List[int]) -> None:
        pass
    def predict(self, x_train: List[List[float]]) -> List[int]:
        pass
    def predict_proba(self, x_train: List[List[float]]) -> List[List[float]]:
        pass