from sklearn.metrics import *

def get_metrics(y_true, y_pred, y_score):# 传入真实值预测值 返回MAE RMSE MAPE R2的列表
    ans=[]
    ans.append(accuracy_score(y_true,y_pred))
    ans.append(precision_score(y_true,y_pred))
    ans.append(recall_score(y_true,y_pred))
    ans.append(roc_auc_score(y_true,y_score))
    ans.append(balanced_accuracy_score(y_true, y_pred))
    ans.append(f1_score(y_true, y_pred))
    ans.append(confusion_matrix(y_true, y_pred))
    return ans