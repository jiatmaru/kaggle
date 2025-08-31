import lightgbm as lgb
import xgboost as xgb
from sklearn import ensemble
from sklearn import tree

models={
    "decision_tree_gini": tree.DecisionTreeClassifier(
        criterion="gini"
    ),
    "decision_tree_entropy": tree.DecisionTreeClassifier(
        criterion="entropy"
    ),
    "rf": ensemble.RandomForestClassifier(),
    "lgbm": lgb.LGBMClassifier(
        objective="binary", # 二値分類タスクを指定
        metric="binary_logloss",
    ),
    "xgboost": xgb.XGBClassifier(
        objective="binary:logistic", # 二値分類タスクを指定
        eval_metric="logloss",
        use_label_encoder=False # 推奨される設定
    ),
}