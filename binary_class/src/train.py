# train.py (完全版の例)

import pandas as pd
import os
import importlib.util
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import sys

# --- パスとモジュールの設定 ---
PROJECT_ROOT = "/content/drive/MyDrive/kaggle/binary_class"
src_path = os.path.join(PROJECT_ROOT, 'src')
sys.path.append(src_path)

def load_module_from_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module

config = load_module_from_path(os.path.join(src_path, 'config.py'), 'config')
model_dispatcher = load_module_from_path(os.path.join(src_path, 'model_dispatcher.py'), 'model_dispatcher')
roc_metrics = load_module_from_path(os.path.join(src_path, 'roc.py'), 'roc_metrics')
from preprocessor import CustomPreprocessor

# ... (データの読み込みと準備のコード) ...
df = pd.read_csv(os.path.join(PROJECT_ROOT, config.training_file.replace('train.csv', 'train_folds.csv')))
y_df = df[['y']]
df = df.drop('y', axis=1)
# preprocessorを使って前処理
preprocessor = CustomPreprocessor()
df = preprocessor.fit_transform(df) # この行を前に追加する必要があります
# 前処理されたデータフレームにy_dfを結合し直す
df = pd.concat([df, y_df], axis=1)
# 念のため、列名を確認
print("前処理後のデータフレームの列名:", df.columns)
df.rename(columns={col: col.replace('.', '').replace('_', '').replace(' ', '') for col in df.columns}, inplace=True)
target = 'y' 
features = [col for col in df.columns if col not in ['kfold', 'id','y']] # 'id'も除外します

# 全てのフォールドの評価結果を格納するリスト
final_eval_metrics = []

# K-Foldクロスバリデーションの実行
for fold in range(5):
    print(f"--- Fold {fold+1}/5 の学習を開始 ---")
    
    # 訓練データと検証データに分割
    train_df = df[df['kfold'] != fold].reset_index(drop=True)
    val_df = df[df['kfold'] == fold].reset_index(drop=True)
    
    # 特徴量と目的変数を抽出
    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]

    # LGBMモデルを呼び出す
    model = model_dispatcher.models["lgbm"]

    # モデルの訓練
    model.fit(X_train, y_train)

    # 検証データで予測
    val_preds = model.predict_proba(X_val)[:, 1]
    val_pred_labels = (val_preds >= 0.5).astype(int)

    # 評価指標を計算
    roc_auc = roc_auc_score(y_val, val_preds)
    accuracy = accuracy_score(y_val, val_pred_labels)
    # ... (precision, recall, f1_scoreの計算も追加) ...

    print(f"--- Fold {fold+1} の評価指標 ---")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"正解率:  {accuracy:.4f}")
    
    final_eval_metrics.append({
        'fold': fold + 1,
        'roc_auc': roc_auc,
        'accuracy': accuracy
        # ... (他の指標も追加) ...
    })

print("\n全てのフォールドの学習が完了しました。")
# 全フォールドの平均を計算
avg_roc_auc = np.mean([m['roc_auc'] for m in final_eval_metrics])
avg_accuracy = np.mean([m['accuracy'] for m in final_eval_metrics])
print(f"平均ROC AUC: {avg_roc_auc:.4f}")
print(f"平均正解率:  {avg_accuracy:.4f}")