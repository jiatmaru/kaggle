# predict.py

import pandas as pd
import os
import importlib.util
import sys
import joblib # モデルの保存・読み込み用

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
from preprocessor import CustomPreprocessor

def run_prediction():
    # Step 1: データの読み込み
    train_df = pd.read_csv(os.path.join(PROJECT_ROOT, config.training_file))
    test_df = pd.read_csv(os.path.join(PROJECT_ROOT, config.training_file.replace('train.csv', 'test.csv')))
    
    # id列と目的変数yを分離
    id_col = 'id'
    target = 'y'
    
    # 最終モデルを訓練するためのデータ準備
    X_train_full = train_df.drop(columns=[id_col, target])
    y_train_full = train_df[target]
    
    # テストデータの特徴量を準備
    X_test = test_df.drop(columns=[id_col])

    # Step 2: 前処理
    # preprocessorのインスタンスを作成し、trainデータ全体でfit_transform
    preprocessor = CustomPreprocessor()
    X_train_full_processed = preprocessor.fit_transform(X_train_full)

    # testデータに対してはtransformのみ
    X_test_processed = preprocessor.transform(X_test)
    
    # 列名を統一
    X_train_full_processed.rename(columns={col: col.replace('.', '').replace('_', '').replace(' ', '') for col in X_train_full_processed.columns}, inplace=True)
    X_test_processed.rename(columns={col: col.replace('.', '').replace('_', '').replace(' ', '') for col in X_test_processed.columns}, inplace=True)
    
    # Step 3: 最終モデルの訓練
    print("全訓練データで最終モデルの学習を開始...")
    model = model_dispatcher.models["lgbm"]
    model.fit(X_train_full_processed, y_train_full)
    
    # Step 4: 予測の生成
    test_preds = model.predict_proba(X_test_processed)[:, 1]
    
    # Step 5: 結果の保存
    submission_df = pd.DataFrame({
        id_col: test_df[id_col],
        'y': test_preds
    })
    submission_df.to_csv(os.path.join(PROJECT_ROOT, config.model_output, 'submission.csv'), index=False)
    print(f"予測結果を {os.path.join(PROJECT_ROOT, config.model_output, 'submission.csv')} に保存しました。")

# モデルを保存
output_model_path = os.path.join(PROJECT_ROOT, config.model_output, 'model.joblib')
joblib.dump(model, output_model_path)
print(f"モデルを {output_model_path} に保存しました。")

    
if __name__ == '__main__':
    run_prediction()