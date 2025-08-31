import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os

def create_folds(n_splits=5):
    """
    データセットをK-Foldに分割し、分割情報をCSVファイルとして保存します。
    Args:
        n_splits (int): 分割するフォールドの数。デフォルトは5。
    """
    # プロジェクトのルートパスを定義
    # このパスはあなたのプロジェクト構造に合わせてください
    PROJECT_ROOT = "/content/drive/MyDrive/kaggle/binary_class"
    # ファイルの絶対パスを構築
    input_file_path = os.path.join(PROJECT_ROOT, "input/train.csv")
    output_file_path = os.path.join(PROJECT_ROOT, "input/train_folds.csv")
    
    try:
        df = pd.read_csv(input_file_path)
    except FileNotFoundError:
        print(f"Error: train.csv not found at {input_file_path}")
        return

    # k-fold分割のために新しい'kfold'列を追加
    df['kfold'] = -1
    # 目的変数 'y' を確認
    # ここでは 'y' という列が存在することを前提とします
    # もし目的変数の列名が異なる場合は適宜変更してください
    if 'y' not in df.columns:
        print("Error: 'y' column not found in train.csv.")
        return

    # StratifiedKFoldで層化抽出分割
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # enumerateでインデックスと分割情報を取得
    for fold, (train_idx, val_idx) in enumerate(skf.split(X=df, y=df['y'])):
        df.loc[val_idx, 'kfold'] = fold

    # /input ディレクトリに新しいCSVを保存
    output_path = "/content/drive/MyDrive/kaggle/binary_class/input/train_folds.csv" # <-- 修正後
    df.to_csv(output_path, index=False)
    print(f"Successfully created {output_path}")

if __name__ == '__main__':
    create_folds(n_splits=5)