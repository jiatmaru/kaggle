import os

# train.pyの絶対パス
train_file_path = "/content/drive/MyDrive/kaggle/binary_class/src/train.py"

# --- ステップ1: ファイルの内容を確認 ---
print("--- 実行環境が読み込んでいる train.py の内容 ---")
try:
    with open(train_file_path, 'r', encoding='utf-8') as f:
        # ファイルの最初の数行を読み込み、表示
        for i in range(10):
            line = f.readline()
            if not line:
                break
            print(line.strip())
except FileNotFoundError:
    print(f"エラー: ファイル '{train_file_path}' が見つかりません。パスを確認してください。")

# --- ステップ2: ファイルの内容を文字列として読み込み、強制的に実行 ---
print("\n--- スクリプトの実行（最終手段） ---")
try:
    with open(train_file_path, 'r', encoding='utf-8') as f:
        # ファイル全体の内容を文字列として読み込む
        script_content = f.read()
    
    # exec()を使ってスクリプトの内容を実行
    exec(script_content)
    print("\nスクリプトが正常に実行されました。")

except Exception as e:
    print(f"\nスクリプトの実行中にエラーが発生しました: {e}")