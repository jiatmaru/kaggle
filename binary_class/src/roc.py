# src/roc.py

import numpy as np

def calculate_roc_metrics(y_true, y_pred, thresholds=None):
    """
    閾値ごとの真陽性率（TPR）と偽陽性率（FPR）を計算します。

    Args:
        y_true (np.array): 正解ラベルの配列（0または1）。
        y_pred (np.array): 予測確率の配列（0.0から1.0）。
        thresholds (np.array, optional): 評価する閾値の配列。
                                         Noneの場合、0.0から1.0まで0.01刻みで生成。

    Returns:
        dict: 閾値、TPR、FPRのリストを含む辞書。
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    tpr_list = []
    fpr_list = []
    
    # y_trueとy_predをnumpy配列に変換
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for thresh in thresholds:
        # 閾値に基づいて予測をバイナリ（0か1）に変換
        temp_pred = (y_pred >= thresh).astype(int)

        # 真陽性、偽陽性、偽陰性、真陰性の数を計算
        tp = np.sum((temp_pred == 1) & (y_true == 1))
        fp = np.sum((temp_pred == 1) & (y_true == 0))
        fn = np.sum((temp_pred == 0) & (y_true == 1))
        tn = np.sum((temp_pred == 0) & (y_true == 0))

        # 真陽性率（TPR）と偽陽性率（FPR）を計算
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return {
        'thresholds': thresholds.tolist(),
        'tpr': tpr_list,
        'fpr': fpr_list
    }

# このファイルが直接実行された場合のテスト用コード
if __name__ == "__main__":
    y_true_test = [0, 1, 0, 1, 1, 0]
    y_pred_test = [0.1, 0.8, 0.3, 0.6, 0.9, 0.2]
    
    metrics = calculate_roc_metrics(y_true_test, y_pred_test)
    
    print("ROC指標の計算結果:")
    for i in range(len(metrics['thresholds'])):
        print(f"閾値: {metrics['thresholds'][i]:.2f}, TPR: {metrics['tpr'][i]:.4f}, FPR: {metrics['fpr'][i]:.4f}")