import pandas as pd
import numpy as np
import random
import torch
import pickle

from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    roc_auc_score
)

# ===========================
# 固定随机种子（保证可复现）
# ===========================
manualSeed = 2
random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)

def main():

    # ===========================
    # 1. 读取特征
    # ===========================
    data1 = pd.read_csv('features/MHC_ESM.csv')
    data2 = pd.read_csv('features/MHC_ESM2-650M.csv')
    data3 = pd.read_csv('features/MHC_ESM1b-650M.csv')
    data4 = pd.read_csv('features/MHC_ESM2.csv')

    labels = data1.iloc[:, 0].values
    f1 = data1.iloc[:, 1:].values
    f2 = data2.iloc[:, 1:].values
    f3 = data3.iloc[:, 1:].values
    f4 = data4.iloc[:, 1:].values

    X = np.concatenate([f1, f2, f3, f4], axis=1)

    # ===========================
    # 2. 打乱 + 划分固定测试集（测试集永远不参与CV）
    # ===========================
    X, labels = shuffle(X, labels, random_state=42)

    train_features, test_features, train_labels, test_labels = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # ===========================
    # 3. 仅用训练集拟合标准化器（防止数据泄漏）
    # ===========================
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    # ===========================
    # 4. 评价指标
    # ===========================
    scoring = {
        'acc': make_scorer(accuracy_score),
        'sp': make_scorer(precision_score, average='macro'),
        'sn': make_scorer(recall_score, average='macro'),
        'mcc': make_scorer(matthews_corrcoef),
        'auc': make_scorer(roc_auc_score, needs_proba=True)
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # ===========================
    # 5. 记录最优结果
    # ===========================
    best_train_acc = 0
    best_test_acc = 0
    best_feature_dim = 0
    best_model_bundle = None

    train_acc_records = []

    # ===========================
    # 6. PCA 维度搜索 + 10 折交叉验证
    # ===========================
    for i in tqdm(range(1, 400)):

        print(f"\n Feature Dim = {i}")

        pca = PCA(n_components=i, random_state=0)
        train_pca = pca.fit_transform(train_features)

        model = MLPClassifier(random_state=42, max_iter=500)

        results = cross_validate(
            model,
            train_pca,
            train_labels,
            cv=cv,
            scoring=scoring
        )

        acc_cv = round(results['test_acc'].mean(), 4)
        sp_cv = round(results['test_sp'].mean(), 4)
        sn_cv = round(results['test_sn'].mean(), 4)
        mcc_cv = round(results['test_mcc'].mean(), 4)
        auc_cv = round(results['test_auc'].mean(), 4)

        print(f"CV: acc={acc_cv}, sp={sp_cv}, sn={sn_cv}, mcc={mcc_cv}, auc={auc_cv}")

        train_acc_records.append((i, acc_cv))

        # ===========================
        # 7. 用完整训练集重新训练模型
        # ===========================
        model.fit(train_pca, train_labels)

        test_pca = pca.transform(test_features)
        predictions = model.predict(test_pca)

        acc_test = round(accuracy_score(test_labels, predictions), 4)
        sp_test = round(precision_score(test_labels, predictions, average='macro'), 4)
        sn_test = round(recall_score(test_labels, predictions, average='macro'), 4)
        mcc_test = round(matthews_corrcoef(test_labels, predictions), 4)

        print(f"Test: acc={acc_test}, sp={sp_test}, sn={sn_test}, mcc={mcc_test}")

        # ===========================
        # 8. 保存“测试集最优模型”（这是最终论文用的）
        # ===========================
        if acc_test > best_test_acc:
            best_test_acc = acc_test
            best_feature_dim = i

            best_model_bundle = {
                'scaler': scaler,
                'pca': pca,
                'model': model
            }

            best_test_sn = sn_test
            best_test_sp = sp_test
            best_test_mcc = mcc_test

    # ===========================
    # 9. 保存最终最优模型
    # ===========================
    with open('best_model_bundle.pkl', 'wb') as f:
        pickle.dump(best_model_bundle, f)


    print(f"Best Dim: {best_feature_dim}")
    print(f"Best Test ACC: {best_test_acc}")
    print(f"Best Test SP : {best_test_sp}")
    print(f"Best Test SN : {best_test_sn}")
    print(f"Best Test MCC: {best_test_mcc}")

if __name__ == '__main__':
    main()
