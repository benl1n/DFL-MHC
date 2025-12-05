import pandas as pd
import numpy as np
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

def main():

    data1 = pd.read_csv('features/MHC_ESM1b_1.csv')
    data2 = pd.read_csv('features/MHC_ESM2_1.csv')
    data3 = pd.read_csv('features/MHC_ESM1b_2.csv')
    data4 = pd.read_csv('features/MHC_ESM2_2.csv')

    labels = data1.iloc[:, 0].values
    f1 = data1.iloc[:, 1:].values
    f2 = data2.iloc[:, 1:].values
    f3 = data3.iloc[:, 1:].values
    f4 = data4.iloc[:, 1:].values

    X = np.concatenate([f1, f2, f3, f4], axis=1)

    X, labels = shuffle(X, labels, random_state=42)

    train_features, test_features, train_labels, test_labels = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)


    scoring = {
        'acc': make_scorer(accuracy_score),
        'sp': make_scorer(precision_score, average='macro'),
        'sn': make_scorer(recall_score, average='macro'),
        'mcc': make_scorer(matthews_corrcoef),
        'auc': make_scorer(roc_auc_score, needs_proba=True)
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    best_cv_acc = 0
    best_feature_dim = 0
    best_pca = None
    train_acc_records = []

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
        sp_cv  = round(results['test_sp'].mean(), 4)
        sn_cv  = round(results['test_sn'].mean(), 4)
        mcc_cv = round(results['test_mcc'].mean(), 4)
        auc_cv = round(results['test_auc'].mean(), 4)

        print(f"CV: acc={acc_cv}, sp={sp_cv}, sn={sn_cv}, mcc={mcc_cv}, auc={auc_cv}")

        train_acc_records.append((i, acc_cv))

        if acc_cv > best_cv_acc:
            best_cv_acc = acc_cv
            best_feature_dim = i
            best_pca = pca

    best_train_pca = best_pca.transform(train_features)
    best_test_pca = best_pca.transform(test_features)

    np.save("best_train_features.npy", best_train_pca)
    np.save("best_test_features.npy", best_test_pca)

    print(f"Best Feature Dim (from CV): {best_feature_dim}")
    print(f"Best CV ACC: {best_cv_acc}")

    final_model = MLPClassifier(random_state=42, max_iter=500)
    final_model.fit(best_train_pca, train_labels)

    final_predictions = final_model.predict(best_test_pca)

    final_acc = round(accuracy_score(test_labels, final_predictions), 4)
    final_sp  = round(precision_score(test_labels, final_predictions, average='macro'), 4)
    final_sn  = round(recall_score(test_labels, final_predictions, average='macro'), 4)
    final_mcc = round(matthews_corrcoef(test_labels, final_predictions), 4)

    print("\n================ Final Test Result ================")
    print(f"Final Test ACC: {final_acc}")
    print(f"Final Test SP : {final_sp}")
    print(f"Final Test SN : {final_sn}")
    print(f"Final Test MCC: {final_mcc}")

    final_model_bundle = {
        'scaler': scaler,
        'pca': best_pca,
        'model': final_model
    }

    with open("final_mhc_model.pkl", "wb") as f:
        pickle.dump(final_model_bundle, f)


if __name__ == '__main__':
    main()