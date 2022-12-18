import pandas as pd
import numpy as np
import os

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostClassifier

import optuna
from optuna import Trial
from optuna.samplers import TPESampler

import pickle

import warnings
warnings.filterwarnings(action='ignore')


# 초모수는 learning_rate, n_estimators, max_depth 활용
def objective(trial: Trial) -> float:
    params_cat = {
        "random_state": 39,
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 1),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 16)
    }

    model = CatBoostClassifier(**params_cat)
    model.fit(X_partrain, y_partrain, eval_set=[(X_val, y_val)],
              early_stopping_rounds=100, cat_features=categorical_features, verbose=False)

    cat_pred = model.predict(X_val)
    AUC = roc_auc_score(y_val, cat_pred)

    return AUC


if __name__ == "__main__":
    try_optuna = False
    train_data_path = './dataset/train.csv'

    # 변수 선택에 대한 EDA는 ipynotebook
    teacher_train_features = ['COMPONENT_ARBITRARY', 'ANONYMOUS_1', 'YEAR', 'ANONYMOUS_2', 'AG',
                              'CO', 'CR', 'CU', 'FE', 'H2O', 'MN', 'MO', 'NI', 'PQINDEX', 'TI', 'V',
                              'V40', 'ZN', 'Y_LABEL', 'AL', 'BA']

    Rdata_train = pd.read_csv(train_data_path)

    train2 = Rdata_train.loc[:, teacher_train_features]

    # 범주형 변수인 COMPONENT_ARBITRARY와 YEAR를 LabelEncoder 변환

    le1 = LabelEncoder()
    le2 = LabelEncoder()
    train2['COMPONENT_ARBITRARY_category'] = le1.fit_transform(train2['COMPONENT_ARBITRARY'])
    train2['YEAR_category'] = le2.fit_transform(train2['YEAR'])
    # 원래 범주형 변수는 제거해준다.
    train2 = train2.drop(['COMPONENT_ARBITRARY', 'YEAR'], axis=1)
    categorical_features = ['COMPONENT_ARBITRARY_category', 'YEAR_category']

    X_train = train2.drop(['Y_LABEL'], axis=1)
    y_train = train2['Y_LABEL']

    X_partrain, X_val, y_partrain, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=39,
                                                            stratify=y_train)
    print("[INFO] Completely Load data")
    print("\t - Shape of Train data: ", X_partrain.shape)
    print("\t - Shape of Val data: ", X_val.shape)
    print("\t - Train & Val Column Info: ", X_partrain.columns)

    if try_optuna:
        # Optuna 초모수 작업 시작
        print("[INFO] Finding optimal hyper-parameter using Optuna")
        sampler = TPESampler(seed=39)
        study = optuna.create_study(
            study_name="cat_parameter_opt",
            direction="maximize",
            sampler=sampler)
        study.optimize(objective, n_trials=100)

        # 가장 좋은 초모수와 성능 확인
        print("\t - Best Score :", study.best_value)
        print("\t - Best trial :", study.best_trial.params)

        opt_hyper_params = study.best_trial.params
    else:
        opt_hyper_params = {'learning_rate': 0.03142344166841527,
                            'n_estimators': 513,
                            'max_depth': 6}

    # 위의 초모수 적용하고, StratifiedKFold을 이용해 모델 적합
    n_fold = 5
    cv = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=39)

    cat_val = np.zeros((X_train.shape[0], 2))
    cat_partrain = np.zeros((X_partrain.shape[0], 2))

    for i, (i_trn, i_val) in enumerate(cv.split(X_train, y_train), 1):
        print(f'training model for CV #{i}')
        optuna_cat = CatBoostClassifier(
            random_state=39,
            learning_rate=opt_hyper_params["learning_rate"],
            n_estimators=opt_hyper_params["n_estimators"],
            max_depth=opt_hyper_params["max_depth"])

        optuna_cat.fit(X_train.loc[i_trn, :], y_train[i_trn], verbose=False, cat_features=categorical_features)

        # 학습이 완료된 Teacher 모델 5개를 저장
        path_teacher_model = os.path.join('models', 'teacher', 'Teacher_model' + str(i) + '.pickle')
        with open(path_teacher_model, 'wb') as fw:
            pickle.dump(optuna_cat, fw)
        print('[INFO] Successfully saved teacher model, path: ', path_teacher_model)

        cat_val[i_val, :] = optuna_cat.predict_proba(X_train.loc[i_val, :])
        cat_partrain += optuna_cat.predict_proba(X_partrain) / n_fold

