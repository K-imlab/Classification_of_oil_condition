import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
import pickle

import warnings
warnings.filterwarnings(action = 'ignore')

train_data_path = './dataset/train.csv'

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


teacher_saved_paths = os.path.join('models', 'teacher', '*.pickle')
models = []
for model_path in glob.glob(teacher_saved_paths):
    with open(model_path, 'rb') as f:
        models.append(pickle.load(f))

print("[INFO] Completely Loaded Teacher Model")

n_fold = 5
cv = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=39)
cat_val = np.zeros((X_train.shape[0], 2))
cat_partrain = np.zeros((X_partrain.shape[0], 2))
for i, (i_trn, i_val) in enumerate(cv.split(X_train, y_train), 1):
    cat_val[i_val, :] = models[i-1].predict_proba(X_train.loc[i_val, :])
    cat_partrain += models[i-1].predict_proba(X_partrain) / n_fold

# Teacher model의 예측 결과를 Student의 Label로 지정
y_train2 = cat_val[:, 1]
X_train2 = train2.drop(['AL', 'BA', 'Y_LABEL'], axis=1)

print("[INFO] Completely Got Teacher's Prediction result ")

X_partrain, X_val, y_partrain, y_val = train_test_split(X_train2, y_train2, test_size = 0.3, random_state = 39)

opt_hyper_parameter = {'learning_rate': 0.01310047432090872,
                       'n_estimators': 848,
                       'max_depth': 9}

# Optuna로 구한 초모수를 적용하고, KFold을 이용해 모델 적합

n_fold = 5
cv = KFold(n_splits=n_fold, shuffle=True, random_state=39)

cat_val = np.zeros((X_train2.shape[0]))

print(cat_val.shape)

for i, (i_trn, i_val) in enumerate(cv.split(X_train2, y_train2), 1):
    print(f'training student model for CV #{i}')
    optuna_cat = CatBoostRegressor(
        random_state=39,
        learning_rate=opt_hyper_parameter['learning_rate'],
        n_estimators=opt_hyper_parameter['n_estimators'],
        max_depth=opt_hyper_parameter['max_depth'])

    optuna_cat.fit(X_train2.loc[i_trn, :], y_train2[i_trn], verbose=False, cat_features=categorical_features)

    # 학습이 완료된 Student 모델 5개를 저장
    path_student_model = os.path.join('models', 'student', 'Student_model' + str(i) + '.pickle')
    with open(path_student_model, 'wb') as fw:
        pickle.dump(optuna_cat, fw)
    print('[INFO] Successfully saved student model, path: ', path_student_model)

    cat_val[i_val] = optuna_cat.predict(X_train2.loc[i_val, :])
