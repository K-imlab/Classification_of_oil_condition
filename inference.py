import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import LabelEncoder

import pickle

import warnings
warnings.filterwarnings(action = 'ignore')

train_data_path = './dataset/train.csv'
test_data_path = './dataset/test.csv'

teacher_train_features = ['COMPONENT_ARBITRARY', 'ANONYMOUS_1', 'YEAR', 'ANONYMOUS_2', 'AG',
                          'CO', 'CR', 'CU', 'FE', 'H2O', 'MN', 'MO', 'NI', 'PQINDEX', 'TI', 'V',
                          'V40', 'ZN', 'Y_LABEL', 'AL', 'BA']

Rdata_train = pd.read_csv(train_data_path)
Rdata_test = pd.read_csv(test_data_path)

train2 = Rdata_train.loc[:, teacher_train_features]
test2 = Rdata_test.drop(['ID'], axis=1)

# 범주형 변수인 COMPONENT_ARBITRARY와 YEAR를 LabelEncoder 변환

le1 = LabelEncoder()
le2 = LabelEncoder()
train2['COMPONENT_ARBITRARY_category'] = le1.fit_transform(train2['COMPONENT_ARBITRARY'])
train2['YEAR_category'] = le2.fit_transform(train2['YEAR'])
test2['COMPONENT_ARBITRARY_category'] = le1.transform(test2['COMPONENT_ARBITRARY'])
test2['YEAR_category'] = le2.transform(test2['YEAR'])

# 원래 범주형 변수는 제거해준다.
test2 = test2.drop(['COMPONENT_ARBITRARY', 'YEAR'], axis=1)
X_test = test2

print("[INFO] Completely Load data")
print("\t - Shape of Test data: ", test2.shape)
print("\t - Test Column Info: ", test2.columns)

student_saved_paths = os.path.join('models', 'student', '*.pickle')
models = []
for model_path in glob.glob(student_saved_paths):
    with open(model_path, 'rb') as f:
        models.append(pickle.load(f))

print("[INFO] Completely Loaded Student Model")

n_fold = 5
cat_test = np.zeros((X_test.shape[0]))
for idx, model in enumerate(models):
    cat_test += model.predict(X_test) / n_fold

print("[INFO] Inference is Completed")

# answer 만들기
answer = np.zeros(cat_test.shape[0])

for i in range(cat_test.shape[0]):
    if cat_test[i] >= 0.15:
        answer[i] = 1

answer = answer.astype('int64')
submission_preds = answer

sample_submission = os.path.join('dataset', 'sample_submission.csv')
submission = pd.read_csv(sample_submission)
submission['Y_LABEL'] = submission_preds

result_submission = os.path.join('result', 'submission_CatBoost.csv')
submission.to_csv(result_submission, index=False)
print("[INFO] Completely Saved result, path: ", result_submission)

