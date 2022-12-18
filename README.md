# 건설기계 오일 상태 분류 AI 경진대회
건설장비에서 작동오일의 상태를 실시간으로 모니터링하기 위한 오일 상태 판단 모델 개발 (정상, 이상의 이진분류)
<br>[Competition Link](https://dacon.io/competitions/official/236013/overview/description)
* 주최: 현대제뉴인 [[Link]](https://www.hyundai-genuine.com/?locale=ko)
* 후원: AWS
* 주관: Dacon
* **Private 12th, Score 0.577764**
***

## Structure
Train/Test data and sample submission file must be placed under **dataset** folder.
```
repo.
  |——dataset
        |——train.csv
        |——test.csv
        |——sample_submission.csv
        |——data_info.xlsx
  |——models
        |——teacher
            |——Teacher_model1.pickle
            |—— ...
            |——Teacher_model5.pickle
        |——student
            |——Student_model1.pickle
            |—— ...
            |——Student_model5.pickle
  |——result
        |——submission_CatBoost.csv
  |——ipynotebook
        |——학습코드.ipynb
  |——requirements.txt
  |——teacher_train.py
  |——student_train.py
  |——inference.py
```
***
## Development Environment
### Our Training Resource
```
Windows 10
11th Gen Intel(R) Core(TM) i7-11850H
NVIDIA RTX A2000 Laptop GPU
```

### Or you can execute in Web based Development Services
If you want to run it with Colab or AWS Sagemaker, run the ipynb code below. Also, It includes an EDA and results analysis process. It will output the same result.
* Google Colab
* AWS Sagemaker Studio Lab
```
repo.
  |——ipynotebook
        |——학습코드.ipynb
```
***
## Training Code Run Solution
### 0. Dependency
the details of packages version are listed in **requirements.txt**

```shell
> python --version
Python 3.9.15
> pip install -r requirements.txt
```
```shell
> pip list
...
numpy         1.23.5
pandas        1.5.2
scikit-learn  1.2.0
catboost      1.1.1
optuna        3.0.4
...
```
Because the number of variables in the learning environment and the inference environment
are different, we used the knowledge distillation method.
So, We go through 2 trainning process to obtain **Teacher model and Student Model**
### 1. Teacher Model Training
* learning method: [CatBoostClassifier](https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier)
* 5 StratifiedKFold ensemble
  * So, 5 model weights are saved after training
* includes hyper-parameter tuning preocess with Optuna
 
```shell
> python teacher_train.py
```

### Result

```
models
      |——teacher
            |——Teacher_model1.pickle
            |—— ...
            |——Teacher_model5.pickle
```
***
### 2. Student Model Training
* train with the teacher's output as the label
* learning method: [CatBoostRegressor](https://catboost.ai/en/docs/concepts/python-reference_catboostregressor)
* 5 StratifiedKFold ensemble 
  * So, 5 model weights are saved after training
* includes hyper-parameter tuning preocess with Optuna
 
```shell
> python student_train.py
```

### Result

```
models
      |——student
            |——Student_model1.pickle
            |—— ...
            |——Student_model5.pickle
```
***
### 3. Inference
* 5 fold ensemble (soft-voting) inference
* metric: F1 score
* thershold: 0.15
```shell
python inference.py
```
### Result
check **submission_CatBoost.csv**
```
result
      |——submission_CatBoost.csv
```
***
## Download Best Model
You can download our best model weight [Here GoogleDrive](https://drive.google.com/file/d/1A0aJ9Al_ZiSdLVuRMNippPkuegFyMj1i/view).
***
## Inference With Best Model
Best model must be unzipped under **models** folder

### 1. Unzip Best Model
unzip and move best model to **models** directory
### 2. Inference
```shell
python inferecne.py
```

### 3. Result
check **submission_CatBoost.csv**
```
models  # (it is best model)
    |——teacher
          |——Teacher_model1.pickle
          |—— ...
          |——Teacher_model5.pickle
    |——student
          |——Student_model1.pickle
          |—— ...
          |——Student_model5.pickle
result  # (it is best result)
    |——submission_CatBoost.csv
inference.py
```