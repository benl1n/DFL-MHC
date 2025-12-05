# DFL-MHC

## 1. Introduction

DFL-MHC is a dual-stage learning framework for MHC protein identification that integrates multi-view feature fusion from multiple protein language models with BiLSTM-based deep sequence modeling.

![Model Architecture](https://github.com/benl1n/DFL-MHC/blob/main/DFL-MHC.png)

## 2. Python Environment







## 3.Prediction

1. search the best feature 

```python
python first_stage.py
```

2. get the best feature

```
python process_features.py
```

3. get the result of MHC prediction

```
python train_test.py
```













