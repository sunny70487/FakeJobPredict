#!/usr/bin/env python
# coding: utf-8

import pandas as pd 
import lightgbm as lgb
from Data_Base import dataHandle

def main(random_num):
    #讀取資料
    test_data = pd.read_csv('test.csv')
    X_test = test_data.drop(test_data.columns[[0]],axis=1)
    y_test = pd.read_csv("answer.csv")
    result = use_model(X_test, y_test, random_num)
    #讀入資料庫
    dataHandle(result)
    return result

def use_model(X_test, y_test, random_num):
    #讀取訓練好的模型
    load_data = lgb.Booster(model_file = 'lightgbm_model.txt')
    pred = load_data.predict(X_test.iloc[random_num])
    if pred[0] >0.5:
        pred[0] = 1
    else:
        pred[0] = 0
    answer = y_test.iloc[random_num][1]
    if int(pred[0]) == int(answer):
        pred_result = 1
    else:
        pred_result = 0
    result_array = [pred, answer, random_num, pred_result]
    return result_array





