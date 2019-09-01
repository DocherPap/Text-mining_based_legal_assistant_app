import jieba
import logging
import sys
import codecs
import traceback
import pandas as pd
import numpy as np
import re
from collections import Counter
jieba.load_userdict('/Users/olivia/Desktop/text-classification/自立自强独立更生/台灣法律用語.txt')


if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False

def delCNf(line):
    regex = re.compile('[^\u4e00-\u9fa5a-zA-Z\s]')
    return regex.sub('', line)


def str_replace(str_source, char, *words):
    str_temp = str_source
    for word in words:
        str_temp = str_temp.replace(word, char)
    return str_temp


def seg_words(sentence):
    stopwords = {}.fromkeys([ line.rstrip() for line in open('/Users/olivia/Desktop/text-classification/自立自强独立更生/stopword.txt') ])
    segs = jieba.cut(sentence, cut_all=False)  # 默认是精确模式
    return " ".join(segs)     # 分词，然后将结果列表形式转换为字符串


def segmentation(file):
    files = re.split(",", file)
    lst = []
    for line in files:
        line = delCNf(line)
        line = str_replace(line, "", "\t", "\n", " ", "\u3000", "○")
        seg_list = jieba.cut(line, cut_all = False)
        words = " ".join(seg_list)
        words = "".join(words.split('\n')) # 去除回车符
        words = delCNf(words)
        word = re.split(r" ", words)
        lst.extend(word)
    return lst


from sklearn.linear_model import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn import *
from random import randint
from sklearn.externals import joblib
import math
import pickle
import lightgbm as lgb 
import os

def made_matrix(string):
    matrix = pd.read_csv("/Users/olivia/Desktop/text-classification/自立自强独立更生/matrix_column.csv")
    matrix.loc[0] = randint(0,0)

    for col in matrix.columns:
        count = 0
        for word in segmentation(string):
            if str(word) == str(col):
                count += 1
        matrix.loc[0, col] = count

    predict_x = matrix.dropna().astype('int')
    
    return predict_x

def predict_vic(predict_x):
    reload = lgb.Booster(model_file='/Users/olivia/Desktop/text-classification/自立自强独立更生/lightgbm_model.txt')
    
    y_pred = reload.predict(predict_x)
    y_pred = [list(x).index(max(x)) for x in y_pred]
    
    return y_pred # 0 勝，1 平， 2 敗訴


def knn_five_judgement(predict_x):
    df_dataframe = pd.read_csv("/Users/olivia/Desktop/text-classification/自立自强独立更生/matrix1500_new.csv").reset_index().dropna()
    
    reload_knn = joblib.load('/Users/olivia/Desktop/text-classification/自立自强独立更生/knn_model.model')
    neighborpoint = reload_knn.kneighbors(predict_x, 5, False)
    neighborlst = []
    for idx in neighborpoint[0]:
        neighborlst.append(df_dataframe.loc[idx,'filename'])
        
    return neighborlst


