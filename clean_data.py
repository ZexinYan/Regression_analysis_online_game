import pandas as pd
import numpy as np
from helper import Helper
from background_analysis import Trainer
from background_analysis import Text_Analysis
import re
import warnings
warnings.filterwarnings("ignore")

raw_data = pd.read_csv('./data/online_game.csv')
helper = Helper()


# 将`背景`文本量化
def process_background():
    quantills = 2
    per_quantile = 1.0 / quantills
    background_set = []
    comment = raw_data[u'评论']
    for i in range(quantills):
        up = comment.quantile((i + 1) * per_quantile)
        down = comment.quantile(i * per_quantile)
        indexs = raw_data[(up >= raw_data[u'评论']) & (raw_data[u'评论'] >= down)].index
        local_list = [raw_data.iloc[each][u'背景'] for each in indexs.values]
        background_set.append(local_list)
    Text_Analysis.Text_Analysis().process_data(background_set, 20)
    classifier = Trainer.Category_Classifier()
    classifier.train()
    raw_data[u'背景'] = list(map(classifier.predict, raw_data[u'背景']))


# 分割开始时间
def process_time():
    def get_date(text):
        groups = re.search(r'游戏于(.*)年(.*)月入库', str(text))
        if groups:
            return str(groups.group(1)) + str(groups.group(2))
        else:
            return 0

    def get_times(text):
        groups = re.search(r'共(.*)次测试', str(text))
        if groups:
            return str(groups.group(1))
        else:
            return 0
    raw_data[u'入库时间'] = list(map(get_date, raw_data[u'开发时间']))
    raw_data[u'测试次数'] = list(map(get_times, raw_data[u'开发时间']))
    raw_data.drop(u'开发时间', axis=1, inplace=True)


# 处理分类变量
def process_dummy(col):
    dummy_type = pd.get_dummies(raw_data[col], prefix=col)
    new_raw_data = pd.DataFrame(pd.concat([raw_data, dummy_type], axis=1))
    new_raw_data.drop([col], axis=1, inplace=True)
    return new_raw_data


def fill_zero(value):
    return value + 1


def closure(dict):
    def fill_dict(value):
        if value not in dict:
            return 0
        else:
            return dict[value]
    return fill_dict

developer = helper.construct_dict(raw_data, u'开发商', [0.15, 0.10, 0.05, 0])
raw_data[u'开发商'] = list(map(closure(developer), raw_data[u'开发商']))

Operator = helper.construct_dict(raw_data, u'运营商', [0.20, 0.15, 0.10, 0.05, 0])
raw_data[u'运营商'] = list(map(closure(Operator), raw_data[u'运营商']))

comment = list(map(fill_zero, raw_data[u'评论']))
y = pd.DataFrame(np.log(comment))

process_background()
process_time()
raw_data = process_dummy(u'类型')
raw_data = process_dummy(u'收费')
raw_data = process_dummy(u'模式')
raw_data = process_dummy(u'题材')
raw_data = process_dummy(u'画风')
raw_data = process_dummy(u'画面')
raw_data.drop([u'网游名称', u'评论'], axis=1, inplace=True)

raw_data.to_csv('./data/clean_data.csv', encoding='utf-8', index_label=False)
y.to_csv('./data/result.csv', encoding='utf-8', index_label=False)
