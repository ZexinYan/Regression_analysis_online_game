import pandas as pd
import numpy as np
import json
import math
import os
import codecs
import jieba
import nltk
import re


class Text_Analysis:
    def __init__(self):
        self.features = []

    def process_data(self, corpus, feature_num=10):
        stopwords = self.loan_stopwords()
        for each in corpus:
            self.clean_text(each)
        # process the data
        seg_list = []
        for each in corpus:
            seg_list.append(self.get_seg_list(each, stopwords))
        freq_dist = []
        for each in seg_list:
            freq_dist.append(self.get_freq_dist(each))
        words_set = []
        for each in seg_list:
            words_set.append(self.get_words_set(each))
        total_seg_list = []
        for seg in seg_list:
            for each in seg:
                total_seg_list.append(each)
        words_PMI = []
        for each in range(len(words_set)):
            words_PMI.append(self.sort_by_value(self.PMI(words_set[each],
                                                         seg_list[each], len(seg_list[each]) / len(total_seg_list))))
        self.features = self.feature_list(words_PMI, feature_num)
        self.save_feature_model(self.features)
        self.create_train_csv(self.features, freq_dist)

    def loan_stopwords(self):
        stopwords = []
        with codecs.open('./background_analysis/corpus/stopwords.txt', 'r', 'utf-8') as f:
            for each in f.readlines():
                each = each.strip('\n')
                each = each.strip('\r')
                stopwords.append(each)
        return stopwords

    def clean_text(self, sentenses):
        for i in range(len(sentenses)):
            sentenses[i] = re.sub(r'<div>|</div>|\n|\r|&nbsp;', '', str(sentenses[i])).replace(' ', '')

    def get_seg_list(self, array, stopwords):
        seg_list = []
        for each in array:
            local_list = jieba.cut(each, False)
            final_list = []
            for word in local_list:
                if word not in stopwords and word != ' ':
                    final_list.append(word)
            seg_list.append(final_list)
        return seg_list

    def get_freq_dist(self, seg_list):
        freq_dist = []
        for each in seg_list:
            freq_dist.append(nltk.FreqDist(each))
        return freq_dist

    def get_words_set(self, seg_list):
        word_set = set()
        for each in seg_list:
            for word in each:
                word_set.add(word)
        return word_set

    def PMI(self, words_set, seg_list, prob):
        PMI_words = {}
        for each in words_set:
            occur = 0
            for sent in seg_list:
                if each in sent:
                    occur += 1
            word_prob = occur / len(seg_list)
            PMI_words[each] = math.log(word_prob / prob)
        return PMI_words

    def sort_by_value(self, d):
        items=d.items()
        backitems=[[v[1], v[0]] for v in items]
        backitems.sort(reverse=True)
        return [backitems[i] for i in range(0, len(backitems))]

    def feature_list(self, factors, features_num):
        features = set()
        for factor in factors:
            for each in factor[:features_num]:
                features.add(each[1])
        print(features)
        return features

    def save_feature_model(self, features, filename='./background_analysis/model/feature.json'):
        with codecs.open(filename, 'w', 'utf-8') as f:
            f.write(json.dumps(list(features), ensure_ascii=False))
            f.close()

    def load_feature_model(self, filename='./background_analysis/model/feature.json'):
        with codecs.open(filename, 'r', 'utf-8') as f:
            self.features = json.loads(f.read())

    def create_train_csv(self, features, freq_dist):
        with codecs.open('./background_analysis/corpus/train.csv', 'w', 'utf-8') as f:
            f.write(','.join(features) + ',sentiment\n')
            for i in range(len(freq_dist)):
                words_vec = self.words2vec(features, freq_dist[i])
                for each in words_vec:
                    f.write(','.join(each) + ',' + str(int(i - len(freq_dist) / 2)) + '\n')
            f.close()

    def words2vec(self, features, freq_dist):
        word_vec = []
        for sentence in freq_dist:
            local_vec = []
            sum = 0
            for each in sentence:
                sum += sentence[each]
            sum += 1
            for each in features:
                local_vec.append(str(sentence[each] / sum))
            word_vec.append(local_vec)
        return word_vec

    def sentence2vec(self, sentence):
        if len(self.features) == 0:
            self.load_feature_model()
        seg_list = jieba.cut(sentence, False)
        freq_dist = nltk.FreqDist(seg_list)
        local_list = []
        for each in self.features:
            local_list.append(freq_dist[each])
        return local_list
