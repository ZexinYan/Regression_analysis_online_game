#coding:utf-8
import codecs
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import column_or_1d
from sklearn.utils import shuffle
from sklearn.multiclass import OneVsRestClassifier
from background_analysis.Text_Analysis import Text_Analysis
sns.set_style("whitegrid")


class Category_Classifier:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.model = None
        self.load_model()

    def load_data(self, filename='./background_analysis/corpus/train.csv'):
        if os.path.exists(filename):
            data = pd.read_csv(filename)
            self.data = shuffle(data)
            X_data = pd.DataFrame(data.drop('sentiment', axis=1))
            Y_data = column_or_1d(data[:]['sentiment'], warn=True)
            self.X_train, self.X_val,\
            self.y_train, self.y_val = train_test_split(X_data, Y_data, test_size=0.3, random_state=1)
        else:
            print('No Source!')

    def load_model(self, filename='./background_analysis/model/model.pickle'):
        if os.path.exists(filename):
            with codecs.open(filename, 'rb') as f:
                f = open(filename, 'rb')
                self.model = pickle.load(f)
        else:
            self.train()

    def save_model(self, filename='./background_analysis/model/model.pickle'):
        with codecs.open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def train(self):
        self.load_data()
        self.model = LinearSVC(random_state=7)
        self.model.fit(self.X_train, self.y_train)
        self.save_model()
        print('Accuracy: ' + str(round(self.model.score(self.X_val, self.y_val), 2)))

    def predict(self, sentence):
        vec = Text_Analysis().sentence2vec(sentence)
        return int(self.model.predict(vec))

    def show_heat_map(self):
        pd.set_option('precision', 2)
        plt.figure(figsize=(20, 6))
        sns.heatmap(self.data.corr(), square=True)
        plt.xticks(rotation=90)
        plt.yticks(rotation=360)
        plt.suptitle("Correlation Heatmap")
        plt.show()

    def show_heat_map_to(self, target='sentiment'):
        correlations = self.data.corr()[target].sort_values(ascending=False)
        plt.figure(figsize=(40, 6))
        correlations.drop(target).plot.bar()
        pd.set_option('precision', 2)
        plt.xticks(rotation=90, fontsize=7)
        plt.yticks(rotation=360)
        plt.suptitle('The Heatmap of Correlation With ' + target)
        plt.show()

    def plot_learning_curve(self):
        # Plot the learning curve
        plt.figure(figsize=(9, 6))
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X=self.X_train, y=self.y_train,
            cv=3, scoring='neg_mean_squared_error')
        self.plot_learning_curve_helper(train_sizes, train_scores, test_scores, 'Learning Curve')
        plt.show()

    def plot_learning_curve_helper(self, train_sizes, train_scores, test_scores, title, alpha=0.1):
        train_scores = -train_scores
        test_scores = -test_scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        plt.plot(train_sizes, train_mean, label='train score', color='blue', marker='o')
        plt.fill_between(train_sizes, train_mean + train_std,
                         train_mean - train_std, color='blue', alpha=alpha)
        plt.plot(train_sizes, test_mean, label='test score', color='red', marker='o')
        plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)
        plt.title(title)
        plt.xlabel('Number of training points')
        plt.ylabel(r'Mean Squared Error')
        plt.grid(ls='--')
        plt.legend(loc='best')
        plt.show()

    def choose_best_model(self):
        seed = 7
        pipelines = []
        pipelines.append(
            ('SVC',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ("SVC", OneVsRestClassifier(SVC(random_state=seed)))
             ])
             )
        )
        pipelines.append(
            ('AdaBoostClassifier',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ('AdaBoostClassifier', OneVsRestClassifier(AdaBoostClassifier(random_state=seed)))
             ]))
        )
        pipelines.append(
            ('RandomForestClassifier',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ('RandomForestClassifier', OneVsRestClassifier(RandomForestClassifier(random_state=seed)))
             ]))
        )
        pipelines.append(
            ('RandomForestClassifier',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ('RandomForestClassifier', OneVsRestClassifier(RandomForestClassifier(random_state=seed)))
             ]))
        )
        pipelines.append(
            ('LinearSVC',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ('LinearSVC', OneVsRestClassifier(LinearSVC(random_state=seed)))
             ]))
        )
        pipelines.append(
            ('KNeighborsClassifier',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ('KNeighborsClassifier', OneVsRestClassifier(KNeighborsClassifier()))
             ]))
        )

        pipelines.append(
            ('GaussianNB',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ('GaussianNB', OneVsRestClassifier(GaussianNB()))
             ]))
        )

        pipelines.append(
            ('Perceptron',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ('Perceptron', OneVsRestClassifier(Perceptron(random_state=seed)))
             ]))
        )
        pipelines.append(
            ('SGDClassifier',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ('SGDClassifier', OneVsRestClassifier(SGDClassifier(random_state=seed)))
             ]))
        )
        pipelines.append(
            ('DecisionTreeClassifier',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ('DecisionTreeClassifier', OneVsRestClassifier(DecisionTreeClassifier(random_state=seed)))
             ]))
        )
        pipelines.append(
            ('LogisticRegression',
             Pipeline([
                 ('Scaler', StandardScaler()),
                 ('LogisticRegression', OneVsRestClassifier(LogisticRegression(random_state=seed)))
             ]))
        )
        scoring = 'accuracy'
        n_folds = 10
        results, names = [], []
        for name, model in pipelines:
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            cv_results = cross_val_score(model, self.X_train, self.y_train, cv=kfold,
                                         scoring=scoring, n_jobs=-1)
            names.append(name)
            results.append(cv_results)
            msg = "%s: %f (+/- %f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)

# if __name__ == '__main__':
    # classifier = SentimentClassifier()
    # classifier.choose_best_model()
    # classifier.train()
    # classifier.plot_learning_curve()
