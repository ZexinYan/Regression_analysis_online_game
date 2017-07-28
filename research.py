import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from helper import Helper

warnings.filterwarnings("ignore")
data = pd.read_csv('./data/online_game.csv')


def related_to(col):
    data_dummy = pd.get_dummies(data[col], col)
    data_dummy = pd.DataFrame(pd.concat([data_dummy, data[u'评论']], axis=1))
    Helper().plot_correlation_heap(data_dummy, u'评论', title=str(col) + '的相关关系')


def fill_zero(value):
    return value + 1

# for each in data.columns:
#     related_to(each)


skewness = data[u'评论'].skew()
sns.distplot(data[u'评论'], kde=True, label=str(round(skewness, 2)))
plt.legend()
plt.show()

comment = list(map(fill_zero, data[u'评论']))
skewness = np.log(pd.DataFrame(comment).skew())
sns.distplot(np.log(pd.DataFrame(comment)), kde=True)
plt.legend()
plt.show()
