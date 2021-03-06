# [机器学习]回归分析案例 --- 网游排名分析
## 项目描述
[基础案例 | 玩家必看之网游排行](http://mp.weixin.qq.com/s?__biz=MzA5MjEyMTYwMg==&mid=2650239087&idx=4&sn=fd39d699d76844f360e35d0344b694d4&chksm=88722f02bf05a6142328e900dc84311003469969692e35b57b82f67622045c39ba0021e0e085&mpshare=1&scene=1&srcid=0727NlwQeB4EYe93C3PbUlIg#rd)

## big picture
从项目描述以及数据上看，可以很明显地发现，这是一个非常文本化的数据，也就是无法直接使用机器学习来解决问题。那么，这就非常依赖于将文本量化的能力。本文，我会尝试着完整地把整个项目实现一遍，并给出我的一些理解和感悟。


## Research
首先，我们需要对文本数据有一个大体的认识。

### 各字段的分类数

```
网游名称
(1943, 1943)
画面
(1943, 3)
画风
(1943, 4)
开发商
(1943, 896)
运营商
(1943, 518)
题材
(1943, 8)
模式
(1943, 2)
收费
(1943, 6)
评论
(1943, 823)
类型
(1943, 17)
开发时间
(1943, 675)
背景
(1943, 1943)
```

### 网游名称

![Stary 2017-07-27 at 9.18.31 P](http://o7d2h0gjo.bkt.clouddn.com/2017-07-28-Stary%202017-07-27%20at%209.18.31%20PM.png)

因为每一个网游名称都是唯一的，所以不应该用dummy的方法来构造新的字段。而这个图的形成有可能只是因为有的游戏名称具有更多的评论数。虽然说，网游的名称可以给我们一个对游戏非常直观的认识，但在实际操作中，如果想要对游戏名本身进行数据挖掘，可能是一件不那么容易的事情。这里，我不对此做研究。

### 背景
虽然背景信息对于每一个游戏来说都是唯一的，但背景信息中涵盖了很多的内容，这些内容是不应该被忽略的，因此我们需要对背景信息进行一定的数据挖掘。我在之前的文章中提到了一种实现情感分析的方法（[情感分析实例](http://blog.csdn.net/stary_yan/article/details/75313259)），在这里其实也可以这么做，但是具体策略不同。对于背景信息来说，不再是单纯的二分情感分析，而是`情感分类`，简单地说，很难通过阅读背景信息知道这个这个游戏会不会足够吸引人。因为数据量并不够大（2000条），如果要细分的话，对于任何一个分类来说，数据都是不够的。因此，这个简单地就采用2分了。
结果是，我们training model有超过50%的准确性。这已经足够了。因为，就算是任何一个人去阅读这个背景信息都很难有确切的把握能够预测这个游戏是否足够吸引人，（如果真的有这样的人，那他一定已经统领世界游戏圈了。）并且事实上，游戏的受喜好程度也绝不是仅仅靠背景信息就能确定的。这里只要能够不引入噪声，并且确保这个参数能够对回归分析有一定的效果即可。
另外，有一个非常重要的事情需要注意，这里我们决定不可以把背景信息直接根据评论数量量化，也就是说不能因为一个游戏的评论数量最大，就把这个背景信息转化为1。因为这相当于直接根据结果构造参数，因为对于一个新的游戏来说，你是不知道评论数的，也就是说你无法对一个新的数据做相应的量化。这里我能使用的model predict的结果来量化背景信息。因为我相信我的背景信息model是准确的，他能够通过阅读一个游戏的背景信息来发现游戏是否能够被喜好，并且他也能适用于一个新的数据，所以这么做是可行。

重要参数：

```
{'中国', '风格', '运营', '自主', '网游', '背景', '模式', '研发', '开发', '世界', '网络游戏', '2D', '玩法', '网络', '韩国', '系统', '动作', '新', 'OL', '角色', '引擎', '题材', '大型', '体验', '3D', 'MMORPG'}
```


### 开发商&运营商
![Stary 2017-07-27 at 9.19.07 P](http://o7d2h0gjo.bkt.clouddn.com/2017-07-28-Stary%202017-07-27%20at%209.19.07%20PM.png)

![Stary 2017-07-27 at 9.19.31 P](http://o7d2h0gjo.bkt.clouddn.com/2017-07-28-Stary%202017-07-27%20at%209.19.31%20PM.png)

从相关系数的分析来看，开发商和运营商对于决定游戏的被喜好程度是十分重要的。这也符合一般认识，毕竟大公司做出的游戏一般总应该比小公司做出的游戏要更好。但因为分类数目太多，无法进行特别细化的区分，只能根据他们的相关关系大小做一个简单地划分，并且进行相应的量化。例如相关关系更大的开发商或运营商具有更大的参数值。

### 画面与评论数的关系
![Stary 2017-07-27 at 9.18.41 P](http://o7d2h0gjo.bkt.clouddn.com/2017-07-28-Stary%202017-07-27%20at%209.18.41%20PM.png)


![Stary 2017-07-27 at 9.18.50 P](http://o7d2h0gjo.bkt.clouddn.com/2017-07-28-Stary%202017-07-27%20at%209.18.50%20PM.png)

![Stary 2017-07-27 at 9.19.41 P](http://o7d2h0gjo.bkt.clouddn.com/2017-07-28-Stary%202017-07-27%20at%209.19.41%20PM.png)

![Stary 2017-07-27 at 9.19.49 P](http://o7d2h0gjo.bkt.clouddn.com/2017-07-28-Stary%202017-07-27%20at%209.19.41%20PM.png)

![Stary 2017-07-27 at 9.19.55 P](http://o7d2h0gjo.bkt.clouddn.com/2017-07-28-Stary%202017-07-27%20at%209.19.55%20PM.png)

![Stary 2017-07-27 at 9.20.13 P](http://o7d2h0gjo.bkt.clouddn.com/2017-07-28-Stary%202017-07-27%20at%209.20.13%20PM.png)


以上的分类数相对较少，而且有比较显著的相关关系，所以可以不做修改，直接采用dummies作为参数。

### 优化
根据以上的结果直接进行分类以后会发现`r2`结果非常的糟糕，可能就将近0.2。这基本可以认为这是一个无效model。但仔细想一想，我选用的每一个参数都是有意义的，为什么做到的model却几乎认为这些参数都是无效，甚至是噪声？

这里我忽略了一个问题。自己回顾分类数目的时候，会发现他们之间大多相差巨大，在这样的情况下，进行回归分析是非常困难的。并且我也发现原始参数的分布类似于高斯分布。

![Stary 2017-07-28 at 8.08.10 P](http://o7d2h0gjo.bkt.clouddn.com/2017-07-28-Stary%202017-07-28%20at%208.08.10%20PM.png)

因此，在这里可以直接把对评论数进行log运算。可以得到以下的结果，可以认为是一个非常漂亮的高斯分布了。事实上，对于回归分析来说，最易于成功实现回归分析的数据分布情况就是高斯分布。这也提醒我在以后进行回归分析的时候，首先需要考虑的就是因变量的分布情况。

![Stary 2017-07-28 at 8.08.16 P](http://o7d2h0gjo.bkt.clouddn.com/2017-07-28-Stary%202017-07-28%20at%208.08.16%20PM.png)



## 结果
最后模型测试结果如下。并不是特别好的结果，最好的情况也仅仅只是0.63。至于如何优化模型，在此我尚未想到更好的方法。当然，也有可能是因为数据量不足，也可能是因为数据本身就无法被很好地回归分析。当然，肯定还有需要方法可以用于优化模型，只是我能力不足，无法做到。未来有机会，可以继续优化此模型。


```
Scaled_Ridge: 0.621350 (+/- 0.026157)
Scaled_Lasso: 0.464849 (+/- 0.060175)
Scaled_SVR: 0.638342 (+/- 0.035749)
Scaled_RF: 0.627724 (+/- 0.044188)
Scaled_ET: 0.589728 (+/- 0.050206)
Scaled_BR: 0.626910 (+/- 0.041922)
LinearRegression: 0.620649 (+/- 0.025597)
```


## 完整代码
[Github](https://github.com/ZexinYan/Regression_analysis_online_game)
