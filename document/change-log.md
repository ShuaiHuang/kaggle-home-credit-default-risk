# Change Log

## 20180621

### 赛题分析

- 规则
    + 每天提交5次
    + 可以选择2个提交作为最终结果
- 重要时间点
    + 开始时间: 2018年5月17日
    + 组队截止日期: 2018年8月22日
    + 首次提交截止日期: 2018年8月22日
    + 最终截止日期: 2018年8月29日 11:59 PM UTC

    > 比赛初期使用个人身份进行参赛, 使用不同的策略和模型进行提交, 充分利用每人每天10次的提交机会, 在8月初进行组队.

### 准备工作

- 搭建环境
    + 开发环境 Anaconda sklearn
    + 网络环境 VPN
    + [Repository@GitOSC](https://gitee.com/ShuaiHuang/kaggle-home-credit-default-risk)
- 理解数据
- 理解业务背景
    + 风控基本知识
        - [风控模型评价指标](https://zhuanlan.zhihu.com/p/27362846)
        - [风控中机器学习算法比较](https://zhuanlan.zhihu.com/p/27326824)
    + 比赛赞助商情况
        - [Link](https://www.kaggle.com/c/home-credit-default-risk/discussion/57054)
    + 相关文献
    + 竞赛讨论区

### 计划

- 20180622 梳理数据, 搞清楚数据各个字段的含义; 搞清楚评价标准; 初步思考数据清洗策略.
- 20180623 - 0624 了解业界的经典方法, 以及state-of-the-art所能达到的水平.
- 20180625 - 0629 实现模型, 两个账号分别实现第一次有效提交.

## 20180622

- `application_{train|test}.csv`
    + 申请人原始资料
    + 原始数据高维, 稀疏, 有大量缺失值; *是否可以参考蚂蚁金服的做法, 使用周志华的deep forest作为模型?*
    + 训练数据集30w+条数据, 训练数据集4.8w+条数据; *考虑是否需要借助数据库进行数据管理*
- `bureau.csv`
    + 其他金融机构提供给信用管理机构的数据
- `bureau_balances.csv`
    + 申请人月度信用记录
    + 可以考虑抽取序列特征
- `POS_CASH_balance.csv`
    + 现金账单收支平衡
- `credit_card_balance.csv`
    + 信用卡收支平衡
- `previous_application.csv`
    + 历史申请记录
- `installments_payments.csv`
    + 历史还款记录
- `HomeCredit_columns_description.csv`
    + **原始数据每一列的解释, 有助于建立特征工程.**

![img](https://storage.googleapis.com/kaggle-media/competitions/home-credit/home_credit.png)

## 20180624

- 业界内常用模型
    + GBDT / MART
    + XGBOOST
        + [XGBOOST浅入浅出](http://wepon.me/2016/05/07/XGBoost%E6%B5%85%E5%85%A5%E6%B5%85%E5%87%BA/)
        + [XGBOOST与Boosted Tree](http://www.52cs.org/?p=429)
        + [XGBOOST slide](https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf)
        + [开源代码](https://xgboost.readthedocs.io/en/latest/)
    + Deep Forest
        + [论文](https://arxiv.org/pdf/1702.08835.pdf)
        + [开源代码](http://lamda.nju.edu.cn/code_gcForest.ashx)
- 计划本周内使用上述三种模型, 在两个账号上分别进行首次有效提交.
- 后续依据特征工程以及模型训练效果, 使用单个模型或者使用stacking方法进行结果融合.

## 20180630

- Ubuntu系统中将`xgboost`安装到`anaconda`环境的方法
    + 按照[说明](https://github.com/dmlc/xgboost/tree/master/python-package)编译`xgboost`
    + 执行`setup.py`脚本时, 使用`sudo ~/anaconda3/envs/py3/bin/python3 setup.py install`进行安装(因为`steup.py`默认调用的是系统的python解释器, 尽管使用`which python时看到的是anaconda的解释器`)

## 20180701

- `xgboost`方法与传统`boost`方法的区别在于
    + 将分类器启发式`heuristic`生成过程等价替换成目标式`objective`
    + 将CART树的属性分类过程目标函数中引入预剪枝策略
    + `xgboost`的并行化是在特征粒度上的并行化

## 20180705

- 不错的数据入门介绍, 可以在此基础上进行拓展
    + 数据预处理[Start Here: A Gentle Introduction](https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction)
    + 特征工程 [Introduction to Manual Feature Engineering](https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering)

## 20180714

- 配置好`gcForest`环境: 为了保证版本兼容性, 在`Anaconda`基于`Python 3.5`版本配置了全新的环境. `gcForest`工程目录下的`requirement.txt`中所要求的安装包除了`argparse`都进行了安装. `argparse`要求的Python版本号是2.6, 在3.5版本中得到了支持, 因此跑示例程序的时候没有报错.
- 按照不同的粒度整理`application_{train|test}.csv`表格中的特征, 并进行进一步的数据清洗. 具体清理结果参见[这里](./exploratory-data-analysis.md).
- 看到一片关于特征选择的[知乎文章](https://zhuanlan.zhihu.com/p/39695931), 其中介绍了[Feature Selector](https://github.com/WillKoehrsen/feature-selector)这个工具箱.

## 20180724

- 完成主表的数据清洗以及数值化操作, 所有的缺失值均使用`np.nan`进行填充.
- 下一步计划
    1. 分隔训练集以及验证集
    2. 使用`xgboost`进行训练

## 总结

Home Credit Default Risk比赛于2018年8月29日截止, 最终成绩为3455名, auc指标为0.78755, 并不是一个好名次. 但是反思自己的比赛过程以及观摩别人的成功经验, 可以得到如下的启示:

- 大数据比赛的重点在于**特征工程**, 如果想得到好的名次, 必须要从原始数据中的原始特征进行抽取加工, 得到更有启发性的新特征. 但是在这次比赛中, 我的精力更多集中在软件架构的设计上, 先后设计出网格搜索和调参的基本软件框架, 提升了比赛后期的操作便利性. 但是由于忽视了特征工程, 最后提升的空间很有限.
- 对于时序数据的特征提取缺乏经验. 尤其是比赛中提供的一对多映射关系的数据, 很难直接利用原始特征.
- 比赛过程忽视了业务知识. 这个从比赛选题就需要加以重视, 选择一个熟悉的业务背景更加有助于特征工程的开展.