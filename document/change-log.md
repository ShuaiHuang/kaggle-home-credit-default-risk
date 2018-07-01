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
