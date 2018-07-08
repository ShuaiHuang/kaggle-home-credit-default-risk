# Exploratory Data Analysis

## What is EDA(Exploratory Data Analysis) ?

> The goal is to show the data, summarize the evidence and identify interesting patterns while eliminating ideas that likely won’t pan out.　　——Exploratory Data Analysis wiht R

- [Wiki](https://en.wikipedia.org/wiki/Exploratory_data_analysis)
- [Book](http://theta.edu.pl/wp-content/uploads/2012/10/exploratorydataanalysis_tukey.pdf)
- [知乎](https://zhuanlan.zhihu.com/p/29364225)

## 什么是EDA?

EDA是对**真实数据**进行深入认识的套路. 通常从以下几个方面进行分析:

1. 基本数据信息
2. 缺失值/有效值处理 [Ref](https://zhuanlan.zhihu.com/p/32473525)
    - 简单插补
    - 多重插补 *缺失数据集——MCMC估计插补成几个数据集——每个数据集进行插补建模（glm、lm模型）——将这些模型整合到一起（pool）——评价插补模型优劣（模型系数的t统计量）——输出完整数据集（compute）*
3. 单变量分析
4. 关联性分析
5. 组合分析