
# coding: utf-8

# # 数据清洗
# 
# 数据清洗策略子项:
# - 申请者个人 / 家庭基本信息
# - 申请者联系方式
# - 申请者车辆购置情况
# - 申请者工作情况
# - 申请者房产情况
# - 贷款申请材料
# - 申请者社交状况
# - 风险评估
# 
# 每个策略子项从两个方面进行清洗:
# - 与业务逻辑相结合判断是否存在异常值
# - 从统计学意义上进行判断是否存在异常值
# 
# 清洗完成后:
# - **标记**异常项, 暂时不删除异常项
# - 对文本类型的特征项进行数值化处理
# - 对数值化类型的特征项进行合适的数值化处理
# - 对于缺失值填充`np.nan`或者单独列成一类

# In[2]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging

class DataCleaner(object):
    def __init__(self, data_dir, app_train, app_test,
                 bureau, bureau_balance, credit_card_balance,
                 installments_payments, pos_cache_balance, previous_app,
                 output_dir):
        self.__data_dir = data_dir
        self.__app_train = app_train
        self.__app_test = app_test
        self.__bureau = bureau
        self.__bureau_balance = bureau_balance
        self.__credit_card_balance = credit_card_balance
        self.__installments_payments = installments_payments
        self.__pos_cache_balance = pos_cache_balance
        self.__previous_app = previous_app
        self.__output_dir = output_dir

    def clean_data(self):
        user_df = pd.read_csv(os.path.join(self.__data_dir, self.__app_train))
        cleaned_df = user_df.loc[:, ['SK_ID_CURR', 'TARGET']].copy().astype(np.float64)
        cleaned_df = self.clean_app_info(user_df, cleaned_df)

    def discretize_column(self, column, replace_nan=False):
        assert isinstance(column, pd.Series), "Invalid column type: %s"%(type(column),)
        str_values = column.loc[column.notnull()].unique()
        numeric_values = list(range(len(str_values)))
        column = column.replace(str_values, numeric_values).astype(np.float32)
        if replace_nan:
            column = column.fillna(-1)
        column = column.astype(np.float64)
        logging.debug("[%s] %s -> %s"%(column.name, str_values, numeric_values,))
        logging.debug("[%s] replace_nan: %r"%(column.name, replace_nan,))
        return column

    def clean_app_info(self, user_df, cleaned_df):
        assert isinstance(user_df, pd.DataFrame) and isinstance(cleaned_df, pd.DataFrame)
        logging.debug("clean_app_info begin...")

        cleaned_df.loc[:, 'CODE_GENDER'] = self.discretize_column(user_df['CODE_GENDER'].replace(['XNA'], [np.nan]))
        cleaned_df.loc[:, 'DAYS_BIRTH'] = -user_df['DAYS_BIRTH'].astype(np.float64)
        cleaned_df.loc[:, 'DAYS_REGISTRATION'] = -user_df['DAYS_REGISTRATION'].astype(np.float64)
        cleaned_df.loc[:, 'DAYS_ID_PUBLISH'] = -user_df['DAYS_ID_PUBLISH'].astype(np.float64)
        cleaned_df.loc[:, 'NAME_EDUCATION_TYPE'] = self.discretize_column(user_df['NAME_EDUCATION_TYPE'])
        cleaned_df.loc[:, 'CNT_CHILDREN'] = user_df['CNT_CHILDREN'].astype(np.float64)
        cleaned_df.loc[:, 'CNT_FAM_MEMBERS'] = user_df['CNT_FAM_MEMBERS'].fillna(np.nan).astype(np.float64)
        cleaned_df.loc[:, 'NAME_FAMILY_STATUS'] = self.discretize_column(user_df['NAME_FAMILY_STATUS'])

        cleaned_df.loc[:, 'FLAG_MOBIL'] = user_df['FLAG_MOBIL'].astype(np.float64)
        cleaned_df.loc[:, 'FLAG_PHONE'] = user_df['FLAG_PHONE'].astype(np.float64)
        cleaned_df.loc[:, 'FLAG_EMAIL'] = user_df['FLAG_EMAIL'].astype(np.float64)
        cleaned_df.loc[:, 'FLAG_CONT_MOBILE'] = user_df['FLAG_CONT_MOBILE'].astype(np.float64)

        cleaned_df.loc[:, 'FLAG_OWN_CAR'] = self.discretize_column(user_df['FLAG_OWN_CAR'])
        cleaned_df.loc[:, 'OWN_CAR_AGE'] = user_df['OWN_CAR_AGE'].astype(np.float64)

        cleaned_df.loc[:, 'DAYS_EMPLOYED'] = -(user_df['DAYS_EMPLOYED'].replace(365243, np.nan)).astype(np.float64)
        cleaned_df.loc[:, 'FLAG_EMP_PHONE'] = user_df['FLAG_EMP_PHONE'].astype(np.float64)
        cleaned_df.loc[:, 'FLAG_WORK_PHONE'] = user_df['FLAG_WORK_PHONE'].astype(np.float64)
        cleaned_df.loc[:, 'ORGANIZATION_TYPE'] = self.discretize_column(user_df['ORGANIZATION_TYPE'].replace(['XNA'], [np.nan]))
        cleaned_df.loc[:, 'NAME_INCOME_TYPE'] = self.discretize_column(user_df['NAME_INCOME_TYPE'])
        cleaned_df.loc[:, 'AMT_INCOME_TOTAL'] = user_df['AMT_INCOME_TOTAL'].astype(np.float64)
        cleaned_df.loc[:, 'OCCUPATION_TYPE'] = self.discretize_column(user_df['OCCUPATION_TYPE'])

        cleaned_df.loc[:, 'FLAG_OWN_REALTY'] = self.discretize_column(user_df['FLAG_OWN_REALTY'])
        cleaned_df.loc[:, 'NAME_HOUSING_TYPE'] = self.discretize_column(user_df['NAME_HOUSING_TYPE'])
        cleaned_df.loc[:, 'FONDKAPREMONT_MODE'] = self.discretize_column(user_df['FONDKAPREMONT_MODE'])
        cleaned_df.loc[:, 'HOUSETYPE_MODE'] = self.discretize_column(user_df['HOUSETYPE_MODE'])
        cleaned_df.loc[:, 'TOTALAREA_MODE'] = user_df['TOTALAREA_MODE'].fillna(np.nan).astype(np.float64)
        cleaned_df.loc[:, 'WALLSMATERIAL_MODE'] = self.discretize_column(user_df['WALLSMATERIAL_MODE'])
        cleaned_df.loc[:, 'EMERGENCYSTATE_MODE'] = self.discretize_column(user_df['EMERGENCYSTATE_MODE'])

        realtyFactors = [
            'REGION_POPULATION_RELATIVE',
            'REGION_RATING_CLIENT',
            'REGION_RATING_CLIENT_W_CITY',
            'APARTMENTS_AVG',
            'BASEMENTAREA_AVG',
            'YEARS_BEGINEXPLUATATION_AVG',
            'YEARS_BUILD_AVG',
            'COMMONAREA_AVG',
            'ELEVATORS_AVG',
            'ENTRANCES_AVG',
            'FLOORSMAX_AVG',
            'FLOORSMIN_AVG',
            'LANDAREA_AVG',
            'LIVINGAPARTMENTS_AVG',
            'LIVINGAREA_AVG',
            'NONLIVINGAPARTMENTS_AVG',
            'NONLIVINGAREA_AVG',
            'APARTMENTS_MODE',
            'BASEMENTAREA_MODE',
            'YEARS_BEGINEXPLUATATION_MODE',
            'YEARS_BUILD_MODE',
            'COMMONAREA_MODE',
            'ELEVATORS_MODE',
            'ENTRANCES_MODE',
            'FLOORSMAX_MODE',
            'FLOORSMIN_MODE',
            'LANDAREA_MODE',
            'LIVINGAPARTMENTS_MODE',
            'LIVINGAREA_MODE',
            'NONLIVINGAPARTMENTS_MODE',
            'NONLIVINGAREA_MODE',
            'APARTMENTS_MEDI',
            'BASEMENTAREA_MEDI',
            'YEARS_BEGINEXPLUATATION_MEDI',
            'YEARS_BUILD_MEDI',
            'COMMONAREA_MEDI',
            'ELEVATORS_MEDI',
            'ENTRANCES_MEDI',
            'FLOORSMAX_MEDI',
            'FLOORSMIN_MEDI',
            'LANDAREA_MEDI',
            'LIVINGAPARTMENTS_MEDI',
            'LIVINGAREA_MEDI',
            'NONLIVINGAPARTMENTS_MEDI',
            'NONLIVINGAREA_MEDI']
        for curFactor in realtyFactors:
            cleaned_df.loc[:, curFactor] = user_df[curFactor].fillna(np.nan).astype(np.float64)

        cleaned_df.loc[:, 'DAYS_LAST_PHONE_CHANGE'] = -(user_df['DAYS_LAST_PHONE_CHANGE']).fillna(np.nan).astype(np.float64)

        riskFactors = [
            'REG_REGION_NOT_LIVE_REGION',
            'REG_CITY_NOT_LIVE_CITY',
            'REG_CITY_NOT_WORK_CITY',
            'LIVE_CITY_NOT_WORK_CITY',
            'REG_REGION_NOT_WORK_REGION',
            'LIVE_REGION_NOT_WORK_REGION',
            'EXT_SOURCE_1',
            'EXT_SOURCE_2',
            'EXT_SOURCE_3'
        ]
        for curFactor in riskFactors:
            cleaned_df.loc[:, curFactor] = user_df[curFactor].fillna(np.nan).astype(np.float64)

        cleaned_df.loc[:, 'NAME_CONTRACT_TYPE'] = self.discretize_column(user_df['NAME_CONTRACT_TYPE'])
        cleaned_df.loc[:, 'NAME_TYPE_SUITE'] = self.discretize_column(user_df['NAME_TYPE_SUITE'])

        weekday = [
            'MONDAY',
            'TUESDAY',
            'WEDNESDAY',
            'THURSDAY',
            'FRIDAY',
            'SATURDAY',
            'SUNDAY'
        ]
        cleaned_df.loc[:, 'WEEKDAY_APPR_PROCESS_START'] = user_df['WEEKDAY_APPR_PROCESS_START'].replace(weekday, range(
            len(weekday))).fillna(np.nan).astype(np.float64)

        cleaned_df.loc[:, 'HOUR_APPR_PROCESS_START'] = user_df['HOUR_APPR_PROCESS_START'].fillna(np.nan).astype(
            np.float64)

        appFactors = [
            'AMT_CREDIT',
            'AMT_ANNUITY',
            'AMT_GOODS_PRICE',
            'AMT_REQ_CREDIT_BUREAU_HOUR',
            'AMT_REQ_CREDIT_BUREAU_DAY',
            'AMT_REQ_CREDIT_BUREAU_WEEK',
            'AMT_REQ_CREDIT_BUREAU_MON',
            'AMT_REQ_CREDIT_BUREAU_QRT',
            'AMT_REQ_CREDIT_BUREAU_YEAR',
            'FLAG_DOCUMENT_2',
            'FLAG_DOCUMENT_3',
            'FLAG_DOCUMENT_4',
            'FLAG_DOCUMENT_5',
            'FLAG_DOCUMENT_6',
            'FLAG_DOCUMENT_7',
            'FLAG_DOCUMENT_8',
            'FLAG_DOCUMENT_9',
            'FLAG_DOCUMENT_10',
            'FLAG_DOCUMENT_11',
            'FLAG_DOCUMENT_12',
            'FLAG_DOCUMENT_13',
            'FLAG_DOCUMENT_14',
            'FLAG_DOCUMENT_15',
            'FLAG_DOCUMENT_16',
            'FLAG_DOCUMENT_17',
            'FLAG_DOCUMENT_18',
            'FLAG_DOCUMENT_19',
            'FLAG_DOCUMENT_20',
            'FLAG_DOCUMENT_21'
        ]
        for curFactor in appFactors:
            cleaned_df.loc[:, curFactor] = user_df[curFactor].fillna(np.nan).astype(np.float64)

        socialFactors = [
            'OBS_30_CNT_SOCIAL_CIRCLE',
            'DEF_30_CNT_SOCIAL_CIRCLE',
            'OBS_60_CNT_SOCIAL_CIRCLE',
            'DEF_60_CNT_SOCIAL_CIRCLE'
        ]

        for curFactor in socialFactors:
            cleaned_df.loc[:, curFactor] = user_df[curFactor].fillna(np.nan).astype(np.float64)

        logging.debug("clean_app_info done!")
        return cleaned_df

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--data_dir', type=str,
                           default='../../data',
                           help='Directory for storing input data')
    argParser.add_argument('--app_train', type=str,
                           default='application_train.csv',
                           help='File name for application_train')
    argParser.add_argument('--app_test', type=str,
                           default='application_test.csv',
                           help='File name for application_test')
    argParser.add_argument('--bureau', type=str,
                           default='bereau.csv',
                           help='File name for bereau')
    argParser.add_argument('--bureau_balance', type=str,
                           default='bureau_balance.csv',
                           help='File name for bereau_balance')
    argParser.add_argument('--credit_card_balance', type=str,
                           default='credit_card_balance.csv',
                           help='File name for credit_card_balance')
    argParser.add_argument('--installments_payments', type=str,
                           default='installments_payments.csv',
                           help='File name for installments_payments')
    argParser.add_argument('--POS_CACHE_balance', type=str,
                           default='POS_CACHE_balance.csv',
                           help='File name for POS_CACHE_balance')
    argParser.add_argument('--previous_application', type=str,
                           default='previous_application.csv',
                           help='File name for previous_application')
    argParser.add_argument('--output_dir', type=str,
                           default='../../data/output',
                           help='Directory for storing output data')
    FLAGS, _ = argParser.parse_known_args()

    dataCleaner = DataCleaner(FLAGS.data_dir,
                              FLAGS.app_train,
                              FLAGS.app_test,
                              FLAGS.bureau,
                              FLAGS.bureau_balance,
                              FLAGS.credit_card_balance,
                              FLAGS.installments_payments,
                              FLAGS.POS_CACHE_balance,
                              FLAGS.previous_application,
                              FLAGS.output_dir)
    dataCleaner.clean_data()