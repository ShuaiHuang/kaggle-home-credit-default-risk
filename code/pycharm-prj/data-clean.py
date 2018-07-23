
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

FLAGS = None

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

    def clean_app_train(self):
        pass

# In[3]:


DATA_DIR = '../data'
APP_TRAIN_FILENAME = 'application_train.csv'
APP_TEST_FILENAME = 'application_test.csv'
BUREAU_FILENAME = 'bereau.csv'
BUREAU_BALANCE_FILENAME = 'bureau_balance.csv'
CREDIT_CARD_BALANCE_FILENAME = 'credit_card_balance.csv'
INSTALLMENTS_PAYMENTS = 'installments_payments.csv'
POS_CACHE_BALANCE_FILENAME = 'POS_CACHE_balance.csv'
PREVIOUS_APP_FILENAME = 'previous_application.csv'


# In[4]:


user_df = pd.read_csv(os.path.join(DATA_DIR, APP_TRAIN_FILENAME))
cleaned_df = user_df.loc[:, ['SK_ID_CURR', 'TARGET']].copy()


# In[5]:


def detectOutlier(dataSeries):
    quantile_1_4 = dataSeries.quantile(0.25)
    quantile_3_4 = dataSeries.quantile(0.75)
    validRange = 1.5 * (quantile_3_4 - quantile_1_4)
    minValue = quantile_1_4 - validRange
    maxValue = quantile_3_4 + validRange
    return dataSeries[(dataSeries > maxValue) | (dataSeries) < minValue]


# In[6]:


def detectOutlierRecords(dataFrame, cols):
    outlierCounter = np.full((dataFrame.shape[0], ), 0, dtype=np.int)
    for col in cols:
        curOutlierFlag = detectOutlier(dataFrame[col])
        print(col, curOutlierFlag)
        outlierCounter[curOutlierFlag] = outlierCounter[curOutlierFlag] + 1
    return outlierCounter


# ## 申请者个人 / 家庭基本信息
# 
# 主要包含以下字段:
# - SK_ID_CURR
# - TARGET
# - CODE_GENDER
# - DAYS_BIRTH
# - DAYS_REGISTRATION
# - DAYS_ID_PUBLISH
# - NAME_EDUCATION_TYPE
# - CNT_CHILDREN
# - CNT_FAM_MEMBERS
# - NAME_FAMILY_STATUS
# 
# 有效性检查包括:
# 
# - DAYS_BIRTH >= DAYS_REGISTRATION
# - ~~DAYS_BIRTH >= DAYS_ID_PUBLISH~~
# - ~~NAME_FAMILY_STATUS 是否与 CNT_CHILDREN 以及 CNT_FAM_MEMBERS 冲突~~
# - ~~CNT_FAM_MEMBERS < CNT_CHILDREN~~
# 
# **ATTN**:
# - 对于原表使用`loc`选择单列是引用, 选择多列是复制.

# In[7]:


PER_FAM_FACTORS = ['SK_ID_CURR', 'TARGET', 'CODE_GENDER',
                   'DAYS_BIRTH', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
                   'NAME_EDUCATION_TYPE', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS',
                   'NAME_FAMILY_STATUS']


# In[8]:


print('-' * 100)
print(user_df.loc[user_df['DAYS_BIRTH'] > user_df['DAYS_REGISTRATION'], ['SK_ID_CURR', 'TARGET', 'DAYS_BIRTH', 'DAYS_REGISTRATION']])
print('-' * 100)
print(user_df.loc[user_df['DAYS_BIRTH'] > user_df['DAYS_ID_PUBLISH'], ['SK_ID_CURR', 'TARGET', 'DAYS_BIRTH', 'DAYS_ID_PUBLISH']])
print('-' * 100)
print(user_df.loc[user_df['CNT_FAM_MEMBERS'] < user_df['CNT_CHILDREN'], ['SK_ID_CURR', 'TARGET', 'CNT_FAM_MEMBERS', 'CNT_CHILDREN']])
print('-' * 100)


# ### CODE_GENDER
# 
# - 'M' - 0
# - 'F' - 1
# - 'XNA' - `np.nan`

# In[9]:


user_df.loc[:, 'CODE_GENDER'].unique()


# In[10]:


cleaned_df.loc[:, 'CODE_GENDER'] = user_df['CODE_GENDER'].replace(['M', 'F', 'XNA'], [0, 1, np.nan])


# ### DAYS_BIRTH

# In[11]:


cleaned_df.loc[:, 'DAYS_BIRTH'] = -user_df['DAYS_BIRTH']


# ### DAYS_REGISTRATION

# In[12]:


cleaned_df.loc[:, 'DAYS_REGISTRATION'] = -user_df['DAYS_REGISTRATION']


# ### DAYS_ID_PUBLISH

# In[13]:


cleaned_df.loc[:, 'DAYS_ID_PUBLISH'] = -user_df['DAYS_ID_PUBLISH']


# ### NAME_EDUCATION_TYPE
# 
# - `Secondary / secondary special` - 0
# - `Higher education` - 1
# - `Incomplete higher` - 2
# - `Lower secondary` - 3
# - `Academic degree` - 4

# In[14]:


cleaned_df.loc[:, 'NAME_EDUCATION_TYPE'] = user_df['NAME_EDUCATION_TYPE'].replace(['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree'], range(0, 5))


# ### CNT_CHILDREN

# In[15]:


cleaned_df.loc[:, 'CNT_CHILDREN'] = user_df['CNT_CHILDREN']


# ### CNT_FAM_MEMBERS

# In[16]:


cleaned_df.loc[:, 'CNT_FAM_MEMBERS'] = user_df['CNT_FAM_MEMBERS'].fillna(np.nan)


# ### NAME_FAMILY_STATUS
# 
# - Married 0
# - Single / not married 1
# - Civil marriage 2
# - Separated 3
# - Widow 4
# - Unknown 5

# In[17]:


cleaned_df.loc[:, 'NAME_FAMILY_STATUS'] = user_df['NAME_FAMILY_STATUS'].replace(['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow', 'Unknown'], [0, 1, 2, 3, 4, np.nan])


# ## 申请者联系方式
# 
# - FLAG_MOBIL
# - FLAG_PHONE
# - FLAG_EMAIL
# - FLAG_CONT_MOBILE

# In[18]:


CONTACT_FACTOR = ['FLAG_MOBIL', 'FLAG_PHONE', 'FLAG_EMAIL', 'FLAG_CONT_MOBILE']


# ### FLAG_MOBIL

# In[19]:


cleaned_df.loc[:, 'FLAG_MOBIL'] = user_df['FLAG_MOBIL']


# ### FLAG_PHONE

# In[20]:


cleaned_df.loc[:, 'FLAG_PHONE'] = user_df['FLAG_PHONE']


# ### FLAG_EMAIL

# In[21]:


cleaned_df.loc[:, 'FLAG_EMAIL'] = user_df['FLAG_EMAIL']


# ### FLAG_CONT_MOBILE

# In[22]:


cleaned_df.loc[:, 'FLAG_CONT_MOBILE'] = user_df['FLAG_CONT_MOBILE']


# ## 申请者车辆购置情况
# 
# - FLAG_OWN_CAR
# - OWN_CAR_AGE

# ### FLAG_OWN_CAR
# 
# - N - 0
# - Y - 1

# In[23]:


cleaned_df.loc[:, 'FLAG_OWN_CAR'] = user_df['FLAG_OWN_CAR'].replace(['N', 'Y'], [0, 1])


# ### OWN_CAR_AGE

# In[24]:


cleaned_df.loc[:, 'OWN_CAR_AGE'] = user_df['OWN_CAR_AGE'].fillna(np.nan)


# ## 申请者工作情况
# 
# - DAYS_EMPLOYED
# - FLAG_EMP_PHONE
# - FLAG_WORK_PHONE
# - ORGANIZATION_TYPE
# - NAME_INCOME_TYPE
# - OCCUPATION_TYPE
# - AMT_INCOME_TOTAL

# ### DAYS_EMPLOYED

# In[25]:


cleaned_df.loc[:, 'DAYS_EMPLOYED'] = -(user_df['DAYS_EMPLOYED'].replace(365243, np.nan))


# ### FLAG_EMP_PHONE

# In[26]:


cleaned_df.loc[:, 'FLAG_EMP_PHONE'] = user_df['FLAG_EMP_PHONE']


# ### FLAG_WORK_PHONE

# In[27]:


cleaned_df.loc[:, 'FLAG_WORK_PHONE'] = user_df['FLAG_WORK_PHONE']


# ### ORGANIZATION_TYPE
# 
# 没有缺失值, 直接映射成数值.

# In[28]:


cleaned_df.loc[:, 'ORGANIZATION_TYPE'] = user_df['ORGANIZATION_TYPE'].replace(user_df['ORGANIZATION_TYPE'].unique(), range(len(user_df['ORGANIZATION_TYPE'].unique())))


# ### NAME_INCOME_TYPE
# 
# 没有缺失值, 直接映射成数值.

# In[29]:


cleaned_df.loc[:, 'NAME_INCOME_TYPE'] = user_df['NAME_INCOME_TYPE'].replace(user_df['NAME_INCOME_TYPE'].unique(), range(len(user_df['NAME_INCOME_TYPE'].unique())))


# ### AMT_INCOME_TOTAL

# In[30]:


cleaned_df.loc[:, 'AMT_INCOME_TOTAL'] = user_df['AMT_INCOME_TOTAL']


# ### OCCUPATION_TYPE

# In[31]:


occupationType = user_df.loc[user_df['OCCUPATION_TYPE'].notnull(), 'OCCUPATION_TYPE'].unique()


# In[32]:


cleaned_df.loc[:, 'OCCUPATION_TYPE'] = user_df['OCCUPATION_TYPE'].replace(occupationType, range(len(occupationType)))


# ## 申请者房产情况
# 
# - FLAG_OWN_REALTY
# - NAME_HOUSING_TYPE
# - REGION_POPULATION_RELATIVE
# - REGION_RATING_CLIENT
# - REGION_RATING_CLIENT_W_CITY
# - APARTMENTS_AVG
# - BASEMENTAREA_AVG
# - YEARS_BEGINEXPLUATATION_AVG
# - YEARS_BUILD_AVG
# - COMMONAREA_AVG
# - ELEVATORS_AVG
# - ENTRANCES_AVG
# - FLOORSMAX_AVG
# - FLOORSMIN_AVG
# - LANDAREA_AVG
# - LIVINGAPARTMENTS_AVG
# - LIVINGAREA_AVG
# - NONLIVINGAPARTMENTS_AVG
# - NONLIVINGAREA_AVG
# - APARTMENTS_MODE
# - BASEMENTAREA_MODE
# - YEARS_BEGINEXPLUATATION_MODE
# - YEARS_BUILD_MODE
# - COMMONAREA_MODE
# - ELEVATORS_MODE
# - ENTRANCES_MODE
# - FLOORSMAX_MODE
# - FLOORSMIN_MODE
# - LANDAREA_MODE
# - LIVINGAPARTMENTS_MODE
# - LIVINGAREA_MODE
# - NONLIVINGAPARTMENTS_MODE
# - NONLIVINGAREA_MODE
# - APARTMENTS_MEDI
# - BASEMENTAREA_MEDI
# - YEARS_BEGINEXPLUATATION_MEDI
# - YEARS_BUILD_MEDI
# - COMMONAREA_MEDI
# - ELEVATORS_MEDI
# - ENTRANCES_MEDI
# - FLOORSMAX_MEDI
# - FLOORSMIN_MEDI
# - LANDAREA_MEDI
# - LIVINGAPARTMENTS_MEDI
# - LIVINGAREA_MEDI
# - NONLIVINGAPARTMENTS_MEDI
# - NONLIVINGAREA_MEDI
# - FONDKAPREMONT_MODE
# - HOUSETYPE_MODE
# - TOTALAREA_MODE
# - WALLSMATERIAL_MODE
# - EMERGENCYSTATE_MODE

# In[33]:


cleaned_df.loc[:, 'FLAG_OWN_REALTY'] = user_df['FLAG_OWN_REALTY'].replace(['N', 'Y'], [0, 1])


# ### NAME_HOUSING_TYPE

# In[34]:


housingType = user_df.loc[user_df['NAME_HOUSING_TYPE'].notnull(), 'NAME_HOUSING_TYPE'].unique()
print(housingType)
cleaned_df.loc[:, 'NAME_HOUSING_TYPE'] = user_df['NAME_HOUSING_TYPE'].replace(housingType, range(len(housingType)))


# ### FONDKAPREMONT_MODE

# In[35]:


fondKapremont = user_df.loc[user_df['FONDKAPREMONT_MODE'].notnull(), 'FONDKAPREMONT_MODE'].unique()
print(fondKapremont)


# In[36]:


cleaned_df.loc[:, 'FONDKAPREMONT_MODE'] = user_df['FONDKAPREMONT_MODE'].replace(fondKapremont, range(len(fondKapremont))).fillna(np.nan)


# ### HOUSETYPE_MODE

# In[37]:


houseTypeMode = user_df.loc[user_df['HOUSETYPE_MODE'].notnull(), 'HOUSETYPE_MODE'].unique()
print(houseTypeMode)


# In[38]:


cleaned_df.loc[:, 'HOUSETYPE_MODE'] = user_df['HOUSETYPE_MODE'].replace(houseTypeMode, range(len(houseTypeMode))).fillna(np.nan)


# ### TOTALAREA_MODE

# In[39]:


cleaned_df.loc[:, 'TOTALAREA_MODE'] = user_df['TOTALAREA_MODE'].fillna(np.nan)


# ### WALLSMATERIAL_MODE

# In[40]:


wallsMaterialMode = user_df.loc[user_df['WALLSMATERIAL_MODE'].notnull(), 'WALLSMATERIAL_MODE'].unique()
print(wallsMaterialMode)


# In[41]:


cleaned_df.loc[:, 'WALLSMATERIAL_MODE'] = user_df['WALLSMATERIAL_MODE'].replace(wallsMaterialMode, range(len(wallsMaterialMode))).fillna(np.nan)


# ### EMERGENCYSTATE_MODE

# In[42]:


emergencyStateMode = user_df.loc[user_df['EMERGENCYSTATE_MODE'].notnull(), 'EMERGENCYSTATE_MODE'].unique()
print(emergencyStateMode)


# In[43]:


cleaned_df.loc[:, 'EMERGENCYSTATE_MODE'] = user_df['EMERGENCYSTATE_MODE'].replace(emergencyStateMode, range(len(emergencyStateMode))).fillna(np.nan)


# ### OTHERS
# 
# - REGION_POPULATION_RELATIVE
# - REGION_RATING_CLIENT
# - REGION_RATING_CLIENT_W_CITY
# - APARTMENTS_AVG
# - BASEMENTAREA_AVG
# - YEARS_BEGINEXPLUATATION_AVG
# - YEARS_BUILD_AVG
# - COMMONAREA_AVG
# - ELEVATORS_AVG
# - ENTRANCES_AVG
# - FLOORSMAX_AVG
# - FLOORSMIN_AVG
# - LANDAREA_AVG
# - LIVINGAPARTMENTS_AVG
# - LIVINGAREA_AVG
# - NONLIVINGAPARTMENTS_AVG
# - NONLIVINGAREA_AVG
# - APARTMENTS_MODE
# - BASEMENTAREA_MODE
# - YEARS_BEGINEXPLUATATION_MODE
# - YEARS_BUILD_MODE
# - COMMONAREA_MODE
# - ELEVATORS_MODE
# - ENTRANCES_MODE
# - FLOORSMAX_MODE
# - FLOORSMIN_MODE
# - LANDAREA_MODE
# - LIVINGAPARTMENTS_MODE
# - LIVINGAREA_MODE
# - NONLIVINGAPARTMENTS_MODE
# - NONLIVINGAREA_MODE
# - APARTMENTS_MEDI
# - BASEMENTAREA_MEDI
# - YEARS_BEGINEXPLUATATION_MEDI
# - YEARS_BUILD_MEDI
# - COMMONAREA_MEDI
# - ELEVATORS_MEDI
# - ENTRANCES_MEDI
# - FLOORSMAX_MEDI
# - FLOORSMIN_MEDI
# - LANDAREA_MEDI
# - LIVINGAPARTMENTS_MEDI
# - LIVINGAREA_MEDI
# - NONLIVINGAPARTMENTS_MEDI
# - NONLIVINGAREA_MEDI

# In[44]:


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


# In[45]:


for curFactor in realtyFactors:
    cleaned_df.loc[:, curFactor] = user_df[curFactor].fillna(np.nan)


# ## 风险评估
# 
# - REG_REGION_NOT_LIVE_REGION
# - REG_CITY_NOT_LIVE_CITY
# - REG_CITY_NOT_WORK_CITY
# - LIVE_CITY_NOT_WORK_CITY
# - DAYS_LAST_PHONE_CHANGE
# - REG_REGION_NOT_WORK_REGION
# - LIVE_REGION_NOT_WORK_REGION
# - EXT_SOURCE_1
# - EXT_SOURCE_2
# - EXT_SOURCE_3

# ### DAYS_LAST_PHONE_CHANGE

# In[46]:


cleaned_df.loc[:, 'DAYS_LAST_PHONE_CHANGE'] = -(user_df['DAYS_LAST_PHONE_CHANGE']).fillna(np.nan)


# ### OTHERS

# In[47]:


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


# In[48]:


for curFactor in riskFactors:
    cleaned_df.loc[:, curFactor] = user_df[curFactor].fillna(np.nan)


# ## 贷款申请材料
# 
# - NAME_CONTRACT_TYPE
# - AMT_CREDIT
# - AMT_ANNUITY
# - AMT_GOODS_PRICE
# - NAME_TYPE_SUITE
# - WEEKDAY_APPR_PROCESS_START
# - HOUR_APPR_PROCESS_START
# - AMT_REQ_CREDIT_BUREAU_HOUR
# - AMT_REQ_CREDIT_BUREAU_DAY
# - AMT_REQ_CREDIT_BUREAU_WEEK
# - AMT_REQ_CREDIT_BUREAU_MON
# - AMT_REQ_CREDIT_BUREAU_QRT
# - AMT_REQ_CREDIT_BUREAU_YEAR
# - FLAG_DOCUMENT_2
# - FLAG_DOCUMENT_3
# - FLAG_DOCUMENT_4
# - FLAG_DOCUMENT_5
# - FLAG_DOCUMENT_6
# - FLAG_DOCUMENT_7
# - FLAG_DOCUMENT_8
# - FLAG_DOCUMENT_9
# - FLAG_DOCUMENT_10
# - FLAG_DOCUMENT_11
# - FLAG_DOCUMENT_12
# - FLAG_DOCUMENT_13
# - FLAG_DOCUMENT_14
# - FLAG_DOCUMENT_15
# - FLAG_DOCUMENT_16
# - FLAG_DOCUMENT_17
# - FLAG_DOCUMENT_18
# - FLAG_DOCUMENT_19
# - FLAG_DOCUMENT_20
# - FLAG_DOCUMENT_21

# ### NAME_CONTRACT_TYPE

# In[49]:


contractType = user_df.loc[user_df['NAME_CONTRACT_TYPE'].notnull(), 'NAME_CONTRACT_TYPE'].unique()
print(contractType)


# In[50]:


cleaned_df.loc[:, 'NAME_CONTRACT_TYPE'] = user_df['NAME_CONTRACT_TYPE'].replace(contractType, range(len(contractType))).fillna(np.nan)


# ### NAME_TYPE_SUITE

# In[51]:


typeSuite = user_df.loc[user_df['NAME_TYPE_SUITE'].notnull(), 'NAME_TYPE_SUITE'].unique()
print(typeSuite)


# In[52]:


cleaned_df.loc[:, 'NAME_TYPE_SUITE'] = user_df['NAME_TYPE_SUITE'].replace(typeSuite, range(len(typeSuite))).fillna(np.nan)


# ### WEEKDAY_APPR_PROCESS_START

# In[53]:


weekday = [
    'MONDAY',
    'TUESDAY',
    'WEDNESDAY',
    'THURSDAY',
    'FRIDAY',
    'SATURDAY',
    'SUNDAY'
]


# In[54]:


cleaned_df.loc[:, 'WEEKDAY_APPR_PROCESS_START'] = user_df['WEEKDAY_APPR_PROCESS_START'].replace(weekday, range(len(weekday))).fillna(np.nan)


# ### HOUR_APPR_PROCESS_START

# In[55]:


cleaned_df.loc[:, 'HOUR_APPR_PROCESS_START'] = user_df['HOUR_APPR_PROCESS_START'].fillna(np.nan)


# ### OTHERS

# In[56]:


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


# In[57]:


for curFactor in appFactors:
    cleaned_df.loc[:, curFactor] = user_df[curFactor].fillna(np.nan)


# ## 申请者社交状况
# 
# - OBS_30_CNT_SOCIAL_CIRCLE
# - DEF_30_CNT_SOCIAL_CIRCLE
# - OBS_60_CNT_SOCIAL_CIRCLE
# - DEF_60_CNT_SOCIAL_CIRCLE

# In[59]:


socialFactors=[
    'OBS_30_CNT_SOCIAL_CIRCLE',
    'DEF_30_CNT_SOCIAL_CIRCLE',
    'OBS_60_CNT_SOCIAL_CIRCLE',
    'DEF_60_CNT_SOCIAL_CIRCLE'
]



for curFactor in socialFactors:
    cleaned_df.loc[:, curFactor] = user_df[curFactor].fillna(np.nan)

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--data-dir', type=str,
                           default='../../data',
                           help='Directory for storing input data')
    argParser.add_argument('--app-train', type=str,
                           default='application_train.csv',
                           help='File name for application_train')
    argParser.add_argument('--app-test', type=str,
                           default='application_test.csv',
                           help='File name for application_test')
    argParser.add_argument('--bureau', type=str,
                           default='bereau.csv',
                           help='File name for bereau')
    argParser.add_argument('--bureau-balance', type=str,
                           default='bureau_balance.csv',
                           help='File name for bereau_balance')
    argParser.add_argument('--credit-card-balance', type=str,
                           default='credit_card_balance.csv',
                           help='File name for credit_card_balance')
    argParser.add_argument('--installments-payments', type=str,
                           default='installments_payments.csv',
                           help='File name for installments_payments')
    argParser.add_argument('--POS-CACHE-balance', type=str,
                           default='POS_CACHE_balance.csv',
                           help='File name for POS_CACHE_balance')
    argParser.add_argument('--previous-application', type=str,
                           default='previous_application.csv',
                           help='File name for previous_application')
    argParser.add_argument('--output-dir', type=str,
                           default='../../data/output',
                           help='Directory for storing output data')
    FLAGS, _ = argParser.parse_known_args()