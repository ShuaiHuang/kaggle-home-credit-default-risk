# -*- coding:utf-8 -*-
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
        app_train_df = pd.read_csv(os.path.join(self.__data_dir, self.__app_train))
        app_test_df = pd.read_csv(os.path.join(self.__data_dir, self.__app_test))
        cleaned_app_train_df = app_train_df.loc[:, ['SK_ID_CURR', 'TARGET']].copy().astype(np.float64)
        cleaned_app_test_df = pd.DataFrame(app_test_df.loc[:, 'SK_ID_CURR'])
        cleaned_app_train_df, cleaned_app_test_df = \
            self.clean_app_info(app_train_df, cleaned_app_train_df, app_test_df, cleaned_app_test_df)
        self.save_cleaned_df(cleaned_app_train_df, cleaned_app_test_df)

    def save_cleaned_df(self, train_df, test_df):
        """
        :param train_df: cleaned DataFrame for training dataset
        :param test_df: cleaned DataFrame for test dataset
        :return: None
        """
        assert isinstance(train_df, pd.DataFrame) and isinstance(test_df, pd.DataFrame)
        train_data_frame_filename = os.path.join(self.__output_dir, "cleaned_train.csv")
        logging.debug("saving %s"%(train_data_frame_filename,))
        train_df.to_csv(train_data_frame_filename)
        logging.debug("done!")

        test_data_frame_filename = os.path.join(self.__output_dir, "cleaned_test.csv")
        logging.debug("saving %s"%(test_data_frame_filename,))
        test_df.to_csv(test_data_frame_filename)
        logging.debug("done!")

        train_np_filename = os.path.join(self.__output_dir, "cleaned_train.npz")
        logging.debug("saving %s" % (train_np_filename,))
        train_user_id = train_df.values[:, 0]
        train_label = train_df.values[:, 1]
        train_data = train_df.values[:, 2:]
        np.savez(train_np_filename, user_id=train_user_id, label=train_label, data=train_data)
        logging.debug("done!")

        test_np_filename = os.path.join(self.__output_dir, "cleaned_test.npz")
        logging.debug("saving %s"%(test_np_filename,))
        test_user_id = test_df.values[:, 0]
        test_data = test_df.values[:, 1:]
        np.savez(test_np_filename, user_id=test_user_id, data=test_data)
        logging.debug("done!")

        logging.debug("train array %d*%d"%tuple(train_data.shape))
        logging.debug("test array %d*%d"%tuple(test_data.shape))

    def discretize_column(self, train_column, test_column, replace_nan=False):
        assert isinstance(train_column, pd.Series), "Invalid column type: %s"%(type(train_column),) \
                and isinstance(test_column, pd.Series, "Invalid column type: %s"%(type(test_column),))
        str_values = train_column.loc[train_column.notnull()].unique()
        numeric_values = list(range(len(str_values)))
        train_column = train_column.replace(str_values, numeric_values).astype(np.float32)
        test_column = test_column.replace(str_values, numeric_values).astype(np.float32)
        if replace_nan:
            train_column = train_column.fillna(-1)
            test_column = test_column.fillna(-1)
        train_column = train_column.astype(np.float64)
        test_column = test_column.astype(np.float64)
        logging.debug("[%s, %s] %s -> %s"%(train_column.name, test_column.name, str_values, numeric_values,))
        logging.debug("[%s, %s] replace_nan: %r"%(train_column.name, test_column.name, replace_nan,))
        return train_column, test_column

    def clean_app_info(self, app_train_df, cleaned_train_df, app_test_df, cleaned_test_df):
        assert isinstance(app_train_df, pd.DataFrame) and isinstance(app_test_df, pd.DataFrame) \
        and isinstance(cleaned_train_df, pd.DataFrame) and isinstance(cleaned_test_df, pd.DataFrame)
        logging.debug("clean_app_info begin...")

        cleaned_train_df.loc[:, 'CODE_GENDER'], cleaned_test_df.loc[:, 'CODE_GENDER'] = self.discretize_column(
            app_train_df['CODE_GENDER'].replace(['XNA'], [np.nan]),
            app_test_df['CODE_GENDER'].replace(['XNA'], [np.nan]))
        cleaned_train_df.loc[:, 'DAYS_BIRTH'] = -app_train_df['DAYS_BIRTH'].astype(np.float64)
        cleaned_test_df.loc[:, 'DAYS_BIRTH'] = -app_test_df['DAYS_BIRTH'].astype(np.float64)
        cleaned_train_df.loc[:, 'DAYS_REGISTRATION'] = -app_train_df['DAYS_REGISTRATION'].astype(np.float64)
        cleaned_test_df.loc[:, 'DAYS_REGISTRATION'] = -app_test_df['DAYS_REGISTRATION'].astype(np.float64)
        cleaned_train_df.loc[:, 'DAYS_ID_PUBLISH'] = -app_train_df['DAYS_ID_PUBLISH'].astype(np.float64)
        cleaned_test_df.loc[:, 'DAYS_ID_PUBLISH'] = -app_test_df['DAYS_ID_PUBLISH'].astype(np.float64)
        cleaned_train_df.loc[:, 'NAME_EDUCATION_TYPE'], cleaned_test_df.loc[:, 'NAME_EDUCATION_TYPE'] = \
            self.discretize_column(app_train_df['NAME_EDUCATION_TYPE'], app_test_df['NAME_EDUCATION_TYPE'])
        cleaned_train_df.loc[:, 'CNT_CHILDREN'] = app_train_df['CNT_CHILDREN'].astype(np.float64)
        cleaned_test_df.loc[:, 'CNT_CHILDREN'] = app_test_df['CNT_CHILDREN'].astype(np.float64)
        cleaned_train_df.loc[:, 'CNT_FAM_MEMBERS'] = app_train_df['CNT_FAM_MEMBERS'].fillna(np.nan).astype(np.float64)
        cleaned_test_df.loc[:, 'CNT_FAM_MEMBERS'] = app_test_df['CNT_FAM_MEMBERS'].fillna(np.nan).astype(np.float64)
        cleaned_train_df.loc[:, 'NAME_FAMILY_STATUS'], cleaned_test_df.loc[:, 'NAME_FAMILY_STATUS'] = \
            self.discretize_column(app_train_df['NAME_FAMILY_STATUS'], app_test_df['NAME_FAMILY_STATUS'])
        cleaned_train_df.loc[:, 'FLAG_MOBIL'] = app_train_df['FLAG_MOBIL'].astype(np.float64)
        cleaned_test_df.loc[:, 'FLAG_MOBIL'] = app_test_df['FLAG_MOBIL'].astype(np.float64)
        cleaned_train_df.loc[:, 'FLAG_PHONE'] = app_train_df['FLAG_PHONE'].astype(np.float64)
        cleaned_test_df.loc[:, 'FLAG_PHONE'] = app_test_df['FLAG_PHONE'].astype(np.float64)
        cleaned_train_df.loc[:, 'FLAG_EMAIL'] = app_train_df['FLAG_EMAIL'].astype(np.float64)
        cleaned_test_df.loc[:, 'FLAG_EMAIL'] = app_test_df['FLAG_EMAIL'].astype(np.float64)
        cleaned_train_df.loc[:, 'FLAG_CONT_MOBILE'] = app_train_df['FLAG_CONT_MOBILE'].astype(np.float64)
        cleaned_test_df.loc[:, 'FLAG_CONT_MOBILE'] = app_test_df['FLAG_CONT_MOBILE'].astype(np.float64)

        cleaned_train_df.loc[:, 'FLAG_OWN_CAR'], cleaned_test_df.loc[:, 'FLAG_OWN_CAR'] = \
            self.discretize_column(app_train_df['FLAG_OWN_CAR'], app_test_df['FLAG_OWN_CAR'])
        cleaned_train_df.loc[:, 'OWN_CAR_AGE'] = app_train_df['OWN_CAR_AGE'].astype(np.float64)
        cleaned_test_df.loc[:, 'OWN_CAR_AGE'] = app_test_df['OWN_CAR_AGE'].astype(np.float64)

        cleaned_train_df.loc[:, 'DAYS_EMPLOYED'] = -(app_train_df['DAYS_EMPLOYED'].replace(365243, np.nan)).astype(
            np.float64)
        cleaned_test_df.loc[:, 'DAYS_EMPLOYED'] = -(app_test_df['DAYS_EMPLOYED'].replace(365243, np.nan)).astype(
            np.float64)
        cleaned_train_df.loc[:, 'FLAG_EMP_PHONE'] = app_train_df['FLAG_EMP_PHONE'].astype(np.float64)
        cleaned_test_df.loc[:, 'FLAG_EMP_PHONE'] = app_test_df['FLAG_EMP_PHONE'].astype(np.float64)
        cleaned_train_df.loc[:, 'FLAG_WORK_PHONE'] = app_train_df['FLAG_WORK_PHONE'].astype(np.float64)
        cleaned_test_df.loc[:, 'FLAG_WORK_PHONE'] = app_test_df['FLAG_WORK_PHONE'].astype(np.float64)
        cleaned_train_df.loc[:, 'ORGANIZATION_TYPE'], cleaned_test_df.loc[:, 'ORGANIZATION_TYPE'] = \
            self.discretize_column(app_train_df['ORGANIZATION_TYPE'].replace(['XNA'], [np.nan]),
                                   app_test_df['ORGANIZATION_TYPE'].replace(['XNA'], [np.nan]))
        cleaned_train_df.loc[:, 'NAME_INCOME_TYPE'], cleaned_test_df.loc[:, 'NAME_INCOME_TYPE'] = \
            self.discretize_column(app_train_df['NAME_INCOME_TYPE'], app_test_df['NAME_INCOME_TYPE'])
        cleaned_train_df.loc[:, 'AMT_INCOME_TOTAL'] = app_train_df['AMT_INCOME_TOTAL'].astype(np.float64)
        cleaned_test_df.loc[:, 'AMT_INCOME_TOTAL'] = app_test_df['AMT_INCOME_TOTAL'].astype(np.float64)
        cleaned_train_df.loc[:, 'OCCUPATION_TYPE'], cleaned_test_df.loc[:, 'OCCUPATION_TYPE'] = \
            self.discretize_column(app_train_df['OCCUPATION_TYPE'], app_test_df['OCCUPATION_TYPE'])

        cleaned_train_df.loc[:, 'FLAG_OWN_REALTY'], cleaned_test_df.loc[:, 'FLAG_OWN_REALTY'] = \
            self.discretize_column(app_train_df['FLAG_OWN_REALTY'], app_test_df['FLAG_OWN_REALTY'])
        cleaned_train_df.loc[:, 'NAME_HOUSING_TYPE'], cleaned_test_df.loc[:, 'NAME_HOUSING_TYPE'] = \
            self.discretize_column(app_train_df['NAME_HOUSING_TYPE'], app_test_df['NAME_HOUSING_TYPE'])
        cleaned_train_df.loc[:, 'FONDKAPREMONT_MODE'], cleaned_test_df.loc[:, 'FONDKAPREMONT_MODE'] = \
            self.discretize_column(app_train_df['FONDKAPREMONT_MODE'], app_test_df['FONDKAPREMONT_MODE'])
        cleaned_train_df.loc[:, 'HOUSETYPE_MODE'], cleaned_test_df.loc[:, 'HOUSETYPE_MODE'] = \
            self.discretize_column(app_train_df['HOUSETYPE_MODE'], app_test_df['HOUSETYPE_MODE'])
        cleaned_train_df.loc[:, 'TOTALAREA_MODE'] = app_train_df['TOTALAREA_MODE'].fillna(np.nan).astype(np.float64)
        cleaned_test_df.loc[:, 'TOTALAREA_MODE'] = app_test_df['TOTALAREA_MODE'].fillna(np.nan).astype(np.float64)
        cleaned_train_df.loc[:, 'WALLSMATERIAL_MODE'], cleaned_test_df.loc[:, 'WALLSMATERIAL_MODE'] = \
            self.discretize_column(app_train_df['WALLSMATERIAL_MODE'], app_test_df['WALLSMATERIAL_MODE'])
        cleaned_train_df.loc[:, 'EMERGENCYSTATE_MODE'], cleaned_test_df.loc[:, 'EMERGENCYSTATE_MODE'] = \
            self.discretize_column(app_train_df['EMERGENCYSTATE_MODE'], app_test_df['EMERGENCYSTATE_MODE'])

        realty_factors = [
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
        for factor in realty_factors:
            cleaned_train_df.loc[:, factor] = app_train_df[factor].fillna(np.nan).astype(np.float64)
            cleaned_test_df.loc[:, factor] = app_test_df[factor].fillna(np.nan).astype(np.float64)

        cleaned_train_df.loc[:, 'DAYS_LAST_PHONE_CHANGE'] = -(app_train_df['DAYS_LAST_PHONE_CHANGE']).fillna(
            np.nan).astype(np.float64)
        cleaned_test_df.loc[:, 'DAYS_LAST_PHONE_CHANGE'] = -(app_test_df['DAYS_LAST_PHONE_CHANGE']).fillna(
            np.nan).astype(np.float64)

        risk_factors = [
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
        for factor in risk_factors:
            cleaned_train_df.loc[:, factor] = app_train_df[factor].fillna(np.nan).astype(np.float64)
            cleaned_test_df.loc[:, factor] = app_test_df[factor].fillna(np.nan).astype(np.float64)

        cleaned_train_df.loc[:, 'NAME_CONTRACT_TYPE'], cleaned_test_df.loc[:, 'NAME_CONTRACT_TYPE'] = \
            self.discretize_column(app_train_df['NAME_CONTRACT_TYPE'], app_test_df['NAME_CONTRACT_TYPE'])
        cleaned_train_df.loc[:, 'NAME_TYPE_SUITE'], cleaned_test_df.loc[:, 'NAME_TYPE_SUITE'] = \
            self.discretize_column(app_train_df['NAME_TYPE_SUITE'], app_test_df['NAME_TYPE_SUITE'])

        weekday = [
            'MONDAY',
            'TUESDAY',
            'WEDNESDAY',
            'THURSDAY',
            'FRIDAY',
            'SATURDAY',
            'SUNDAY'
        ]
        cleaned_train_df.loc[:, 'WEEKDAY_APPR_PROCESS_START'] = app_train_df['WEEKDAY_APPR_PROCESS_START'].replace(
            weekday, range(len(weekday))).fillna(np.nan).astype(np.float64)
        cleaned_test_df.loc[:, 'WEEKDAY_APPR_PROCESS_START'] = app_test_df['WEEKDAY_APPR_PROCESS_START'].replace(
            weekday, range(len(weekday))).fillna(np.nan).astype(np.float64)

        cleaned_train_df.loc[:, 'HOUR_APPR_PROCESS_START'] = app_train_df['HOUR_APPR_PROCESS_START'].fillna(np.nan).astype(np.float64)
        cleaned_test_df.loc[:, 'HOUR_APPR_PROCESS_START'] = app_test_df['HOUR_APPR_PROCESS_START'].fillna(np.nan).astype(np.float64)

        app_factors = [
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
        for factor in app_factors:
            cleaned_train_df.loc[:, factor] = app_train_df[factor].fillna(np.nan).astype(np.float64)
            cleaned_test_df.loc[:, factor] = app_test_df[factor].fillna(np.nan).astype(np.float64)

        social_factors = [
            'OBS_30_CNT_SOCIAL_CIRCLE',
            'DEF_30_CNT_SOCIAL_CIRCLE',
            'OBS_60_CNT_SOCIAL_CIRCLE',
            'DEF_60_CNT_SOCIAL_CIRCLE'
        ]

        for factor in social_factors:
            cleaned_train_df.loc[:, factor] = app_train_df[factor].fillna(np.nan).astype(np.float64)
            cleaned_test_df.loc[:, factor] = app_test_df[factor].fillna(np.nan).astype(np.float64)

        logging.debug("clean_app_info done!")
        return cleaned_train_df, cleaned_test_df

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

    data_cleaner = DataCleaner(FLAGS.data_dir,
                               FLAGS.app_train,
                               FLAGS.app_test,
                               FLAGS.bureau,
                               FLAGS.bureau_balance,
                               FLAGS.credit_card_balance,
                               FLAGS.installments_payments,
                               FLAGS.POS_CACHE_balance,
                               FLAGS.previous_application,
                               FLAGS.output_dir)
    data_cleaner.clean_data()
