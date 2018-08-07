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
        cleaned_app_train_df = pd.DataFrame()
        cleaned_app_test_df = pd.DataFrame()
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
        test_data = test_df.values[:, 2:]
        np.savez(test_np_filename, user_id=test_user_id, data=test_data)
        logging.debug("done!")

        logging.debug("train array %d*%d"%tuple(train_data.shape))
        logging.debug("test array %d*%d"%tuple(test_data.shape))

    def clean_app_info(self, app_train_df, cleaned_train_df, app_test_df, cleaned_test_df):
        assert isinstance(app_train_df, pd.DataFrame) and isinstance(app_test_df, pd.DataFrame) \
        and isinstance(cleaned_train_df, pd.DataFrame) and isinstance(cleaned_test_df, pd.DataFrame)
        logging.debug("clean_app_info begin...")

        train_user_id = app_train_df['SK_ID_CURR']
        test_user_id = app_test_df['SK_ID_CURR']

        merged_df = app_train_df.append(app_test_df, ignore_index=True, sort=False)
        merged_df.loc[:, 'NAME_FAMILY_STATUS'] = merged_df.loc[:, 'NAME_FAMILY_STATUS'].replace('Unknown', np.nan)
        merged_df = merged_df.replace('XNA', np.nan)
        merged_df = pd.get_dummies(merged_df, dtype=np.float64)

        cleaned_train_df = merged_df.loc[merged_df['SK_ID_CURR'].isin(train_user_id), :]
        cleaned_test_df = merged_df.loc[merged_df['SK_ID_CURR'].isin(test_user_id), :]

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
