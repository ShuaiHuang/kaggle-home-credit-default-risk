import os
import pandas as pd
import xgboost as xgb
import argparse
import logging


def load_dataset(data_dir, train_data, test_data):
    logging.debug("Loading %s..." % (train_data,))
    train_df = pd.read_csv(os.path.join(data_dir, train_data))
    logging.debug("DataFrame for training size %d*%d. Loading DataFrame done!" % tuple(train_df.shape))

    logging.debug("Loading %s..." % (test_data,))
    test_df = pd.read_csv(os.path.join(data_dir, test_data))
    logging.debug("DataFrame for testing size %d*%d. Loading DataFrame done!" % tuple(test_df.shape))
    return train_df, test_df

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_dir',
                            type=str,
                            default='../../data/output',
                            help='folder for data')
    arg_parser.add_argument('--train_data',
                            type=str,
                            default='cleaned_train.csv',
                            help='numpy file for training dataset')
    arg_parser.add_argument('--test_data',
                            type=str,
                            default='cleaned_test.csv',
                            help='numpy file for validation dataset')
    arg_parser.add_argument('--model_dir',
                            type=str,
                            default='./models/xgboost')
    FLAGS, _ = arg_parser.parse_known_args()

    train_df, test_df = load_dataset(FLAGS.data_dir, FLAGS.train_data, FLAGS.test_data)

    # prepare data for xgboost
    logging.debug("Preparing data for training...")
    train_data = xgb.DMatrix(train_df.drop(['SK_ID_CURR', 'TARGET'], axis=1), label=train_df['TARGET'])
    logging.debug("Done!")

    logging.debug("Preparing data for testing...")
    test_data = xgb.DMatrix(test_df.drop(['SK_ID_CURR', 'TARGET'], axis=1))
    logging.debug("Done!")

    xgb_params = {
        'booster': 'gbtree',
        'silent': 0,
        'nthread': 12,
        'eta': 0.3,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': 43
    }
    xgb.cv(xgb_params, train_data, nfold=10, stratified=True)
