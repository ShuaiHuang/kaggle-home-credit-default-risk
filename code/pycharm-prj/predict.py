# -*- coding:utf-8 -*-
import os
import logging
import argparse
import pandas as pd
import xgboost as xgb

def load_data(data_path, output_dir, test_data_filename, test_buffer_filename):
    csv_file_path = os.path.join(data_path, output_dir, test_data_filename)
    buffer_file_path = os.path.join(data_path, output_dir, test_buffer_filename)

    test_df = pd.read_csv(csv_file_path)
    if not os.path.exists(buffer_file_path):
        test_buffer = xgb.DMatrix(test_df.drop(['SK_ID_CURR', 'TARGET'], axis=1))
        test_buffer.save_binary(buffer_file_path)
    else:
        test_buffer = xgb.DMatrix(buffer_file_path)

    return test_buffer, test_df

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path',
                            type=str,
                            default='../../data',
                            help='folder for data')
    arg_parser.add_argument('--model_dir',
                            type=str,
                            default='models/xgboost',
                            help='sub folder for model')
    arg_parser.add_argument('--model_filename',
                            type=str,
                            default='9ddeef7a8c17a981413da402bafd6dc1.model',
                            help='xgb model filename')
    arg_parser.add_argument('--output_dir',
                            type=str,
                            default='output',
                            help='sub folder for output')
    arg_parser.add_argument('--test_data',
                            type=str,
                            default='cleaned_test.csv',
                            help='csv file for testing dataset')
    arg_parser.add_argument('--test_buffer',
                            type=str,
                            default='test.buffer',
                            help='DMatrix file for testing dataset')

    FLAGS, _ = arg_parser.parse_known_args()

    test_buffer, test_df = load_data(FLAGS.data_path, FLAGS.output_dir, FLAGS.test_data, FLAGS.test_buffer)

    model_path = os.path.join(FLAGS.data_path, FLAGS.model_dir, FLAGS.model_filename)
    bst = xgb.Booster({'nthread': 12})
    bst.load_model(model_path)

    prediction_result = bst.predict(test_buffer)

    test_df.loc[:, 'TARGET'] = prediction_result

    (submission_filename, _) = os.path.splitext(FLAGS.model_filename)
    submission_filename = 'submission_' + submission_filename + '.csv'
    submission_path = os.path.join(FLAGS.data_path, FLAGS.output_dir, submission_filename)
    logging.debug(submission_path)
    test_df.to_csv(submission_path, columns=['SK_ID_CURR', 'TARGET'], index=False)