# -*- coding:utf-8 -*-
import os
import json
import shutil
import hashlib
import pandas as pd
import xgboost as xgb
import argparse
import logging


def load_dataset_from_pandas(data_dir, output_dir, train_data, validate_data):
    logging.debug("Loading %s..." % (train_data,))
    train_df = pd.read_csv(os.path.join(data_dir, output_dir, train_data))
    logging.debug("DataFrame for training size %d*%d. Loading DataFrame done!" % tuple(train_df.shape))

    logging.debug("Loading %s..." % (validate_data,))
    validate_df = pd.read_csv(os.path.join(data_dir, output_dir, validate_data))
    logging.debug("DataFrame for testing size %d*%d. Loading DataFrame done!" % tuple(validate_df.shape))

    # prepare data for xgboost
    logging.debug("Preparing data for training...")
    train_data = xgb.DMatrix(train_df.drop(['SK_ID_CURR', 'TARGET'], axis=1), label=train_df['TARGET'])
    train_data.save_binary(os.path.join(FLAGS.data_path, FLAGS.output_dir, FLAGS.train_buffer))
    logging.debug("Done!")

    logging.debug("Preparing data for testing...")
    validate_data = xgb.DMatrix(validate_df.drop(['SK_ID_CURR', 'TARGET'], axis=1), label=validate_df['TARGET'])
    validate_data.save_binary(os.path.join(FLAGS.data_path, FLAGS.output_dir, FLAGS.validate_buffer))
    logging.debug("Done!")

    return train_data, validate_data

def load_dataset_from_xgboost(data_dir, output_dir, train_buffer, validate_buffer):
    train_data = xgb.DMatrix(os.path.join(data_dir, output_dir, train_buffer))
    validate_data = xgb.DMatrix(os.path.join(data_dir, output_dir, validate_buffer))
    return train_data, validate_data

def load_parameters_from_json(data_path, input_dir, json_filename):
    file_path = os.path.join(data_path, input_dir, json_filename)
    with open(file_path, 'r') as json_file:
        param = json.load(json_file)

    with open(file_path, 'rb') as json_file:
        json_md5_obj = hashlib.md5()
        json_md5_obj.update(json_file.read())
        json_md5_str = json_md5_obj.hexdigest()

    archive_json_filename = json_md5_str+'.json'
    if not os.path.exists(os.path.join(data_path, input_dir, archive_json_filename)):
        os.chdir(os.path.join(data_path, input_dir))
        shutil.copyfile(json_filename, archive_json_filename)
    return param, json_md5_str

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path',
                            type=str,
                            default='../../data',
                            help='folder for data')
    arg_parser.add_argument('--input_dir',
                            type=str,
                            default='input',
                            help='sub folder for input')
    arg_parser.add_argument('--model_dir',
                            type=str,
                            default='models/xgboost',
                            help='sub folder for model')
    arg_parser.add_argument('--output_dir',
                            type=str,
                            default='output',
                            help='sub folder for output')
    arg_parser.add_argument('--train_data',
                            type=str,
                            default='splited_train.csv',
                            help='numpy file for training dataset')
    arg_parser.add_argument('--validate_data',
                            type=str,
                            default='splited_validate.csv',
                            help='numpy file for validation dataset')
    arg_parser.add_argument('--train_buffer',
                            type=str,
                            default='splited_train.buffer',
                            help='numpy file for training dataset')
    arg_parser.add_argument('--validate_buffer',
                            type=str,
                            default='splited_validate.buffer',
                            help='numpy file for validation dataset')
    arg_parser.add_argument('--param_file',
                            type=str,
                            default='param.json',
                            help='parameters for training xgb model stored in json formation')

    FLAGS, _ = arg_parser.parse_known_args()

    if os.path.exists(os.path.join(FLAGS.data_path, FLAGS.output_dir, FLAGS.train_buffer)) and \
            os.path.join(os.path.join(FLAGS.data_path, FLAGS.output_dir, FLAGS.validate_buffer)):
        train_data, validate_data = load_dataset_from_xgboost(FLAGS.data_path, FLAGS.output_dir, FLAGS.train_buffer, FLAGS.validate_buffer)
    else:
        train_data, validate_data = load_dataset_from_pandas(FLAGS.data_path, FLAGS.output_dir, FLAGS.train_data, FLAGS.validate_data)

    param, param_md5_str = load_parameters_from_json(FLAGS.data_path, FLAGS.input_dir, FLAGS.param_file)
    eval_list = [(validate_data, 'eval')]
    num_round = 150

    bst = xgb.train(param, train_data, num_round, eval_list)

    model_path = os.path.join(FLAGS.data_path, FLAGS.model_dir, param_md5_str+'.model')
    model_raw_path = os.path.join(FLAGS.data_path, FLAGS.model_dir, param_md5_str+'.model.raw.txt')
    feature_raw_path = os.path.join(FLAGS.data_path, FLAGS.model_dir, param_md5_str+'.feature.raw.txt')
    logging.debug(model_path)
    logging.debug(model_raw_path)
    logging.debug(feature_raw_path)

    bst.save_model(model_path)
    bst.dump_model(model_raw_path)
