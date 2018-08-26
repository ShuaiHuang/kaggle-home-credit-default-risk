# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
import logging
import argparse
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_dir', type=str,
                            default='../../data/output',
                            help='default data dir')
    arg_parser.add_argument('--train_file', type=str,
                            default='cleaned_train.csv',
                            help='cleaned train numpy file')
    arg_parser.add_argument('--test_file', type=str,
                            default='cleaned_test.csv',
                            help='cleaned test numpy file')
    FLAGS, _ = arg_parser.parse_known_args()

    train_df = pd.read_csv(os.path.join(FLAGS.data_dir, FLAGS.train_file))

    splited_train_data, splited_validate_data =\
        train_test_split(train_df, test_size=0.01, random_state=43)

    splited_train_file = os.path.join(FLAGS.data_dir, "splited_train.csv")
    logging.debug("saving %s"%(splited_train_file,))
    splited_train_data.to_csv(splited_train_file)
    logging.debug("done!")

    splited_validate_file = os.path.join(FLAGS.data_dir, "splited_validate.csv")
    logging.debug("saving %s"%(splited_validate_file,))
    splited_validate_data.to_csv(splited_validate_file)
    logging.debug("done!")

    logging.debug("splited_train_data size: %d*%d, splited_validate_data size:%d*%d"%(splited_train_data.shape + splited_validate_data.shape))