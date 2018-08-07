# -*- coding:utf-8 -*-
import os
import numpy as np
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
                            default='cleaned_train.npz',
                            help='cleaned train numpy file')
    arg_parser.add_argument('--test_file', type=str,
                            default='cleaned_test.npz',
                            help='cleaned test numpy file')
    FLAGS, _ = arg_parser.parse_known_args()

    train_file = np.load(os.path.join(FLAGS.data_dir, FLAGS.train_file))
    train_data = train_file['data']
    train_label = train_file['label']
    train_user_id = train_file['user_id']

    splited_train_data, splited_validate_data, splited_train_user_id, splited_validate_user_id,\
    splited_train_label, splited_validate_label =\
        train_test_split(train_data, train_user_id, train_user_id, test_size=0.01, random_state=43)

    splited_train_file = os.path.join(FLAGS.data_dir, "splited_train.npz")
    logging.debug("saving %s"%(splited_train_file,))
    np.savez(splited_train_file, data=splited_train_data, user_id=splited_train_user_id,
             label=splited_train_label)
    logging.debug("done!")

    splited_validate_file = os.path.join(FLAGS.data_dir, "splited_validate.npz")
    logging.debug("saving %s"%(splited_validate_file,))
    np.savez(splited_validate_file, data=splited_validate_data, user_id=splited_validate_user_id,
             label=splited_validate_label)
    logging.debug("done!")

    logging.debug("splited_train_data size: %d*%d, splited_validate_data size:%d*%d"%(splited_train_data.shape + splited_validate_data.shape))
    logging.debug("splited_train_label size: %d, splited_validate_label size: %d"%(splited_train_label.shape + splited_validate_label.shape))
    logging.debug("splited_train_user_id size: %d, splited_validate_user_id size: %d"%(splited_train_user_id.shape + splited_validate_user_id.shape))