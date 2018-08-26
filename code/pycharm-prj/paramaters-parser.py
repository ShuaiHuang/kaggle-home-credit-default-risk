# -*- coding:utf-8 -*-
import os
import json
import logging
import argparse
import pandas as pd
import numpy as np


class ParameterParser(object):
    def ParameterParser(self):
        pass

    def get_data_frame_for_grid_search(self, raw_dict):
        self.__numeric_dict = self.__get_numeric_dict(raw_dict)
        logging.debug(self.__numeric_dict)

        self.__param_names = list(self.__numeric_dict.keys())
        self.__param_data_frame = pd.DataFrame(columns=self.__param_names)

        self.__get_cartesian_product(0, [])
        logging.debug(self.__param_data_frame)
        self.__param_data_frame.to_csv(os.path.join('../../data', 'output'))


    def __get_cartesian_product(self, element_count, param_content):
        if element_count == len(self.__param_names):
            self.__param_data_frame.loc[self.__param_data_frame.shape[0]] = param_content
            logging.debug(param_content)
        else:
            param_list = self.__numeric_dict.get(self.__param_names[element_count])
            for param_element in param_list:
                param_content.append(param_element)
                self.__get_cartesian_product(element_count+1, param_content)
                param_content.pop(element_count)



    def __get_numeric_dict(self, raw_dict):
        numeric_dict = dict()
        for category_element in raw_dict.items():
            category_element_name = category_element[0]
            category_element_dict = category_element[1]
            for param_element in category_element_dict.items():
                param_element_name = param_element[0]
                param_element_dict = param_element[1]
                ret = self.__convert_dict_to_list(param_element)
                numeric_dict['%s_%s'%(category_element_name, param_element_name)] = ret
        return numeric_dict

    def __convert_dict_to_list(self, element_dict):
        param_element_dict = element_dict[1]
        if 'values' in param_element_dict:
            return param_element_dict.get('values')
        else:
            min_value = param_element_dict.get('min_value')
            max_value = param_element_dict.get('max_value')
            scale_type = param_element_dict.get('scale_type')
            exp_base = param_element_dict.get("exp_base")
            min_index = param_element_dict.get("min_index")
            max_index = param_element_dict.get("max_index")
            step_size = param_element_dict.get('step_size')

            if min_value == max_value:
                return [min_value]
            elif min_value < max_value:
                if scale_type == 'exp':
                    exp_list = [np.power(exp_base, index) for index in range(min_index, max_index+1) if np.power(exp_base, index) < max_value]
                    if min_value < exp_list[0]:
                        exp_list.insert(0, min_value)
                    exp_list.append(max_value)
                    return exp_list
                elif scale_type == 'linear':
                    return list(np.arange(min_value, max_value + step_size, step_size))
                else:
                    logging.error("scale_type=%s"%(scale_type,))
            else:
                logging.error("min_value=%d max_value=%d"%(min_value, max_value))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--file_path',
                            type=str,
                            default='./param-list.json',
                            help='json file storing parameters')
    FLAGS, _ = arg_parser.parse_known_args()

    with open(FLAGS.file_path, 'r') as json_file:
        param_dict_raw = json.load(json_file)
    param_parser = ParameterParser()
    param_parser.get_data_frame_for_grid_search(param_dict_raw)