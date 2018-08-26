# -*- coding=utf-8 -*-
import pandas as pd

test_df = pd.read_csv('../../data/application_test.csv')
result_df = pd.DataFrame()
result_df['SK_ID_CURR'] = test_df['SK_ID_CURR']
result_df['TARGET'] = 0.5
result_df.to_csv('../../data/output/submission_0.csv', index=False)