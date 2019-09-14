import argparse

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import lightgbm as lgb
from sklearn.metrics import f1_score
import numpy as np

CATEGORY = ['ecfg', 'flbmk', 'flg_3dsmk', 'insfg', 'ovrlt', 'scity', 'csmcu', 'cano', 'mchno', 'hcefg', 'bacno', 'contp', 'etymd', 'acqic']


def lgb_f1_score(y_true, y_pred):
    y_hat = np.round(y_pred)
    return 'f1', f1_score(y_true, y_hat), True


def main(args):
    df_train = pd.read_csv(args.train_file)

    for cat in CATEGORY:
        df_train[cat] = df_train[cat].astype('category')#.cat.codes

    y_train = df_train['fraud_ind']
    x_train = df_train.drop('fraud_ind', axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

    estimator = lgb.LGBMClassifier(num_leaves=31)

    param_grid = {
            'learning_rate': [0.1],
            'n_estimators': [70],
            'scale_pos_weight': [3, 5]
            }

    gbm = GridSearchCV(estimator, param_grid, scoring='f1')
    gbm.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric=lgb_f1_score, early_stopping_rounds=5, categorical_feature='auto')

    df_test = pd.read_csv(args.test_file)
    for cat in CATEGORY:
        df_test[cat] = df_test[cat].astype('category')

    
    result = gbm.predict(df_test)
    df_label = pd.DataFrame(result, columns=['fraud_ind'])
    df = pd.merge(df_test, df_label, left_index=True, right_index=True)
    df[['txkey', 'fraud_ind']].to_csv(args.result_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('train_file')
    parser.add_argument('test_file')
    parser.add_argument('result_path')
    
    main(parser.parse_args())
