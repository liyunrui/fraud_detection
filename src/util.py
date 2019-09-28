import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import gc

def string_padding(s):
    """
    s = "819"
    after padding
    s = "000819"
    """
    while len(s)!=6:
        s = "0" + s
    return s

def s_to_time_format(s):
    """
    Test case:
    
    s = "153652"
    s = string_padding(s)
    S, M, H = s[-2:], s[-4:-2], s[:-4]
    datetime_str = "{}:{}:{}".format(H,M,S)
    assert datetime_str == "15:36:52", "It should be the same"
    s = "91819"
    s = string_padding(s)
    S, M, H = s[-2:], s[-4:-2], s[:-4]
    datetime_str = "{}:{}:{}".format(H,M,S)
    assert datetime_str == "09:18:19", "It should be the same"
    s = "819"
    s = string_padding(s)
    S, M, H = s[-2:], s[-4:-2], s[:-4]
    datetime_str = "{}:{}:{}".format(H,M,S)
    assert datetime_str == "00:08:19", "It should be the same"
    s = "5833"
    s = string_padding(s)
    S, M, H = s[-2:], s[-4:-2], s[:-4]
    datetime_str = "{}:{}:{}".format(H,M,S)
    assert datetime_str == "00:58:33", "It should be the same"
 
    """
    s = string_padding(s)
    S, M, H = s[-2:], s[-4:-2], s[:-4]
    datetime_str = "{}:{}:{}".format(H,M,S)
    return datetime_str

def string_to_datetime(datetime_str):
    """
    input: '09:18:19'
    after the function
    return datetime.datetime(1900, 1, 1, 9, 18, 19)
    Please ignore 1900(Year),1(month), 1(day)
    """
    from datetime import datetime
    datetime_object = datetime.strptime(datetime_str, '%H:%M:%S')
    return datetime_object

def hour_to_range(hr):
    if hr > 22 and hr <= 3:
        return 'midnight'
    elif hr > 3 and hr <= 7:
        return 'early_morning'
    elif hr > 7 and hr <= 11:
        return 'morning'
    elif hr > 11 and hr <= 14:
        return 'noon'
    elif hr >14 and hr <= 17:
        return 'afternoon'
    else:
        return 'night'

# Display/plot feature importance
def display_importances(feature_importance_df_, model):
    import matplotlib.pyplot as plt
    import seaborn as sns
    if model == "lgb":
        cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
        best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
        plt.figure(figsize=(16, 10))
        sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout
        plt.savefig('../result/lgbm_importances.png')
    elif model == "xgb":
        cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
        best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
        plt.figure(figsize=(16, 10))
        sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
        plt.title('Xgboost Features (avg over folds)')
        plt.tight_layout
        plt.savefig('../result/xgb_importances.png')
    else:
        print("Now we only support LightGBM or Xgboost model!")  

def lgb_f1_score(y_true, y_pred):
    """evaluation metric"""
    y_hat = np.round(y_pred)
    return 'f1', f1_score(y_true, y_hat), True

# lgb model
def kfold_lightgbm(df_train, df_test, num_folds, args, logger, stratified = False, seed = int(time.time())):
    """
    LightGBM GBDT with KFold or Stratified KFold
    """
    from sklearn.model_selection import KFold, StratifiedKFold
    import multiprocessing   
    import lightgbm as lgb
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=seed)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=seed)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(df_train.shape[0])
    #train_preds = np.zeros(df_train.shape[0])
    sub_preds = np.zeros(df_test.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in df_train.columns if f not in ["fraud_ind"]]
    # k-fold
    if args.TEST_NULL_HYPO:
        # shuffling our label for feature selection
        df_train['fraud_ind'] = df_train['fraud_ind'].copy().sample(frac=1.0).values
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df_train[feats], df_train['fraud_ind'])):
        train_x, train_y = df_train[feats].iloc[train_idx], df_train['fraud_ind'].iloc[train_idx]
        valid_x, valid_y = df_train[feats].iloc[valid_idx], df_train['fraud_ind'].iloc[valid_idx]
        # LightGBM parameters found by Bayesian optimization
        if args.TEST_NULL_HYPO:
            clf = lgb.LGBMClassifier(
                nthread=int(multiprocessing.cpu_count()*args.CPU_USE_RATE),
                n_estimators=10000,
                learning_rate=0.02,
                num_leaves=127,
                max_depth=args.MAX_DEPTH,
                silent=-1,
                verbose=-1,
                random_state=args.seed,
                )
        else:
            clf = lgb.LGBMClassifier(
                n_jobs = -1,
                n_estimators=10000,
                learning_rate=0.02, # 0.02
                num_leaves=args.NUM_LEAVES,
                colsample_bytree=args.COLSAMPLE_BYTREE,
                subsample=args.SUBSAMPLE,
                subsample_freq=args.SUBSAMPLE_FREQ,
                max_depth=args.MAX_DEPTH,
                reg_alpha=args.REG_ALPHA,
                reg_lambda=args.REG_LAMBDA,
                min_split_gain=args.MIN_SPLIT_GAIN,
                min_child_weight=args.MIN_CHILD_WEIGHT,
                max_bin=args.MAX_BIN,
                silent=-1,
                verbose=-1,
                random_state=seed,
                scale_pos_weight=args.SCALE_POS_WEIGHT
                )
        clf.fit(train_x, 
                train_y, 
                eval_set=[(train_x, train_y), (valid_x, valid_y)], 
                eval_metric= lgb_f1_score, 
                verbose= False, 
                early_stopping_rounds= 100, 
                categorical_feature='auto') # early_stopping_rounds= 200
        # probabilty belong to class1(fraud)
        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        #train_preds[train_idx] += clf.predict_proba(train_x, num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
        sub_preds += clf.predict_proba(df_test[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        logger.info('Fold %2d val f1-score : %.6f' % (n_fold + 1, lgb_f1_score(valid_y, oof_preds[valid_idx])[1]))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()
        
    #print('---------------------------------------\nOver-folds train f1-score %.6f' % lgb_f1_score(df_train['fraud_ind'], train_preds)[1])
    logger.info('---------------------------------------\n')
    over_folds_val_score = lgb_f1_score(df_train['fraud_ind'], oof_preds)[1]
    logger.info('Over-folds val f1-score %.6f\n---------------------------------------' % over_folds_val_score)
    # Write submission file and plot feature importance

    if args.ensemble:
        df_test.loc[:,'fraud_ind'] = sub_preds
        df_test[['txkey', 'fraud_ind']].to_csv("../result/lgb.csv", index= False)

    df_test.loc[:,'fraud_ind'] = np.round(sub_preds)
    df_test[['txkey', 'fraud_ind']].to_csv(args.result_path, index= False)
    
    return feature_importance_df, over_folds_val_score

# xgb model
def kfold_xgb(df_train, df_test, num_folds, args, logger, stratified = False, seed = int(time.time())):
    """
    Xgboost with KFold or Stratified KFold
    """
    from sklearn.model_selection import KFold, StratifiedKFold
    import multiprocessing   
    from xgboost import XGBClassifier
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=seed)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=seed)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(df_train.shape[0])
    #train_preds = np.zeros(df_train.shape[0])
    sub_preds = np.zeros(df_test.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in df_train.columns if f not in ["fraud_ind"]]
    # k-fold
    if args.TEST_NULL_HYPO:
        # shuffling our label for feature selection
        df_train['fraud_ind'] = df_train['fraud_ind'].copy().sample(frac=1.0).values
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df_train[feats], df_train['fraud_ind'])):
        train_x, train_y = df_train[feats].iloc[train_idx], df_train['fraud_ind'].iloc[train_idx]
        valid_x, valid_y = df_train[feats].iloc[valid_idx], df_train['fraud_ind'].iloc[valid_idx]
        # LightGBM parameters found by Bayesian optimization
        if args.TEST_NULL_HYPO:
            clf = lgb.LGBMClassifier(
                nthread=int(multiprocessing.cpu_count()*args.CPU_USE_RATE),
                n_estimators=10000,
                learning_rate=0.02,
                num_leaves=127,
                max_depth=args.MAX_DEPTH,
                silent=-1,
                verbose=-1,
                random_state=seed,
                )
        else:
            clf = XGBClassifier(
                n_jobs = -1,
                max_depth=3,
                learning_rate=0.05,
                n_estimators=10000,
                silent=True,
                objective='binary:logistic',
                booster='gbtree',
                gamma=0, 
                min_child_weight=1, 
                max_delta_step=0, 
                subsample=0.8, 
                colsample_bytree=1, 
                colsample_bylevel=1, 
                colsample_bynode=0.8, 
                reg_alpha=0, 
                reg_lambda=1e-05,
                random_state=seed,
                scale_pos_weight=args.SCALE_POS_WEIGHT
                )
        clf.fit(train_x, 
                train_y, 
                eval_set=[(train_x, train_y), (valid_x, valid_y)], 
                eval_metric= lgb_f1_score, 
                verbose= False, 
                early_stopping_rounds= 100, 
                #categorical_feature='auto'
                ) # early_stopping_rounds= 200
        # probabilty belong to class1(fraud)
        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        #train_preds[train_idx] += clf.predict_proba(train_x, num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
        sub_preds += clf.predict_proba(df_test[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        logger.info('Fold %2d val f1-score : %.6f' % (n_fold + 1, lgb_f1_score(valid_y, oof_preds[valid_idx])[1]))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()
        
    #print('---------------------------------------\nOver-folds train f1-score %.6f' % lgb_f1_score(df_train['fraud_ind'], train_preds)[1])
    logger.info('---------------------------------------\n')
    over_folds_val_score = lgb_f1_score(df_train['fraud_ind'], oof_preds)[1]
    logger.info('Over-folds val f1-score %.6f\n---------------------------------------' % over_folds_val_score)

    # Write submission file and plot feature importance
    if args.ensemble:
        df_test.loc[:,'fraud_ind'] = sub_preds
        df_test[['txkey', 'fraud_ind']].to_csv("../result/xgb.csv", index= False)

    df_test.loc[:,'fraud_ind'] = np.round(sub_preds)
    df_test[['txkey', 'fraud_ind']].to_csv(args.result_path, index= False)
    
    return feature_importance_df, over_folds_val_score
    