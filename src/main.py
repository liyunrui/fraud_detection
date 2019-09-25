"""
python3 main.py ../../dataset/train.csv ../../dataset/test.csv ../result/cv_results.csv ../result/submission.csv > ../result/logs.txt

make train

"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import lightgbm as lgb
from sklearn.metrics import f1_score
import numpy as np
from contextlib import contextmanager
import time
import gc 
from util import s_to_time_format, string_to_datetime,hour_to_range
from time import strftime, localtime
import logging
import sys

CATEGORY = ['ecfg', 'flbmk', 'flg_3dsmk', 'insfg', 'ovrlt', 'scity', 'csmcu', 'cano', 'mchno', 'hcefg', 'bacno', 'contp', 'etymd', 'acqic']

# logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def lgb_f1_score(y_true, y_pred):
    y_hat = np.round(y_pred)
    return 'f1', f1_score(y_true, y_hat), True

def kfold_lightgbm(df_train, df_test, num_folds, args, stratified = False, seed = int(time.time())):
    """
    LightGBM GBDT with KFold or Stratified KFold
    """
    from sklearn.model_selection import KFold, StratifiedKFold
    import multiprocessing   
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
                max_depth=MAX_DEPTH,
                silent=-1,
                verbose=-1,
                random_state=seed,
                )
        else:
            clf = lgb.LGBMClassifier(
                nthread=int(multiprocessing.cpu_count()*args.CPU_USE_RATE),
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
    df_test.loc[:,'fraud_ind'] = np.round(sub_preds)
    df_test[['txkey', 'fraud_ind']].to_csv(args.result_path, index= False)
    
    return feature_importance_df, over_folds_val_score

# Display/plot feature importance
def display_importances(feature_importance_df_):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout
    plt.savefig('../result/lgbm_importances.png')
    
def main(args):
    with timer("Process train/test application"):
        #-------------------------
        # load dataset
        #-------------------------
        df_train = pd.read_csv(args.train_file)
        df_test = pd.read_csv(args.test_file)

        #-------------------------
        # pre-processing
        #-------------------------

        for cat in CATEGORY:
            df_train[cat] = df_train[cat].astype('category')#.cat.codes
            df_test[cat] = df_test[cat].astype('category')
            
        logger.info("Train application df shape:", df_train.shape)
        logger.info("Test application df shape:", df_test.shape)
        
        for df in [df_train, df_test]:
            # pre-processing
            df["loctm_"] = df.loctm.astype(int).astype(str)
            df.loctm_ = df.loctm_.apply(s_to_time_format).apply(string_to_datetime)
            # time-related feature
            df["loctm_hour_of_day"] = df.loctm_.apply(lambda x: x.hour).astype('category')
            #df["loctm_minute_of_hour"] = df.loctm_.apply(lambda x: x.minute)
            #df["loctm_second_of_min"] = df.loctm_.apply(lambda x: x.second)
            #df["loctm_absolute_time"] = [h*60+m for h,m in zip(df.loctm_hour_of_day,df.loctm_minute_of_hour)]
            df["hour_range"] = df.loctm_.apply(lambda x: hour_to_range(x.hour)).astype("category")
            # removed the columns no need
            df.drop(columns = ["loctm_"], axis = 1, inplace = True)
        logger.info("Train application df shape:", df_train.shape)
        logger.info("Test application df shape:", df_test.shape)
    with timer("Run LightGBM with kfold"):
        if args.feature_selection == True:
            for df in [df_train, df_test]:
                # drop random features (by null hypothesis)
                df.drop(FEATURE_GRAVEYARD, axis=1, inplace=True, errors='ignore')

                # drop unused features
                # df.drop(features_with_no_imp_at_least_twice, axis=1, inplace=True, errors='ignore')

                gc.collect()   
        logger.info("Train application df shape:", df_train.shape)
        logger.info("Test application df shape:", df_test.shape)    
        
        
        ITERATION = (80 if args.TEST_NULL_HYPO else 1)
        feature_importance_df = pd.DataFrame()
        over_iterations_val_auc = np.zeros(ITERATION)
        for i in range(ITERATION):
            logger.info('Iteration %i' %i)    
            iter_feat_imp, over_folds_val_auc = kfold_lightgbm(df_train, df_test, num_folds = args.NUM_FOLDS, args = args, stratified = args.STRATIFIED, seed = args.SEED)
            feature_importance_df = pd.concat([feature_importance_df, iter_feat_imp], axis=0)
            over_iterations_val_auc[i] = over_folds_val_auc

        logger.info('============================================\nOver-iterations val AUC score %.6f' %over_iterations_val_auc.mean())
        logger.info('Standard deviation %.6f\n============================================' %over_iterations_val_auc.std())
    
    if args.feature_importance_plot == True:
        display_importances(feature_importance_df)
        
    feature_importance_df_median = feature_importance_df[["feature", "importance"]].groupby("feature").median().sort_values(by="importance", ascending=False)
    useless_features_df = feature_importance_df_median.loc[feature_importance_df_median['importance'] == 0]
    feature_importance_df_mean = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)

    if args.TEST_NULL_HYPO:
        feature_importance_df_mean.to_csv("feature_importance-null_hypo.csv", index = True)
    else:
        feature_importance_df_mean.to_csv("feature_importance.csv", index = True)
        useless_features_list = useless_features_df.index.tolist()
        logger.info('Useless features: \'' + '\', \''.join(useless_features_list) + '\'')

    # save log file
    #log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    log_file = '../result/{}.log'.format(strftime("%y%m%d-%H%M", localtime()))

    logger.addHandler(logging.FileHandler(log_file))

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    logger.info("{} - done in {:.0f}s".format(title, time.time() - t0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', default='../../dataset/train.csv', type=str)
    parser.add_argument('--test_file', default='../../dataset/test.csv', type=str)
    parser.add_argument('--result_path', default='../result/submission.csv', type=str)
    # lgbm parameters(needed to be filled in with best parameters eventually)
    parser.add_argument('--NUM_FOLDS', default=5, type=int, help='number of folds we split for k-fold validation')
    parser.add_argument('--SEED', default=1030, type=int, help='set seed for reproducibility')
    parser.add_argument('--NUM_LEAVES', default=31, type=int, help='Maximum tree leaves for base learners.')
    parser.add_argument('--CPU_USE_RATE', default=1.0, type=float, help='0~1 use how many percentanges of cpu')
    parser.add_argument('--COLSAMPLE_BYTREE', default=1.0, type=float, help = "Subsample ratio of columns when constructing each tree.")
    parser.add_argument('--SUBSAMPLE', default=1.0, type=float, help= " Subsample ratio of the training instance.")
    parser.add_argument('--SUBSAMPLE_FREQ', default=0, type=int, help='Frequence of subsample, <=0 means no enable.')
    parser.add_argument('--MAX_DEPTH', default=-1, type=int, help='Maximum tree depth for base learners, <=0 means no limit.')
    parser.add_argument('--REG_ALPHA', default=0.0, type=float, help = "L1 regularization term on weights.")
    parser.add_argument('--REG_LAMBDA', default=0.0, type=float,  help = "L2 regularization term on weights")
    parser.add_argument('--MIN_SPLIT_GAIN', default=0.0, type=float, help = "Minimum loss reduction required to make a further partition on a leaf node of the tree.")
    parser.add_argument('--MIN_CHILD_WEIGHT', default=0.001, type=float, help= "Minimum sum of instance weight (hessian) needed in a child (leaf).")
    parser.add_argument('--MAX_BIN', default=255, type=int, help='max number of bins that feature values will be bucketed in,  constraints: max_bin > 1')
    parser.add_argument('--SCALE_POS_WEIGHT', default=3.0, type=float, help = "weight of labels with positive class")
    # para
    parser.add_argument('--feature_importance_plot', default=True, type=bool, help='plot feature importance')
    parser.add_argument('--feature_selection', default=False, type=bool, help='drop unused features and random features (by null hypothesis). If true, need to provide features set in list format')
    parser.add_argument('--STRATIFIED', default=True, type=bool, help='use STRATIFIED k-fold. Otherwise, use k-fold')
    parser.add_argument('--TEST_NULL_HYPO', default=False, type=bool, help='get random features by null hypothesis')

    main(parser.parse_args())
