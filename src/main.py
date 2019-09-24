"""
python3 main.py ../../dataset/train.csv ../../dataset/test.csv ../result/cv_results.csv ../result/submission.csv > ../result/logs.txt

"""
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import lightgbm as lgb
from sklearn.metrics import f1_score
import numpy as np
#from keras.models import load_model
from util import s_to_time_format, string_to_datetime, hour_to_range
import gc

os.environ["CUDA_VISIBLE_DEVICES"]="0"

CATEGORY = ['ecfg', 'flbmk', 'flg_3dsmk', 'insfg', 'ovrlt', 'scity', 'csmcu', 'cano', 'mchno', 'hcefg', 'bacno', 'contp', 'etymd', 'acqic']


def lgb_f1_score(y_true, y_pred):
    y_hat = np.round(y_pred)
    return 'f1', f1_score(y_true, y_hat), True

def label_encoder(x_train, x_test, df_test):
    from sklearn import preprocessing

    df = pd.concat([x_train,x_test,df_test], axis = 0)
    assert len(df)== len(x_train)+len(x_test)+len(df_test), "it should be same"
    
    for cat in CATEGORY:
        le = preprocessing.LabelEncoder()
        le.fit(df[cat].tolist())

        x_train[cat] = le.transform(x_train[cat].tolist()) 
        x_test[cat] = le.transform(x_test[cat].tolist()) 
        df_test[cat] = le.transform(df_test[cat].tolist()) 

    print ("*"* 100)
    print ("finished label encoding")
    return x_train,x_test,df_test

def pre_processing_for_auto_encoder(df):
    df = df.drop(['txkey'], axis=1)
    return df

def normalizing_for_auto_encoder(x_train,x_test,df_test):
    """
    return array
    """
    from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

    df = pd.concat([x_train,x_test,df_test], axis = 0)
    assert len(df)== len(x_train)+len(x_test)+len(df_test), "it should be same"

    scaler = MinMaxScaler()
    
    scaler.fit(df)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    df_test = scaler.transform(df_test)
    print ("*"* 100)
    print ("finished data normalizing")
    return x_train,x_test,df_test

def add_auto_encoder_feature(df_raw, df, autoencoder, add_reconstructed = True):
    predictions = autoencoder.predict(df) # get reconstructed vector, 2-D, [num_samples, num_features]
    mse = np.mean(np.power(df - predictions, 2), axis=1) # get reconstructed error, 1-D, [num_samples,]
    if add_reconstructed == True:
        df = pd.DataFrame(predictions, columns=["reconstructed_dim_{}".format(i) for i in range(predictions.shape[1])])
        df["reconstruction_error"] = mse
    else:
        df = pd.DataFrame({"reconstruction_error": mse})
    out = pd.concat([df_raw.reset_index(), df.reset_index()], axis = 1)
    assert len(out)==len(df_raw)==len(df), "it should be same"
    print ("*"* 100)
    print ("finished adding auto_encoder_feature")    
    return out

def main(args):
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

    for df in [df_train, df_test]:
        # pre-processing
        df["loctm_"] = df.loctm.astype(int).astype(str)
        df.loctm_ = df.loctm_.apply(s_to_time_format).apply(string_to_datetime)
        # time-related feature
        df["hour_range"] = df.loctm_.apply(lambda x: hour_to_range(x.hour)).astype("category")
        #df["loctm_hour_of_day"] = df.loctm_.apply(lambda x: x.hour)
        #df["loctm_minute_of_hour"] = df.loctm_.apply(lambda x: x.minute)
        #df["loctm_second_of_min"] = df.loctm_.apply(lambda x: x.second)
        #df["loctm_absolute_time"] = [h*60+m for h,m in zip(df.loctm_hour_of_day,df.loctm_minute_of_hour)]

        # removed the columns no need
        df.drop(columns = ["loctm_"], axis = 1, inplace = True)
        
    print ("*"* 100)
    print ("finished pre-processing")    

    del df
    gc.collect()

    y_train = df_train['fraud_ind']
    x_train = df_train.drop('fraud_ind', axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

    # #-------------------------
    # # auto_encoer
    # #-------------------------
    # x_train_, x_test_, df_test_ =  x_train.copy(), x_test.copy(), df_test.copy()

    # x_train_, x_test_, df_test_ = label_encoder(x_train_, x_test_, df_test_)

    # x_train_ = pre_processing_for_auto_encoder(x_train_) 
    # x_test_ = pre_processing_for_auto_encoder(x_test_)
    # df_test_ = pre_processing_for_auto_encoder(df_test_)

    # x_train_, x_test_, df_test_ = normalizing_for_auto_encoder(x_train_, x_test_, df_test_)

    # autoencoder = load_model('/data/yunrui_li/fraud/fraud_detection/models/model.h5')

    # x_train = add_auto_encoder_feature(x_train,x_train_, autoencoder, add_reconstructed = False)
    # x_test = add_auto_encoder_feature(x_test,x_test_, autoencoder, add_reconstructed = False)
    # df_test = add_auto_encoder_feature(df_test,df_test_, autoencoder, add_reconstructed = False)



    # model
    estimator = lgb.LGBMClassifier(num_leaves=31)

    param_grid = {
            'learning_rate': [0.1],
            'n_estimators': [1000],
            'scale_pos_weight': [3, 5]
            }

    gbm = GridSearchCV(estimator, 
                       param_grid, 
                       cv = 10,
                       scoring='f1', 
                       return_train_score = True,
                       n_jobs = -1)
    gbm.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric=lgb_f1_score, early_stopping_rounds=5, categorical_feature='auto')
    
    # cross-validation result
    df = pd.DataFrame(gbm.cv_results_)
    df.sort_values(by = "mean_test_score", inplace =True)
    df.to_csv(args.logs_path, index=False)
    mean_test_score = df.iloc[0].mean_test_score
    std_test_score = df.iloc[0].std_test_score
    print ("10-fold validating result on best paras : {} with +/- {}".format(round(mean_test_score, 4), round(std_test_score,4)))
    

    # prediction
    result = gbm.predict(df_test)
    df_label = pd.DataFrame(result, columns=['fraud_ind'])
    df = pd.merge(df_test, df_label, left_index=True, right_index=True)
    df[['txkey', 'fraud_ind']].to_csv(args.result_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('train_file')
    parser.add_argument('test_file')
    parser.add_argument('logs_path')
    parser.add_argument('result_path')

    main(parser.parse_args())
