"""
the Matrix Fatorization-based features are created in a unsupervised way 
and therefore lower the need of expert knowledge for the creation 
of the fraud detection system.

bacno-mchno matrix
bacno-contp matrix
bacno-scity matrix
"""
import os
import gc
import argparse
import pandas as pd
import numpy as np
from lightfm import LightFM
from scipy import sparse

def check_folder(folder):
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    return True

def main(args):
    #---------------------------------
    # load dataset
    #---------------------------------
    df_train = pd.read_csv(args.train_file)
    df_test = pd.read_csv(args.test_file)
    df = pd.concat([df_train, df_test], axis = 0)

    del df_train, df_test
    gc.collect()
    #---------------------------------
    # prepare bacno-cano count matrix
    #---------------------------------
    ls = ["bacno", args.column_name]
    interactions = df[ls+["loctm"]].groupby(ls).count().reset_index().rename(columns = {"loctm":"num_count"})
    # min-max normalization
    max_ = interactions.num_count.max()
    min_ = interactions.num_count.min()
    interactions.num_count = interactions.num_count.apply(lambda x : 1.0 * (x-min_)/(max_-min_))

    # row
    num_bacno = interactions.bacno.nunique()
    bacno_dict = {e:i for i, e in enumerate(interactions.bacno.unique())}
    bacno_dict_inv = {e:i for i,e in bacno_dict.items()}
    # column
    num_cano = interactions[args.column_name].nunique()
    cano_dict = {e:i for i, e in enumerate(interactions[args.column_name].unique())}
    cano_dict_inv = {e:i for i,e in cano_dict.items()}

    data = np.zeros(shape = (num_bacno,num_cano), dtype = np.float32)
    for ix, row in interactions.iterrows():
        bacno_index = bacno_dict[row.bacno] # row
        cano_index = cano_dict[row[args.column_name]] # column
        data[bacno_index,cano_index] = row.num_count
    data = sparse.csr_matrix(data)

    del interactions
    gc.collect()

    print ("=" *100)
    print ("data preprocessing done")
    #---------------------------------
    # modeling
    #---------------------------------
    no_components = 10
    # Instantiate and train the model
    model = LightFM(loss='logistic',no_components=no_components)
    model.fit(interactions = data,
              epochs=100, 
              num_threads=2,
              verbose = True)
    #---------------------------------
    # saving
    #---------------------------------
    check_folder(args.latent_feature_path)
    # item_embeddings
    df = pd.concat(
        [pd.DataFrame({args.column_name:list(cano_dict_inv.values())}),
         pd.DataFrame(model.item_embeddings,columns = ["{}_latent_features_{}".format(args.column_name,i) for i in range(no_components)])
        ],axis = 1)
    df.to_csv(os.path.join(args.latent_feature_path, "{}_latent_features.csv".format(args.column_name)), index = False)
    # user_embeddings
    df = pd.concat(
        [pd.DataFrame({"bacno":list(bacno_dict_inv.values())}),
         pd.DataFrame(model.user_embeddings,columns = ["bacno_latent_features_{}_w_{}".format(i,args.column_name) for i in range(no_components)])
        ],axis = 1)
    df.to_csv(os.path.join(args.latent_feature_path, "bacno_latent_features_w_{}.csv".format(args.column_name)), index = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', default='../../dataset/train.csv', type=str)
    parser.add_argument('--test_file', default='../../dataset/test.csv', type=str)
    parser.add_argument('--latent_feature_path', default='../features/', type=str)
    parser.add_argument('--column_name', default='cano', type=str, help = "cano, mchno, contp, scity, stocn, loctm_hour_of_day")

    main(parser.parse_args())
