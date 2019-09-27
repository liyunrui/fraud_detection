
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
def display_importances(feature_importance_df_):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(16, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout
    plt.savefig('../result/lgbm_importances.png')



    