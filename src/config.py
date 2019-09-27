import os


class Configs:
    # categorial columns in data
    CATEGORY = ['ecfg', 'flbmk', 'flg_3dsmk', 'insfg', 'ovrlt', 'scity', 'csmcu', 'cano', 'mchno', 'hcefg', 'bacno', 'contp', 'etymd', 'acqic']
    # aggregate 
    CONAM_AGG_RECIPE_1 = [
        (["cano"], [
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]),
        (["bacno","cano"], [
            ('conam', 'min'),
            ('conam', 'max'),
            ('conam', 'mean'),
            ('conam', 'median'),
            ('conam', 'var'),
            ('conam', 'sum'),
        ]),

    ]

    CONAM_AGG_RECIPE_2 = [
        (["acqic"], [
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]),
        (["bacno"], [
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]),
        (["csmcu"], [
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]),
        (["mchno"], [
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]),
        (["mcc"], [
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]),
        (["stscd"], [
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]),
        (["stocn"], [
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]),
        (["scity"], [
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]),
        (["contp"], [
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]),
    ]

    ITERM_AGG_RECIPE = [
        (["cano"], [
                ('iterm', 'min'),
                ('iterm', 'max'),
                ('iterm', 'mean'),
                ('iterm', 'median'),
                ('iterm', 'var'),
                ('iterm', 'sum'),
            ]),
        (["acqic"], [
                ('iterm', 'min'),
                ('iterm', 'max'),
                ('iterm', 'mean'),
                ('iterm', 'median'),
                ('iterm', 'var'),
                ('iterm', 'sum'),
            ]),
        (["bacno"], [
                ('iterm', 'min'),
                ('iterm', 'max'),
                ('iterm', 'mean'),
                ('iterm', 'median'),
                ('iterm', 'var'),
                ('iterm', 'sum'),
            ]),
        (["csmcu"], [
                ('iterm', 'min'),
                ('iterm', 'max'),
                ('iterm', 'mean'),
                ('iterm', 'median'),
                ('iterm', 'var'),
                ('iterm', 'sum'),
            ]),
        (["mchno"], [
                ('iterm', 'min'),
                ('iterm', 'max'),
                ('iterm', 'mean'),
                ('iterm', 'median'),
                ('iterm', 'var'),
                ('iterm', 'sum'),
            ]),
        (["mcc"], [
                ('iterm', 'min'),
                ('iterm', 'max'),
                ('iterm', 'mean'),
                ('iterm', 'median'),
                ('iterm', 'var'),
                ('iterm', 'sum'),
            ]),
        (["stscd"], [
                ('iterm', 'min'),
                ('iterm', 'max'),
                ('iterm', 'mean'),
                ('iterm', 'median'),
                ('iterm', 'var'),
                ('iterm', 'sum'),
            ]),
        (["stocn"], [
                ('iterm', 'min'),
                ('iterm', 'max'),
                ('iterm', 'mean'),
                ('iterm', 'median'),
                ('iterm', 'var'),
                ('iterm', 'sum'),
            ]),
        (["scity"], [
                ('iterm', 'min'),
                ('iterm', 'max'),
                ('iterm', 'mean'),
                ('iterm', 'median'),
                ('iterm', 'var'),
                ('iterm', 'sum'),
            ]),
        (["contp"], [
                ('iterm', 'min'),
                ('iterm', 'max'),
                ('iterm', 'mean'),
                ('iterm', 'median'),
                ('iterm', 'var'),
                ('iterm', 'sum'),
            ]),
        (["bacno","cano"], [
            ('iterm', 'min'),
            ('iterm', 'max'),
            ('iterm', 'mean'),
            ('iterm', 'median'),
            ('iterm', 'var'),
            ('iterm', 'sum'),
        ]),
    ]

    HOUR_AGG_RECIPE = [
        (["loctm_hour_of_day"], [
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]),
        (["loctm_hour_of_day"], [
                ('iterm', 'min'),
                ('iterm', 'max'),
                ('iterm', 'mean'),
                ('iterm', 'median'),
                ('iterm', 'var'),
                ('iterm', 'sum'),
            ]),
        (["loctm_hour_of_day","cano"], [
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]),
        (["loctm_hour_of_day","bacno"], [
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]),
    ]
    # feature selection
    FEATURE_GRAVEYARD = [] # list of feature names
    FEATURE_USELESSNESS = ['sum_iterm_BY_csmcu', 'var_iterm_BY_bacno_cano', 'sum_iterm_BY_stocn', 
    'sum_iterm_BY_stscd', 'var_conam_BY_bacno_cano', 'var_iterm_BY_contp', 'median_iterm_BY_csmcu', 
    'sum_iterm_BY_bacno_cano', 'median_iterm_BY_stocn', 'max_conam_BY_bacno_cano', 'max_conam_BY_stscd', 
    'max_iterm_BY_bacno_cano', 'mean_conam_BY_bacno_cano', 'mean_iterm_BY_bacno_cano', 'mean_iterm_BY_stocn', 
    'median_conam_BY_bacno_cano', 'median_iterm_BY_acqic', 'median_iterm_BY_bacno_cano', 
    'median_iterm_BY_contp', 'median_iterm_BY_loctm_hour_of_day', 'median_iterm_BY_mcc', 
    'median_iterm_BY_mchno', 'median_iterm_BY_scity', 'median_iterm_BY_stscd', 
    'sum_iterm_BY_acqic', 'min_conam_BY_bacno_cano', 'min_conam_BY_loctm_hour_of_day', 
    'min_conam_BY_stscd', 'min_iterm_BY_acqic', 'min_iterm_BY_bacno_cano', 'min_iterm_BY_contp', 
    'min_iterm_BY_csmcu', 'min_iterm_BY_loctm_hour_of_day', 'min_iterm_BY_mcc', 
    'min_iterm_BY_scity', 'min_iterm_BY_stocn', 'min_iterm_BY_stscd', 
    'sum_conam_BY_bacno_cano', 'sum_conam_BY_stscd', 'var_iterm_BY_stscd']




