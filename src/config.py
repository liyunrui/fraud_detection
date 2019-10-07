import os

class Configs:
    # categorial columns in data
    CATEGORY = ['ecfg', 'flbmk', 'flg_3dsmk', 'insfg', 'ovrlt', 'scity', 'csmcu', 'cano', 'mchno', 'hcefg', 'bacno', 'contp', 'etymd', 'acqic']
    # aggregate 
    CONAM_AGG_RECIPE_1 = [
        (["cano"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 同一卡號, 刷了幾次卡, 刷卡最大金額,...,etc.
        (["bacno","cano"], [
            ('conam', 'count'),
            ('conam', 'min'),
            ('conam', 'max'),
            ('conam', 'mean'),
            ('conam', 'median'),
            ('conam', 'var'),
            ('conam', 'sum'),
        ]), # 同一歸戶且同一卡號, 刷了幾次卡, 刷卡最大金額,..,etc.

    ]

    CONAM_AGG_RECIPE_3 = [
        (["bacno","csmcu"], [
            ('conam', 'count'),
            ('conam', 'min'),
            ('conam', 'max'),
            ('conam', 'mean'),
            ('conam', 'median'),
            ('conam', 'var'),
            ('conam', 'sum'),
        ]), # 同一歸戶且同ㄧ消費地幣別, 刷了幾次卡, 刷卡最大金額,...,etc.
        (["bacno","acqic"], [
            ('conam', 'count'),
            ('conam', 'min'),
            ('conam', 'max'),
            ('conam', 'mean'),
            ('conam', 'median'),
            ('conam', 'var'),
            ('conam', 'sum'),
        ]), # 同一歸戶且同ㄧ收單行代碼, 刷了幾次卡, 刷卡最大金額,...,etc.
        (["bacno","mcc"], [
            ('conam', 'count'),
            ('conam', 'min'),
            ('conam', 'max'),
            ('conam', 'mean'),
            ('conam', 'median'),
            ('conam', 'var'),
            ('conam', 'sum'),
        ]), # 同一歸戶且同ㄧmcc code, 刷了幾次卡, 刷卡最大金額,...,etc.
        (["bacno","stscd"], [
            ('conam', 'count'),
            ('conam', 'min'),
            ('conam', 'max'),
            ('conam', 'mean'),
            ('conam', 'median'),
            ('conam', 'var'),
            ('conam', 'sum'),
        ]), # 同一歸戶且同一狀態代碼, 刷了幾次卡, 刷卡最大金額,...,etc.
        (["bacno","contp"], [
            ('conam', 'count'),
            ('conam', 'min'),
            ('conam', 'max'),
            ('conam', 'mean'),
            ('conam', 'median'),
            ('conam', 'var'),
            ('conam', 'sum'),
        ]), # 同一歸戶且同ㄧ交易類別, 刷了幾次卡, 刷卡最大金額,...,etc.
        (["bacno","etymd"], [
            ('conam', 'count'),
            ('conam', 'min'),
            ('conam', 'max'),
            ('conam', 'mean'),
            ('conam', 'median'),
            ('conam', 'var'),
            ('conam', 'sum'),
        ]), # 同一歸戶且同一交易型號, 刷了幾次卡, 刷卡最大金額,...,etc.
    ]

    CONAM_AGG_RECIPE_2 = [
        (["acqic"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 同一收單行代碼, 刷了幾次卡, 刷卡最大金額,...,etc.
        (["bacno"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 同一卡號, 刷了幾次卡, 刷卡最大金額, ...,etc.
        (["csmcu"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), #同一消費地幣別, 刷了幾次卡, 刷卡最大金額, ...,etc.
        (["mchno"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), #同一商店, 刷了幾次卡, 刷卡最大金額, ...,etc.
        (["mcc"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]),  #同一mcc代碼, 刷了幾次卡, 刷卡最大金額, ...,etc.
        (["stscd"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), #同一狀態代碼, 刷了幾次卡, 刷卡最大金額, ...,etc.
        (["stocn"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]),
        (["scity"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]),
        (["contp"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), #同一交易類別, 刷了幾次卡, 刷卡最大金額, ...,etc.
        (["etymd"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), #同一交易型號, 刷了幾次卡, 刷卡最大金額, ...,etc.
    ]

    CONAM_AGG_RECIPE_4 = [
        (["hcefg"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 同一支付型態, 刷了幾次卡, 刷卡最大金額,...,etc.
        (["ovrlt"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 同一超額註記碼, 刷了幾次卡, 刷卡最大金額,...,etc.
        (["flbmk"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 同一Fallback 註記, 刷了幾次卡, 刷卡最大金額,...,etc.
        (["bacno","ecfg"], [
            ('conam', 'min'),
            ('conam', 'max'),
            ('conam', 'mean'),
            ('conam', 'median'),
            ('conam', 'var'),
            ('conam', 'sum'),
        ]),
        (["cano","ecfg"], [
            ('conam', 'min'),
            ('conam', 'max'),
            ('conam', 'mean'),
            ('conam', 'median'),
            ('conam', 'var'),
            ('conam', 'sum'),
        ]),
        (["bacno","flg_3dsmk"], [
            ('conam', 'min'),
            ('conam', 'max'),
            ('conam', 'mean'),
            ('conam', 'median'),
            ('conam', 'var'),
            ('conam', 'sum'),
        ]),
        (["cano","flg_3dsmk"], [
            ('conam', 'min'),
            ('conam', 'max'),
            ('conam', 'mean'),
            ('conam', 'median'),
            ('conam', 'var'),
            ('conam', 'sum'),
        ]),
        (["bacno","ovrlt"], [
            ('conam', 'min'),
            ('conam', 'max'),
            ('conam', 'mean'),
            ('conam', 'median'),
            ('conam', 'var'),
            ('conam', 'sum'),
        ]),
        (["cano","ovrlt"], [
            ('conam', 'min'),
            ('conam', 'max'),
            ('conam', 'mean'),
            ('conam', 'median'),
            ('conam', 'var'),
            ('conam', 'sum'),
        ]),
        (["bacno","flbmk"], [
            ('conam', 'min'),
            ('conam', 'max'),
            ('conam', 'mean'),
            ('conam', 'median'),
            ('conam', 'var'),
            ('conam', 'sum'),
        ]),
        (["cano","flbmk"], [
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
        (["etymd"], [
                ('iterm', 'min'),
                ('iterm', 'max'),
                ('iterm', 'mean'),
                ('iterm', 'median'),
                ('iterm', 'var'),
                ('iterm', 'sum'),
            ]), #同一交易型號, 刷了幾次卡, 刷卡最大金額, ...,etc.
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

    HOUR_AGG_SEC_LEVEL_RECIPE_BACNO = [
        (["bacno","day_hr_min"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 同一歸戶, 在同一天, 同一分鐘, 刷了幾次卡, 刷卡最大金額, ...
        (["bacno","day_hr_min_sec"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 同一歸戶, 在同一天, 同一分鐘, 同一秒鐘, 刷了幾次卡, 刷卡最大金額, ...
    ]
    HOUR_AGG_SEC_LEVEL_RECIPE_CANO = [
        (["cano","day_hr_min"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 同一卡號, 在同一天, 同一分鐘, 刷了幾次卡, 刷卡最大金額, ...
        (["cano","day_hr_min_sec"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 同一卡號, 在同一天, 同一分鐘, 同一秒鐘, 刷了幾次卡, 刷卡最大金額, ...
    ]
    HOUR_AGG_SEC_LEVEL_RECIPE_MCHNO = [
        (["mchno","day_hr_min"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 同一店家, 在同一天, 同一分鐘, 刷了幾次卡, 刷卡最大金額, ...
        (["mchno","day_hr_min_sec"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 同一店家, 在同一天, 同一分鐘, 同一秒鐘, 刷了幾次卡, 刷卡最大金額, ...

    ]
    HOUR_AGG_SEC_LEVEL_RECIPE = [
        (["csmcu","day_hr_min"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 同一消費地幣別, 在同一天, 同一分鐘, 刷了幾次卡, 刷卡最大金額, ...
        (["csmcu","day_hr_min_sec"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 同一消費地幣別, 在同一天, 同一分鐘, 同一秒鐘, 刷了幾次卡, 刷卡最大金額, ...
        (["stocn","day_hr_min"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 同一國家, 在同一天, 同一分鐘, 刷了幾次卡, 刷卡最大金額, ...
        (["stocn","day_hr_min_sec"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 同ㄧ城市, 在同一天, 同一分鐘, 同一秒鐘, 刷了幾次卡, 刷卡最大金額, ...

        (["scity","day_hr_min"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 同一歸戶, 在同一天, 同一分鐘, 刷了幾次卡, 刷卡最大金額, ...
        (["scity","day_hr_min_sec"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 同一歸戶, 在同一天, 同一分鐘, 同一秒鐘, 刷了幾次卡, 刷卡最大金額, ...
    ]

    HOUR_AGG_SEC_LEVEL_RECIPE_2 = [
        (["acqic","day_hr_min"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 同一收單行代碼, 在同一天, 同一分鐘, 刷了幾次卡, 刷卡最大金額, ...
        (["acqic","day_hr_min_sec"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 同一歸戶,同一收單行代碼, 在同一天, 同一分鐘, 同一秒鐘, 刷了幾次卡, 刷卡最大金額, ...
        (["bacno","acqic","day_hr_min"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 同一歸戶,同一收單行代碼, 在同一天, 同一分鐘, 刷了幾次卡, 刷卡最大金額, ...
        (["bacno","acqic","day_hr_min_sec"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 同一收單行代碼, 在同一天, 同一分鐘, 同一秒鐘, 刷了幾次卡, 刷卡最大金額, ...
        (["bacno","stocn","day_hr_min"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 同一歸戶, 同ㄧ國家, 在同一天, 同一分鐘, 刷了幾次卡, 刷卡最大金額, ...
        (["bacno","stocn","day_hr_min_sec"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 同一歸戶, 同ㄧ國家, 在同一天, 同一分鐘, 同一秒鐘, 刷了幾次卡, 刷卡最大金額, ...
        (["bacno","scity","day_hr_min"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 同一歸戶, 同一城市, 在同一天, 同一分鐘, 刷了幾次卡, 刷卡最大金額, ...
        (["bacno","scity","day_hr_min_sec"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 同一歸戶, 同ㄧ城市, 在同一天, 同一分鐘, 同一秒鐘, 刷了幾次卡, 刷卡最大金額, ...
    ]

    CANO_CONAM_COUNT_RECIPE = [
        (["cano","conam"], [
                ('bacno', 'count'),
            ]),
    ]

    LOCDT_CONAM_RECIPE = [
        (["bacno","locdt"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 這個歸戶(這個人)在同一天當中刷卡的次數, 刷卡的最大金額, 最小金額, .., 總金額
        (["bacno","locdt","scity"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]),
        (["bacno","locdt","stocn"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]),
        (["bacno","locdt","mchno"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]),
        (["bacno","locdt","stocn","scity"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]),
    ]
    LOCDT_CONAM_RECIPE_2 = [
        (["cano","locdt"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 這個卡號在同一天當中刷卡的次數, 刷卡的最大金額, 最小金額, .., 總金額
        (["mchno","locdt"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 這個商店在同一天當中刷卡的次數, 刷卡的最大金額, 最小金額, .., 總金額
        (["scity","locdt"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 這個城市在同一天當中刷卡的次數, 刷卡的最大金額, 最小金額, .., 總金額
        (["stocn","locdt"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 這個消費地在同一天當中刷卡的次數, 刷卡的最大金額, 最小金額, .., 總金額
        (["contp","locdt"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 這個交易類別在同一天當中刷卡的次數, 刷卡的最大金額, 最小金額, .., 總金額
        (["etymd","locdt"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 這個交易型號在同一天當中刷卡的次數, 刷卡的最大金額, 最小金額, .., 總金額
        (["mcc","locdt"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 這個mcc code在同一天當中刷卡的次數, 刷卡的最大金額, 最小金額, .., 總金額
        (["acqic","locdt"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]), # 這個收單號在同一天當中刷卡的次數, 刷卡的最大金額, 最小金額, .., 總金額
        (["cano","locdt","scity"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]),
        (["cano","locdt","stocn"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]),
        (["cano","locdt","mchno"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]),
        (["cano","locdt","stocn","scity"], [
                ('conam', 'count'),
                ('conam', 'min'),
                ('conam', 'max'),
                ('conam', 'mean'),
                ('conam', 'median'),
                ('conam', 'var'),
                ('conam', 'sum'),
            ]),
    ]
    MCHNO_CONAM_RECIPE = [
        (["bacno","mchno"], [
            ('conam', 'count'),
            ('conam', 'min'),
            ('conam', 'max'),
            ('conam', 'mean'),
            ('conam', 'median'),
            ('conam', 'var'),
            ('conam', 'sum'),
        ]),# 這個歸戶(這個人)在同一店家刷卡的次數, 刷卡的最大金額, 最小金額, .., 總金額
    ]
    SCITY_CONAM_RECIPE = [
        (["bacno","scity"], [
            ('conam', 'count'),
            ('conam', 'min'),
            ('conam', 'max'),
            ('conam', 'mean'),
            ('conam', 'median'),
            ('conam', 'var'),
            ('conam', 'sum'),
        ]), # 這個歸戶(這個人)在同ㄧ城市刷卡的次數, 刷卡的最大金額, 最小金額, .., 總金額
    ]
    STOCN_CONAM_RECIPE = [
        (["bacno","stocn"], [
            ('conam', 'count'),
            ('conam', 'min'),
            ('conam', 'max'),
            ('conam', 'mean'),
            ('conam', 'median'),
            ('conam', 'var'),
            ('conam', 'sum'),
        ]), # 這個歸戶(這個人)在同ㄧ國家刷卡的次數, 刷卡的最大金額, 最小金額, .., 總金額
    ]

    TIME_ELAPSED_AGG_RECIPE = [
        (["conam"], [
                ('time_elapsed_between_last_transactions', 'min'),
                ('time_elapsed_between_last_transactions', 'max'),
                ('time_elapsed_between_last_transactions', 'mean'),
                ('time_elapsed_between_last_transactions', 'median'),
                ('time_elapsed_between_last_transactions', 'var'),
                ('time_elapsed_between_last_transactions', 'sum'),
            ]), # 這個金額, 距離上一次消費的最小天數, 最大天數,...總天數
        (["cano"], [
                ('time_elapsed_between_last_transactions', 'min'),
                ('time_elapsed_between_last_transactions', 'max'),
                ('time_elapsed_between_last_transactions', 'mean'),
                ('time_elapsed_between_last_transactions', 'median'),
                ('time_elapsed_between_last_transactions', 'var'),
                ('time_elapsed_between_last_transactions', 'sum'),
            ]), # 這個卡號, 距離上一次消費的最小天數, 最大天數,...總天數
        (["bacno"], [
                ('time_elapsed_between_last_transactions', 'min'),
                ('time_elapsed_between_last_transactions', 'max'),
                ('time_elapsed_between_last_transactions', 'mean'),
                ('time_elapsed_between_last_transactions', 'median'),
                ('time_elapsed_between_last_transactions', 'var'),
                ('time_elapsed_between_last_transactions', 'sum'),
            ]), # 這個歸戶(這個人), 距離上一次消費的最小天數, 最大天數,...總天數
        (["bacno","cano"], [
            ('time_elapsed_between_last_transactions', 'min'),
            ('time_elapsed_between_last_transactions', 'max'),
            ('time_elapsed_between_last_transactions', 'mean'),
            ('time_elapsed_between_last_transactions', 'median'),
            ('time_elapsed_between_last_transactions', 'var'),
            ('time_elapsed_between_last_transactions', 'sum'),
        ]), # 這個歸戶(這個人)在這個卡號, 距離上一次消費的最小天數, 最大天數,...總天數
        (["bacno","stocn"], [
            ('time_elapsed_between_last_transactions', 'count'),
            ('time_elapsed_between_last_transactions', 'min'),
            ('time_elapsed_between_last_transactions', 'max'),
            ('time_elapsed_between_last_transactions', 'mean'),
            ('time_elapsed_between_last_transactions', 'median'),
            ('time_elapsed_between_last_transactions', 'var'),
            ('time_elapsed_between_last_transactions', 'sum'),
        ]), # 這個歸戶(這個人)在同ㄧ國家距離上一次消費的最小天數, 最大天數,...總天數
        (["bacno","scity"], [
            ('time_elapsed_between_last_transactions', 'count'),
            ('time_elapsed_between_last_transactions', 'min'),
            ('time_elapsed_between_last_transactions', 'max'),
            ('time_elapsed_between_last_transactions', 'mean'),
            ('time_elapsed_between_last_transactions', 'median'),
            ('time_elapsed_between_last_transactions', 'var'),
            ('time_elapsed_between_last_transactions', 'sum'),
        ]), # 這個歸戶(這個人)同ㄧ城市距離上一次消費的最小天數, 最大天數,...總天數
        (["bacno","mchno"], [
            ('time_elapsed_between_last_transactions', 'count'),
            ('time_elapsed_between_last_transactions', 'min'),
            ('time_elapsed_between_last_transactions', 'max'),
            ('time_elapsed_between_last_transactions', 'mean'),
            ('time_elapsed_between_last_transactions', 'median'),
            ('time_elapsed_between_last_transactions', 'var'),
            ('time_elapsed_between_last_transactions', 'sum'),
        ]), # 這個歸戶(這個人)同一店家距離上一次消費的最小天數, 最大天數,...總天數
        (["bacno","hour_range"], [
            ('time_elapsed_between_last_transactions', 'count'),
            ('time_elapsed_between_last_transactions', 'min'),
            ('time_elapsed_between_last_transactions', 'max'),
            ('time_elapsed_between_last_transactions', 'mean'),
            ('time_elapsed_between_last_transactions', 'median'),
            ('time_elapsed_between_last_transactions', 'var'),
            ('time_elapsed_between_last_transactions', 'sum'),
        ]), # 這個歸戶(這個人)同一時距(midnight, early_morning, ..,night)距離上一次消費的最小天數, 最大天數,...總天數
    ]
    TIME_ELAPSED_AGG_RECIPE_2 = [
        (["bacno","time_elapsed_between_last_transactions"], [
            ('conam', 'count'),
            ('conam', 'min'),
            ('conam', 'max'),
            ('conam', 'mean'),
            ('conam', 'median'),
            ('conam', 'var'),
            ('conam', 'sum'),
        ]),# 這個歸戶(這個人)在相同(Time-delta), 刷卡的最大金額, 最小金額, .., 總金額
        (["cano","time_elapsed_between_last_transactions"], [
            ('conam', 'count'),
            ('conam', 'min'),
            ('conam', 'max'),
            ('conam', 'mean'),
            ('conam', 'median'),
            ('conam', 'var'),
            ('conam', 'sum'),
        ]),# 這個卡號在相同(Time-delta), 刷卡的最大金額, 最小金額, .., 總金額
        (["bacno","cano","time_elapsed_between_last_transactions"], [
            ('conam', 'count'),
            ('conam', 'min'),
            ('conam', 'max'),
            ('conam', 'mean'),
            ('conam', 'median'),
            ('conam', 'var'),
            ('conam', 'sum'),
        ]),# 這個歸戶,在這個卡號,在相同(Time-delta), 刷卡的最大金額, 最小金額, .., 總金額
        (["mchno","time_elapsed_between_last_transactions"], [
            ('conam', 'count'),
            ('conam', 'min'),
            ('conam', 'max'),
            ('conam', 'mean'),
            ('conam', 'median'),
            ('conam', 'var'),
            ('conam', 'sum'),
        ]),# 這個商店在相同(Time-delta), 刷卡的最大金額, 最小金額, .., 總金額

    ]
    # rolling stats
    HISTORY_RECIPE = [
            (["bacno"], [
                ('conam', 'mean'),
            ]), 
            (["cano"], [
                ('conam', 'mean'),
            ]), 
            (["mchno"], [
                ('conam', 'mean'),
            ]), 
            (["stocn"], [
                ('conam', 'mean'),
            ]), 
           (["scity"], [
                ('conam', 'mean'),
            ]), 
    ]

    # feature selection
    FEATURE_GRAVEYARD = [] # list of feature names
    FEATURE_USELESSNESS = ['var_iterm_BY_contp', 'median_iterm_BY_stscd', 'median_iterm_BY_stocn', 'median_iterm_BY_scity', 
    'max_conam_BY_bacno_cano', 'var_iterm_BY_bacno_cano', 'median_iterm_BY_acqic', 'median_iterm_BY_mchno', 
    'median_iterm_BY_mcc', 'median_iterm_BY_csmcu', 'median_iterm_BY_bacno_cano', 'median_iterm_BY_contp',
    'median_iterm_BY_loctm_hour_of_day', 'min_iterm_BY_mcc', 'max_conam_BY_stscd', 'var_conam_BY_bacno_cano', 
    'min_iterm_BY_scity', 'min_iterm_BY_stocn', 'min_iterm_BY_stscd', 'min_iterm_BY_loctm_hour_of_day', 
    'min_iterm_BY_csmcu', 'min_iterm_BY_contp', 'sum_conam_BY_bacno_cano', 'mean_iterm_BY_bacno_cano', 
    'min_iterm_BY_bacno_cano', 'min_iterm_BY_acqic', 'min_conam_BY_stscd', 'mean_conam_BY_bacno_cano', 
    'mean_iterm_BY_stocn', 'min_conam_BY_loctm_hour_of_day', 'sum_conam_BY_stscd', 'sum_iterm_BY_acqic', 
    'min_conam_BY_csmcu', 'sum_iterm_BY_bacno_cano', 'median_conam_BY_bacno_cano', 'sum_iterm_BY_csmcu',
    'min_iterm_BY_mchno', 'sum_iterm_BY_stocn', 'sum_iterm_BY_stscd', 'min_conam_BY_bacno_cano', 
    'max_iterm_BY_bacno_cano', 'var_iterm_BY_stscd',

    'count_conam_BY_cano_day_hr_min_sec', 'var_conam_BY_contp', 
    'min_conam_BY_cano_day_hr_min_sec', 'max_iterm_BY_loctm_hour_of_day',

    'mean_conam_BY_bacno_scity_day_hr_min_sec', 'median_conam_BY_bacno_scity_day_hr_min_sec', 
    'median_conam_BY_bacno_acqic_day_hr_min_sec', 'mean_conam_BY_cano_day_hr_min_sec', 
    'mean_conam_BY_bacno_acqic_day_hr_min_sec', 'count_conam_BY_bacno_acqic_day_hr_min_sec', 
    'count_conam_BY_acqic_day_hr_min_sec', 'mean_conam_BY_bacno_stocn_day_hr_min_sec',


    'max_conam_BY_cano_day_hr_min_sec', 'max_conam_BY_contp', 
    'mean_conam_BY_mchno_day_hr_min_sec', 'count_conam_BY_bacno_cano', 
    'count_conam_BY_scity_day_hr_min_sec', 'min_conam_BY_contp_locdt',
    'min_conam_BY_etymd', 'median_iterm_BY_etymd', 'median_iterm_BY_cano', 
    'min_iterm_BY_cano', 'min_iterm_BY_etymd', 'count_conam_BY_stscd', 
    'median_conam_BY_contp', 'sum_conam_BY_contp', 'max_iterm_BY_contp', 
    'sum_conam_BY_etymd', 'count_conam_BY_contp', 'mean_conam_BY_contp', 
    'count_conam_BY_bacno_stocn_day_hr_min_sec', 
    'var_conam_BY_cano_day_hr_min_sec', 'max_iterm_BY_mchno', 'max_iterm_BY_etymd', 
    'var_iterm_BY_stocn',
     # 'sum_time_elapsed_between_last_transactions_BY_bacno_cano', 'sum_iterm_BY_contp', 'max_iterm_BY_acqic',
     # 'mean_iterm_BY_contp', 'mean_iterm_BY_csmcu', 'max_iterm_BY_contp', 
     # 'mean_time_elapsed_between_last_transactions_BY_bacno_cano', 
     # 'var_time_elapsed_between_last_transactions_BY_bacno_cano', 
     # 'min_time_elapsed_between_last_transactions_BY_bacno_cano', 
     # 'min_iterm_BY_cano', 'var_conam_BY_contp', 'max_time_elapsed_between_last_transactions_BY_bacno_cano', 
     # 'median_time_elapsed_between_last_transactions_BY_bacno_cano', 
     # 'max_iterm_BY_stocn'


    ]
