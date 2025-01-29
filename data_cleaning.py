import pandas as pd
import numpy as np

def max_in_region(df):
    if len(df) == 0:
        return df
    else:
        return df.loc[[df['NDVI'].idxmax()]]
    
def max_value_int(df, window_size=8):
    '''Returns the maximum value and location of maximum value in windows of size window_size'''
    #df.loc[:, 'formatted_time'] = pd.DatetimeIndex(df['formatted_time'])
    if len(df) == 0:
        return None
    else:
        return df.infer_objects().groupby(pd.Grouper(key='formatted_time',freq=f'{window_size}D'))[df.columns].apply(max_in_region, include_groups=True)
    
def map_max_value_int(df, window_size = 8):
    '''Returns the maximum value and location of maximum value in windows of size window_size'''
    df.loc[:, 'formatted_time'] = pd.DatetimeIndex(df['formatted_time'])
    for latlon_index, latlon in enumerate(df.loc[:, ['lat', 'lon']].drop_duplicates().values):
        #print(latlon)
        df_latlon = df.loc[(df['lat'] == latlon[0]) & (df['lon'] == latlon[1])].dropna()
        #print(len(df_latlon))
        df_max_value_int = max_value_int(df_latlon, window_size)
        #print(len(df_max_value_int))
        if latlon_index == 0:
            df_max_value_int_all = df_max_value_int
        else:
            df_max_value_int_all = pd.concat([df_max_value_int_all, df_max_value_int])
    return df_max_value_int_all