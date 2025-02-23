import pandas as pd
import numpy as np
from PIL import Image
import xarray as xr

def max_in_region(df):
    if len(df) == 0:
        return df
    else:
        return df.loc[[df['NDVI'].idxmax()]]
    
def max_value_int(df, window_size=8):
    '''Returns the maximum value and location of maximum value in windows of size window_size'''
    #df.loc[:, 'date'] = pd.DatetimeIndex(df['date'])
    if len(df) == 0:
        return None
    else:
        return df.infer_objects().groupby(pd.Grouper(key='date',freq=f'{window_size}D'))[df.columns].apply(max_in_region, include_groups=True)
    
def map_max_value_int(df, window_size = 8, bands = ['NDVI']):
    '''Returns the maximum value and location of maximum value in windows of size window_size'''
    df.loc[:, 'date'] = pd.DatetimeIndex(df['date'])
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
    df_max_value_int_all = df_max_value_int_all.rename(columns = {'date': 'raw_time'}).reset_index().rename(columns = {'date': 'time'})
    for band in bands:
        df_max_value_int_all[band] = np.interp(df_max_value_int_all['time'], df_max_value_int_all['raw_time'], df_max_value_int_all[band])
    return df_max_value_int_all

def resample_linear(ds):
    ds['date'] = pd.to_datetime(pd.to_datetime(ds['date'], format='%Y-%m-%d-%H-%M-%S').dt.date)
    ds.set_index('date', inplace=True)
    ds = ds[~ds.index.duplicated()]
    ds= ds.resample('D').asfreq()#.infer_objects(copy=False)
    ds.loc[:, ds.columns != 'reducer']  = ds.loc[:, ds.columns != 'reducer'].interpolate()
    ds.reset_index(inplace=True)
    ds.rename(columns={'date': 'time'}, inplace=True)
    return ds


def randomly_sample_windows(df, m_window_size = 6):
    # First sample a location:
    latlon = df.loc[:, ['lat', 'lon']].drop_duplicates().sample(1)
    ds_loc = df.loc[(df['lat'] == latlon['lat'].values[0]) & (df['lon'] == latlon['lon'].values[0])]
    #print(len(ds_loc))
    #Then sample a time window:
    if len(ds_loc) < m_window_size:
        return None
    else:
        start = np.random.randint(len(ds_loc) - m_window_size)
        return ds_loc[start:start+m_window_size]

def make_df_samples(df, sample_number = 100, m_window_size = 5, seed = 0):
    '''Randomly sample windows of length m_window_size from the maximum value composited time series in df'''
    df_time_adjust = df.copy()
    df_time_adjust['time from edge'] = (df_time_adjust['raw_time'] - df_time_adjust['time']).dt.days
    np.random.seed(seed)
    first = True
    for count in range(sample_number):
        sample = randomly_sample_windows(df_time_adjust, m_window_size=m_window_size)
        #print(count)
        if sample is None:
            continue
        else:
            sample_array = xr.Dataset.from_dataframe(sample.loc[:, ['median sur_refl_b03', 'median sur_refl_b04', 'NDVI', 'time from edge']])
            sample_array = sample_array.assign({"indexer": (("index"), np.arange(1, 6))}).set_index({'index': 'indexer'})
        if first == True:
            full_array = sample_array.expand_dims('sample')
            first = False
        else:
            full_array = xr.concat([full_array, sample_array.expand_dims('sample')], dim='sample')
    return full_array

def make_tensor_from_timeseries(df, sample_number = 100, m_window_size = 5, seed = 0, format_choice = 'pytorch'):
    '''Randomly sample windows of length m_window_size from the maximum value composited time series in df'''
    df_time_adjust = df.copy()
    df_time_adjust['time from edge'] = (df_time_adjust['raw_time'] - df_time_adjust['time']).dt.days
    np.random.seed(seed)
    first = True
    for count in range(sample_number):
        sample = randomly_sample_windows(df_time_adjust, m_window_size=m_window_size)
        #print(count)
        if sample is None:
            continue
        else:
            flattened_sample = np.array([sample.loc[:, ['median sur_refl_b03', 'median sur_refl_b04', 'NDVI', 'time from edge']].values.flatten()])
            #print(flattened_sample.shape)
            if first == True:
                tensor =flattened_sample
                first = False
                #print('First')
            else:
                tensor = np.concatenate([tensor, flattened_sample], axis = 0)
    #print(tensor)
    #print(torch.tensor(tensor))
    if format_choice == 'pytorch':
        return torch.tensor(tensor)
    else:
        return tensor
    
def WC_SOS(lon, lat):
    #y = np.int32((2*lat) + 286/2)
    #x = np.int32((2*lon) + 720/2)
    y = np.int32((1 - (lat + 59)/143)*286)
    x = np.int32((lon + 180)*2)
    photo = Image.open("Useful_Files\\M1_SOS_WGS84.tif")
    data = np.array(photo)
    return data[y, x]
def WC_EOS(lon, lat):
    #y = np.int32((2*lat) + 286/2)
    #x = np.int32((2*lon) + 720/2)
    y = np.int32((1 - (lat + 59)/143)*286)
    x = np.int32((lon + 180)*2)
    photo = Image.open("Useful_Files\\M1_EOS_WGS84.tif")
    data = np.array(photo)
    return data[y, x]

def add_SOS_to_df(df):
    df['SOS'] = WC_SOS(df['lon'], df['lat'])
    return df
def add_EOS_to_df(df):
    df['EOS'] = WC_EOS(df['lon'], df['lat'])
    return df

def restrict_to_growing_season(ds, year, SOS, EOS):
    if SOS > EOS:
        start_of_period = (pd.Timestamp(f'{year}-01-01') + pd.Timedelta(SOS - 20, 'D'))#.tz_localize('UTC') 
        end_of_period = (pd.Timestamp(f'{year + 1}-01-01') + pd.Timedelta(EOS + 20, 'D'))#.tz_localize('UTC')
    else:
        start_of_period = (pd.Timestamp(f'{year}-01-01') + pd.Timedelta(SOS - 20, 'D'))#.tz_localize('UTC') 
        end_of_period = (pd.Timestamp(f'{year}-01-01') + pd.Timedelta(EOS + 20, 'D'))#.tz_localize('UTC')
    ds5 = ds.copy()
    ds5['date'] = pd.to_datetime(ds5['date'], format='%Y-%m-%d-%H-%M-%S')
    return ds5.loc[(ds5['date'] > start_of_period) & (ds5['date'] < end_of_period)]

def autocorrelation_matrix(ds, variable = 'NDVI', index_coord = 'index'):
    X = (ds[variable] - ds[variable].mean(dim='sample')).values
    return np.corrcoef(X.T)
