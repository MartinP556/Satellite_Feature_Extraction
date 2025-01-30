import numpy as np
import pandas as pd
import modelling_fctns
from scipy.interpolate import make_smoothing_spline
import scipy.optimize
import scipy.signal
import data_cleaning

def cross_percentile_date(smoothed_series, percentile):
    # Calculate the 90th percentile
    percentile_value = np.percentile(smoothed_series, 10)
    # Identify when the time series crosses the 90th percentile
    crosses_percentile = (smoothed_series > percentile_value).astype(int).diff().fillna(0).astype(bool)

    # Extract the dates when the crossing occurs
    crossing_dates = smoothed_series.index[crosses_percentile]
    return(crossing_dates)

def double_logistic_LSP(values, Times, first_date):
    Tighter_times = np.arange(Times[0], Times[-1], 1)
    params, _ = scipy.optimize.curve_fit(modelling_fctns.double_logistic, Times/365, values, maxfev = 10000)
    smoothed_series = pd.Series(modelling_fctns.double_logistic(Tighter_times/365, *params), index=Tighter_times/365)
    max_green = first_date + pd.Timedelta(Tighter_times[np.int64(smoothed_series.idxmax()*365)], 'D')
    percentile_10 = first_date + pd.Timedelta(cross_percentile_date(smoothed_series, 10)[0]*365, 'D')
    percentile_50 = first_date + pd.Timedelta(cross_percentile_date(smoothed_series, 50)[0]*365, 'D')
    percentile_90 = first_date + pd.Timedelta(cross_percentile_date(smoothed_series, 90)[0]*365, 'D')
    percentile_10_2nd = first_date + pd.Timedelta(cross_percentile_date(smoothed_series, 10)[-1]*365, 'D')
    return percentile_10, percentile_50, percentile_90, percentile_10_2nd, max_green

def spline_LSP(values, Times, first_date):
    Tighter_times = np.arange(Times[0], Times[-1], 1)
    spl = make_smoothing_spline(Times/365, values, lam = 0.00001)
    smoothed_series = pd.Series(spl(Tighter_times/365), index=Tighter_times/365)
    #print(Tighter_times/365, smoothed_series.idxmax(), smoothed_series)
    max_green = first_date + pd.Timedelta(Tighter_times[np.int64(smoothed_series.idxmax()*365)], 'D')
    percentile_10 = first_date + pd.Timedelta(cross_percentile_date(smoothed_series, 10)[0]*365, 'D')
    percentile_50 = first_date + pd.Timedelta(cross_percentile_date(smoothed_series, 50)[0]*365, 'D')
    percentile_90 = first_date + pd.Timedelta(cross_percentile_date(smoothed_series, 90)[0]*365, 'D')
    percentile_10_2nd = first_date + pd.Timedelta(cross_percentile_date(smoothed_series, 10)[-1]*365, 'D')
    print(spl(Tighter_times[np.int64(smoothed_series.idxmax()*365)]/365), max_green, spl(Tighter_times/365).max())
    return percentile_10, percentile_50, percentile_90, percentile_10_2nd, max_green

def savgol_LSP(values, Times, first_date):
    smoothed_series = pd.Series(scipy.signal.savgol_filter(values, window_length=6, polyorder=3, deriv=0), index=Times)
    smoothed_derivs = pd.Series(scipy.signal.savgol_filter(values, window_length=6, polyorder=3, deriv=1), index=Times)
    mingrad = first_date + pd.Timedelta(smoothed_derivs.idxmin(), 'D')
    maxgrad = first_date + pd.Timedelta(smoothed_derivs.idxmax(), 'D')
    minday = first_date + pd.Timedelta(smoothed_series.idxmin(), 'D')
    maxday = first_date + pd.Timedelta(smoothed_series.idxmax(), 'D')
    return mingrad, maxgrad, minday, maxday

def initialize_LSP_frame(LSP_method):
    if LSP_method == 'double_logistic' or LSP_method == 'spline':
        results = pd.DataFrame(columns = ['year', 'Stations_Id', 'percentile_10', 'percentile_50', 'percentile_90', 'percentile_10_2nd', 'max_green'])
    elif LSP_method == 'savgol':
        results = pd.DataFrame(columns = ['year', 'Stations_Id', 'mingrad', 'maxgrad', 'minday', 'maxday'])
    return results

def append_LSP_frame(results, LSPs):
    results.loc[-1] = LSPs
    results.index = results.index + 1
    results = results.sort_index()
    return results

#def calculate_append_LSP_frame(results, values, Times, first_date, LSP_method):
#    if LSP_method == 'double_logistic' or LSP_method == 'spline':
#        try:
#            if LSP_method == 'double_logistic':
#                LSPs = double_logistic_LSP(values, Times, first_date)
#            else:
#                LSPs = spline_LSP(values, Times, first_date)
#            results = append_LSP_frame(results, [year, station, *LSPs])
#        except:
#            print('couldn\'t compute double logistic')
#            continue
#    elif LSP_method == 'savgol':
#        try:
#            LSPs = savgol_LSP(values, Times, first_date)
#            results = append_LSP_frame(results, [year, station, *LSPs])
#        except:
#            print('couldn\'t compute savgol')
#            continue

def calculate_append_LSP_frame(results, values, Times, first_date, LSP_method, year, station):
    if LSP_method == 'double_logistic' or LSP_method == 'spline':
        if LSP_method == 'double_logistic':
            LSPs = double_logistic_LSP(values, Times, first_date)
        else:
            LSPs = spline_LSP(values, Times, first_date)
        results = append_LSP_frame(results, [year, station, *LSPs])
    elif LSP_method == 'savgol':
        LSPs = savgol_LSP(values, Times, first_date)
        results = append_LSP_frame(results, [year, station, *LSPs])
    return results


def LSP_at_stations(ds, start_year, end_year, LSP_method = 'double_logistic'):
    results = initialize_LSP_frame(LSP_method)
    print(results)
    for year in range(start_year, end_year + 1):
        print(f'Year: {year}')
        for station in ds['Stations_Id'].unique():
            ds_station = ds.loc[ds['Stations_Id'] == station]
            ds_station_year = data_cleaning.restrict_to_growing_season(ds_station, year, ds_station['SOS'].iloc[0], ds_station['EOS'].iloc[0])
            if len(ds_station_year) > 0:
                print(f'Station {station} in {year} has {len(ds_station_year)} observations')
            else:
                print(f'Station {station} in {year} has no observations')
                continue
            max_value_interpolated = data_cleaning.map_max_value_int(ds_station_year, window_size=4) #data_cleaning.
            first_date = pd.DatetimeIndex(max_value_interpolated['time']).min()
            Times = (pd.DatetimeIndex(max_value_interpolated['time']) - pd.DatetimeIndex(max_value_interpolated['time']).min()).days.values
            NDVIs = max_value_interpolated['NDVI'].values
            try:
                results = calculate_append_LSP_frame(results, NDVIs, Times, first_date, LSP_method, year, station)
            except:
                print(f'Couldn\'t compute LSP for station {station} in year {year}')
                continue
    return results