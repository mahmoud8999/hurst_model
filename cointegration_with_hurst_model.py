# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 04:44:44 2025

@author: Mahmoud
"""

import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import itertools
import statsmodels.api as sm

# This function downloads each ticker provided in the tickers list
def download_data(tickers_list, start_date='2007-01-01', end_date='2024-01-01', interval='1d'):
    data_df = pd.DataFrame()

    for ticker in tickers_list:
        print(f'Starting to download: {ticker}')
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        symbol = ticker.split('=')[0]
        data_df[f'{symbol}'] = data['Close']
        print(f'Done downloading: {ticker}\n')
        
    data_df = data_df.dropna()
    return data_df

# This function finds Non-Stationary Pairs
def non_stationary_pairs(fx_data):
    
    results = []
    
    for pairs_name, pairs_data in fx_data.items():
        adf_result = adfuller(pairs_data)
        
        # Check if the data is Stationary
        if(adf_result[1] < 0.05):
            continue
        # Check if the data is Non-Stationary
        else:
            results.append({
                'Pairs': pairs_name,
                'ADF_Test': adf_result[0],
                'P-Value': adf_result[1],
                'Stationary': 'No'
            })
            
    results_df = pd.DataFrame(results)
    print(results_df)
    
    non_stationary_pairs = results_df['Pairs'].to_list()
    combinations_list = list(itertools.permutations(non_stationary_pairs, 2))
    combinations_df = pd.DataFrame(combinations_list, columns=['Currency1', 'Currency2'])
    
    return combinations_df

# This function calculates the spread of Non-Stationary Pairs
def calculate_spread_for_cointegrated_pairs(combinations_df, fx_data):
    
    spread_results = pd.DataFrame()
    
    for index, row in combinations_df.iterrows():
        currency1 = row['Currency1']
        currency2 = row['Currency2']
        
        X = fx_data[row.iloc[0]]
        y = fx_data[row.iloc[1]]
        
        X = sm.add_constant(X)
        
        model = sm.OLS(y, X)
        results = model.fit()
        
        residuals = results.resid
        beta_coefficient = results.params.iloc[1]
    
        adfuller_result = adfuller(residuals)
        
        # Check if the residuals are Stationary
        if(adfuller_result[1] < 0.05):
            # Calculate the spread if the residuals are stationary
            log_currency1 = np.log(fx_data[currency1])
            log_currency2 = np.log(fx_data[currency2])
            spread = log_currency2 - beta_coefficient * log_currency1
            
            # Create column name by combining currency pairs
            column_name = f"{currency1}_{currency2}"
            
            # Add spread to results dictionary
            spread_results[column_name] = spread
            
    return spread_results

def calculate_hurst(spread_df):
    # Assuming 'df' is your DataFrame
    chunk_size = 1024
    num_chunks = len(spread_df) // chunk_size  # Ignore any remainder rows
    
    # Create chunks
    spread_chunk = [spread_df.iloc[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]
    
    # Create chunks
    spread_chunk = [spread_df.iloc[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]

    # Calculate mean for each chunk
    chunk_means = [chunk.mean() for chunk in spread_chunk]
    
    # Calculate mean-adjusted series (Y) for each chunk
    chunk_adjusted = [chunk - chunk_mean for chunk, chunk_mean in zip(spread_chunk, chunk_means)]
    
    # Calculate cumulative deviate series (Z) for each chunk
    chunk_cumsum = []
    for adjusted_chunk in chunk_adjusted:
       cumsum = pd.DataFrame()
       for column in adjusted_chunk:
           cumsum[column] = adjusted_chunk[column].cumsum()
       chunk_cumsum.append(cumsum)

    # Step 4: Calculate range series (R)
    R_list = []
    for dataframe in chunk_cumsum:
        min_value = dataframe.expanding().min()
        max_value = dataframe.expanding().max()
        R = max_value - min_value
        R_list.append(R)
        
    # Step 5: Modified Standard Deviation calculation with expanding window
    S_list = []
    for dataframe in chunk_adjusted:
        S_df = pd.DataFrame()
        for column in dataframe:
            # Calculate expanding standard deviation
            S_df[column] = dataframe[column].expanding().std()
        S_list.append(S_df)
        
    # Step 6: Calculate R/S ratio
    RS_list = []
    for R, S in zip(R_list, S_list):
        RS = R.div(S)  # Element-wise division of R by S
        RS_list.append(RS)

    # Define time points as per paper (2^4 to 2^10)
    t_values = [16, 32, 64, 128, 256, 512, 1024]
    log_t = np.log2(t_values)
        
    hurst_results = {}
        
    for column in RS_list[0].columns:
        rs_values = []
        for t in t_values:
            rs_t = np.mean([rs[column].iloc[t-1] for rs in RS_list])
            rs_values.append(rs_t)
            
        log_rs = np.log2(rs_values)
            
        # Add constant for regression
        X = sm.add_constant(log_t)
            
        # Perform OLS regression
        model = sm.OLS(log_rs, X)
        results = model.fit()
            
        # Store Hurst exponent (slope coefficient)
        hurst_results[column] = {
            'H': results.params[1],
            'p_value': results.pvalues[1],
            'r_squared': results.rsquared
        }
        
        # Convert dictionary to DataFrame
        hurst_df = pd.DataFrame.from_dict(hurst_results, orient='index')
        hurst_df.index.name = 'Currency'
        hurst_df = hurst_df.rename(columns={'H': 'Hurst Exponent'})
        
    return hurst_df

# List of tickers
tickers_list = ['EURUSD=X', 'GBPUSD=X', 'JPYUSD=X', 'CHFUSD=X', 'CADUSD=X', 'AUDUSD=X', 'NZDUSD=X']

# Download data from Yahoo Finance
fx_data = download_data(tickers_list)

# Create different combinations of FX pairs
combinations_df = non_stationary_pairs(fx_data)

# Calculate the spread of the FX pairs
spread_df = calculate_spread_for_cointegrated_pairs(combinations_df, fx_data)

# Get Hurst Results
hurst_pairs = calculate_hurst(spread_df)




    

