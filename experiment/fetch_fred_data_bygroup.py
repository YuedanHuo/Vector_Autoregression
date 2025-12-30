# generate data matrix y in the shape of(T,N)

import pandas as pd
from fredapi import Fred
import numpy as np

FRED_API_KEY = "c5bd769e083c29db4135dfff57de42fd"
fred = Fred(api_key=FRED_API_KEY)

def log_difference(series):
    """Apply log difference transformation to a given series."""
    return np.log(series).diff().dropna()  # Compute log difference and drop the first NaN

def log_transformation(series):
    """Apply log transformation to a given series."""
    return np.log(series).dropna()  # Only drop NaN if necessary

# List of 50 series codes and the corresponding transformations
series_info = {
    'RPI': {'transform': 'log_diff', 'group' : 1},                # Log difference
    'DPCERA3M086SBEA': {'transform': 'log_diff', 'group':4},    # Log difference
    #'CMRMTSPL': {'transform': 'log_diff', 'group':4},          # Log difference
    'INDPRO': {'transform': 'log_diff', 'group' : 1},             # Log difference
    'CUMFNS': {'transform': 'none', 'group' : 1},                 # No transformation
    'UNRATE': {'transform': 'none', 'group' : 2},                 # No transformation
    'PAYEMS': {'transform': 'log_diff', 'group' : 2},             # Log difference
    'CES0600000007': {'transform': 'none', 'group' : 2},           # Log transformation
    'CES0600000008': {'transform': 'log_diff', 'group' : 2},      # Log difference
    'WPSFD49207': {'transform': 'log_diff', 'group' : 7},             # Log difference # the series id is renewed according to FRED
    #'PPICMM': {'transform': 'log_diff', 'group' : 7},             # Log difference
    'PCEPI': {'transform': 'log_diff', 'group' : 7},              # Log difference
    #'FEDFUNDS': {'transform': 'none', 'group' : 6},               # No transformation
    'HOUST': {'transform': 'log', 'group' : 3},                   # Log transformation
    'SP500': {'transform': 'log_diff', 'group' : 8},            # Log difference #wasnt able to retrieve this one previously
    'EXUSUK': {'transform': 'log_diff', 'group' : 6},            # Log difference
    #'T1YFFM': {'transform': 'none', 'group' : 6},                 # No transformation
    #'T10YFFM': {'transform': 'none', 'group' : 6},                # No transformation
    'BAAFFM': {'transform': 'none', 'group' : 6},                 # No transformation
    #'NAPMNOI': {'transform': 'none', 'group' : 4},                # No transformation  #wasnt able to retrieve this one previously
    'GS5' :{'transform': 'none', 'group' : 6},
    'GS10' :{'transform': 'none', 'group' : 6},
}

np.random.seed(0)
start_date = '1980-01-01'  # Start and end data
end_date = '2021-03-01'  
first_prediction_data = '2000-01-01'

order = np.arange(1,9)
#np.random.shuffle(order) # the order we want to get the data

# Expected number of rows for the full date range 
expected_length = pd.date_range(start=start_date, end=end_date, freq='M').shape[0]

# Function to fetch and transform the data
def fetch_and_transform_data(variable, info, target_index):
    try:
        raw = fred.get_series(variable, start_date, end_date)
        raw.index = pd.to_datetime(raw.index)

        # If series is empty, return full NaNs
        if raw.empty:
            return pd.Series(index=target_index, data=np.nan, name=variable)

        # Resample to month-start to match target_index
        # This handles daily/weekly series automatically
        raw = raw.resample('MS').last()  # or .mean(), .first(), to be decided

        # Reindex to full canonical index
        raw = raw.reindex(target_index)

        # Transform
        if info['transform'] == 'log':
            out = np.log(raw) 
        elif info['transform'] == 'log_diff':
            out = np.log(raw).diff() * 1200 # according to carriero et al
        else:
            out = raw

        # Fill NaNs: interpolate between known points, then backfill/forward fill edges
        out = out.interpolate(method='linear', limit_direction='both')
        out = out.bfill().ffill()
        out.name = variable
        return out

    except Exception as e:
        print(f"Could not fetch {variable}: {e}")
        return pd.Series(index=target_index, data=np.nan, name=variable)





def fetch_data_by_order(order, series_info, target_index):
    transformed_data = {}
    for group in order:
        for variable, info in series_info.items():
            if info['group'] == group:
                data = fetch_and_transform_data(variable, info, target_index)
                transformed_data[variable] = data
                print(data.shape)
    return transformed_data



def fetch_fred_data():
    target_index = pd.date_range(start=start_date, end=end_date, freq='MS')

    transformed = fetch_data_by_order(order, series_info, target_index)
    df = pd.DataFrame(transformed, index=target_index)

    # clean the missing values in the data
    data_cleaned = df.interpolate(method='linear', limit_direction='forward', axis=0)

    # For any remaining NaNs (e.g., at the start or end), fill with the last valid value
    data_cleaned = data_cleaned.fillna(method='bfill').fillna(method='ffill')

    print("Final dataframe shape:", df.shape)

    # --- NEW PART: Compute observation counts for forecasting ---
    pred_start = pd.to_datetime(first_prediction_data)

    # Ensure index is datetime
    df.index = pd.to_datetime(df.index)

    # Points BEFORE the forecasting window
    n_pre = df[df.index < pred_start].shape[0]

    # Points INSIDE the forecasting window
    n_pred = df[df.index >= pred_start].shape[0]
    
    print(f'Length time series: {df.shape}')
    print(f"Number of data points NOT used for prediction (training): {n_pre}")
    print(f"Number of data points used FOR prediction:                {n_pred}")

    return df, order