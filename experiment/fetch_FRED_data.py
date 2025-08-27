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
    'RPI': {'transform': 'log_diff'},                # Log difference
    'DPCERA3M086SBEA': {'transform': 'log_diff'},    # Log difference
    'CMRMTSPL': {'transform': 'log_diff'},          # Log difference
    'INDPRO': {'transform': 'log_diff'},             # Log difference
    'CUMFNS': {'transform': 'none'},                 # No transformation
    'UNRATE': {'transform': 'none'},                 # No transformation
    'PAYEMS': {'transform': 'log_diff'},             # Log difference
    'CES0600000007': {'transform': 'log'},           # Log transformation
    'CES0600000008': {'transform': 'log_diff'},      # Log difference
    'PPIFGS': {'transform': 'log_diff'},             # Log difference
    'PPICMM': {'transform': 'log_diff'},             # Log difference
    'PCEPI': {'transform': 'log_diff'},              # Log difference
    'FEDFUNDS': {'transform': 'none'},               # No transformation
    'HOUST': {'transform': 'log'},                   # Log transformation
    #'S&P 500': {'transform': 'log_diff'},            # Log difference
    'EXUSUK': {'transform': 'log_diff'},            # Log difference
    'T1YFFM': {'transform': 'none'},                 # No transformation
    'T10YFFM': {'transform': 'none'},                # No transformation
    'BAAFFM': {'transform': 'none'},                 # No transformation
    #'NAPMNOI': {'transform': 'none'},                # No transformation
    #'HWI': {'transform': 'none'},                    # No transformation
    #'HWIURATIO': {'transform': 'none'},              # No transformation
    'CLF16OV': {'transform': 'none'},                # No transformation
    'CE16OV': {'transform': 'none'},                 # No transformation
    'UEMPMEAN': {'transform': 'none'},               # No transformation
    'UEMPLT5': {'transform': 'none'},                # No transformation
    'UEMP5TO14': {'transform': 'none'},              # No transformation
    'UEMP15OV': {'transform': 'none'},               # No transformation
    'UEMP15T26': {'transform': 'none'},              # No transformation
    'UEMP27OV': {'transform': 'none'},               # No transformation
    #'CLAIMS': {'transform': 'none'},                # No transformation
    'BUSINV': {'transform': 'none'},                # No transformation
    'ISRATIO': {'transform': 'none'},               # No transformation
    'CES0600000008': {'transform': 'log_diff'},      # Log difference
    'CES2000000008': {'transform': 'log_diff'},      # Log difference
    'CES3000000008': {'transform': 'log_diff'},      # Log difference
    'PPIFGS': {'transform': 'log_diff'},             # Log difference
    'PPIFCG': {'transform': 'log_diff'},             # Log difference
    'PPIITM': {'transform': 'log_diff'},             # Log difference
    'PPICRM': {'transform': 'log_diff'},             # Log difference
    'OILPRICE': {'transform': 'none'},              # No transformation
    'PPICMM': {'transform': 'log_diff'},             # Log difference
    'CPIAUCSL': {'transform': 'log_diff'},           # Log difference
    'CPIAPPSL': {'transform': 'log_diff'},           # Log difference
    'CPITRNSL': {'transform': 'log_diff'},           # Log difference
    'CPIMEDSL': {'transform': 'log_diff'},           # Log difference
    'CUSR0000SAC': {'transform': 'log_diff'},        # Log difference
    'CUUR0000SAD': {'transform': 'log_diff'},        # Log difference
    'CUSR0000SAS': {'transform': 'log_diff'},        # Log difference
    'CPIULFSL': {'transform': 'log_diff'},           # Log difference
    'CUUR0000SA0L2': {'transform': 'log_diff'},      # Log difference
    'CUSR0000SA0L5': {'transform': 'log_diff'},      # Log difference
    'PCEPI': {'transform': 'log_diff'},              # Log difference
}

start_date = '1960-01-01'  # Start date as per Carriero et al.
end_date = '2014-12-31'  # End date as per Carriero et al.



# Expected number of rows for the full date range (1960-01-01 to 2014-12-31)
expected_length = pd.date_range(start=start_date, end=end_date, freq='M').shape[0]

# Function to fetch and transform the data
def fetch_and_transform_data(series_info):
    transformed_data = {}

    for variable, info in series_info.items():
        try:
            # Fetch the data from FRED with specified date range
            data = fred.get_series(variable, start_date, end_date)

            # Check if the data length matches the expected length
            if len(data) == expected_length:
                # Apply the specified transformation
                if info['transform'] == 'log':
                    transformed_data[variable] = np.log(data)[:expected_length-1]
                    print(f"transformed {variable}",transformed_data[variable].shape, data.shape )
                elif info['transform'] == 'log_diff':
                    transformed_data[variable] = np.diff(np.log(data), axis=0)[:expected_length-1] # Log difference
                    print(f"transformed {variable}",transformed_data[variable].shape, data.shape )
                elif info['transform'] == 'none':
                    transformed_data[variable] = data[:expected_length-1]
                    print(f"transformed {variable}",transformed_data[variable].shape, data.shape )
            else:
                print(f"Skipping {variable} because the data does not have the full length.")
        except Exception as e:
            print(f"Could not fetch or transform data for {variable}: {e}")
            continue

    return transformed_data

def fetch_fred_data():
    # Fetch and transform the data
    transformed_data = fetch_and_transform_data(series_info)

    # Fetch and transform the data
    transformed_data = fetch_and_transform_data(series_info)
    
    # Combine the transformed data into a DataFrame
    df = pd.DataFrame(transformed_data)
    

    # clean the missing values in the data
    data_cleaned = df.interpolate(method='linear', limit_direction='forward', axis=0)

    # For any remaining NaNs (e.g., at the start or end), fill with the last valid value
    data_cleaned = data_cleaned.fillna(method='bfill').fillna(method='ffill')
    #data_cleaned = df.dropna()


    return data_cleaned

