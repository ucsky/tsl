from pdb import set_trace as bp
import numpy as np
import pandas as pd

def array_to_df(
        y: np.ndarray,
        ind: pd.core.indexes.datetimes.DatetimeIndex,
        cols: pd.core.indexes.multi.MultiIndex
) -> pd.DataFrame:
    '''
    Recconstruct time series using MultiIndex columns dataframe.
    '''
    n_batchs, window, n_nodes, n_channels = y.shape
    r = len(ind)-n_batchs
    Y = np.vstack(
        (
            y[:,0,:, :],
            y[-1,r ,:, :].reshape(r, n_nodes, n_channels)
        )
    )
    n_times = Y.shape[0]
    df = pd.DataFrame(
        index=ind,
        data=Y.reshape(n_times, n_nodes*n_channels),
        columns=cols
    )
    return df

