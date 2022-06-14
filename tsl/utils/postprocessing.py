from pdb import set_trace as bp
import numpy as np
import pandas as pd

from tsl.data.utils import SynchMode
from tsl.ops.imputation import prediction_dataframe
from tsl.data.datamodule.spatiotemporal_datamodule import SpatioTemporalDataModule

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


def save_ts(
        tag: str,
        path_results: str,
        y_hat: np.ndarray,
        y_true: np.ndarray,
        dm: SpatioTemporalDataModule,
        dataset,
):
    '''
    Convert batch prediction to times series and save it in HDF5 store.
    '''
    df_hat = prediction_dataframe(
        y_hat,
        dm.torch_dataset.data_timestamps(dm.testset.indices)[SynchMode.HORIZON],
        dataset.df.columns,
        aggregate_by='mean'
    )
    df_true = prediction_dataframe(
        y_true,
        dm.torch_dataset.data_timestamps(dm.testset.indices)[SynchMode.HORIZON],
        dataset.df.columns,
        aggregate_by='mean'
    )
    df_hat.to_hdf(path_results, key=f"hat_{tag}")
    df_true.to_hdf(path_results, key=f"true_{tag}")
