from pdb import set_trace as bp
import pandas as pd

def tensor_to_df(y, ind, cols):
    n_batchs, n_times, n_nodes, n_channels = y.shape
    return pd.DataFrame(
        index=ind,
        data=y.reshape(n_batchs*n_times, n_nodes*n_channels),
        columns=cols
    )

