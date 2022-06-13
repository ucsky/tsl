from pdb import set_trace as bp
import numpy as np
import pandas as pd

def tensor_to_df(y, ind, cols):
    n_batchs, n_times, n_nodes, n_channels = y.shape
    r = len(ind)-n_batchs
    
    bp()
    np.vstack((y[:,0,:, :], y[-1,r ,:, :]))

    return pd.DataFrame(
        index=ind,
        data=y.reshape(n_batchs*n_times, n_nodes*n_channels),
        columns=cols
    )

