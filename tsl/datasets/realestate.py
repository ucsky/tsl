from pdb import set_trace as bp

import os
from typing import Optional, Sequence, List

import numpy as np
import pandas as pd

from tsl.data.datamodule.splitters import disjoint_months, Splitter
from tsl.ops.dataframe import compute_mean
from tsl.ops.similarities import gaussian_kernel
from tsl.ops.similarities import geographical_distance
from tsl.utils import download_url, extract_zip
from .prototypes import PandasDataset
from .prototypes.mixin import MissingValuesMixin
from ..data.utils import HORIZON


def infer_mask(df, infer_from='next'):
    """Infer evaluation mask from DataFrame. In the evaluation mask a value is 1
    if it is present in the DataFrame and absent in the :obj:`infer_from` month.

    Args:
        df (pd.Dataframe): The DataFrame.
        infer_from (str): Denotes from which month the evaluation value must be
            inferred. Can be either :obj:`previous` or :obj:`next`.

    Returns:
        pd.DataFrame: The evaluation mask for the DataFrame.
    """
    mask = (~df.isna()).astype('uint8')
    eval_mask = pd.DataFrame(index=mask.index, columns=mask.columns,
                             data=0).astype('uint8')
    if infer_from == 'previous':
        offset = -1
    elif infer_from == 'next':
        offset = 1
    else:
        raise ValueError('`infer_from` can only be one of {}'
                         .format(['previous', 'next']))
    months = sorted(set(zip(mask.index.year, mask.index.month)))
    length = len(months)
    for i in range(length):
        j = (i + offset) % length
        year_i, month_i = months[i]
        year_j, month_j = months[j]
        cond_j = (mask.index.year == year_j) & (mask.index.month == month_j)
        mask_j = mask[cond_j]
        offset_i = 12 * (year_i - year_j) + (month_i - month_j)
        mask_i = mask_j.shift(1, pd.DateOffset(months=offset_i))
        mask_i = mask_i[~mask_i.index.duplicated(keep='first')]
        mask_i = mask_i[np.in1d(mask_i.index, mask.index)]
        i_idx = mask_i.index
        eval_mask.loc[i_idx] = ~mask_i.loc[i_idx] & mask.loc[i_idx]
    return eval_mask


class RealEstateSplitter(Splitter):

    def __init__(
            self,
            val_len: int = None,
            test_months: Sequence = (3, 6, 9, 12)
    ):
        super(RealEstateSplitter, self).__init__()
        self._val_len = val_len
        self.test_months = test_months

    def fit(self, dataset):
        nontest_idxs, test_idxs = disjoint_months(
            dataset,
            months=self.test_months,
            synch_mode=HORIZON)
        # take equal number of samples before each month of testing
        val_len = self._val_len
        if val_len < 1:
            val_len = int(val_len * len(nontest_idxs))
        val_len = val_len // len(self.test_months)
        # get indices of first day of each testing month
        delta = np.diff(test_idxs)
        delta_idxs = np.flatnonzero(delta > delta.min())
        end_month_idxs = test_idxs[1:][delta_idxs]
        if len(end_month_idxs) < len(self.test_months):
            end_month_idxs = np.insert(end_month_idxs, 0, test_idxs[0])
        # expand month indices
        month_val_idxs = [
            np.arange(v_idx - val_len, v_idx) - dataset.window
            for v_idx in end_month_idxs
        ]
        val_idxs = np.concatenate(month_val_idxs) % len(dataset)
        # remove overlapping indices from training set
        ovl_idxs, _ = dataset.overlapping_indices(
            nontest_idxs,
            val_idxs,

            synch_mode=HORIZON,
            as_mask=True
        )
        train_idxs = nontest_idxs[~ovl_idxs]
        self.set_indices(train_idxs, val_idxs, test_idxs)


class RealEstate(PandasDataset, MissingValuesMixin):
    """ RealEstate Prices """

    similarity_options = {'distance'}
    temporal_aggregation_options = {'mean', 'nearest'}
    spatial_aggregation_options = {'mean'}

    def __init__(
            self,
            root: str = None,
            impute_nans: bool = True,
            small: bool = False,
            test_months: Sequence = (3, 6, 9, 12),
            infer_eval_from: str = 'next',
            freq: Optional[str] = None,
            masked_sensors: Optional[Sequence] = None,
            max_nodes: Optional[int] = 500
    ):
        if small:
            url = os.environ['TSL_URL_DATA_SMALL']
        else:
            url = os.environ['TSL_URL_DATA']
        self.max_nodes = max_nodes
        self.url = url
        self.root = root
        self.small = small
        self.test_months = test_months
        self.infer_eval_from = infer_eval_from  # [next, previous]
        if masked_sensors is None:
            self.masked_sensors = []
        else:
            self.masked_sensors = list(masked_sensors)
        df, mask, eval_mask, dist = self.load(impute_nans=impute_nans)
        super().__init__(
            dataframe=df,
            attributes=dict(dist=dist),
            mask=mask,
            freq=freq,
            similarity_score='distance',
            temporal_aggregation='mean',
            spatial_aggregation='mean',
            default_splitting_method='air_quality',
            name='RES' if self.small else 'RE'
        )
        self.set_eval_mask(eval_mask)

    @property
    def raw_file_names(self) -> List[str]:
        return ['mvmdts_20200101-20220101-t3000000-m20.h5', 'mvmdts_20200101-20220101-t3000000-m20_small.h5']

    @property
    def required_file_names(self) -> List[str]:
        return self.raw_file_names + ['re_dist.npy']

    def download(self):
        path = download_url(self.url, self.root_dir, 'data.zip')
        extract_zip(path, self.root_dir)
        os.unlink(path)
        
    def give_good_nodes(self, path):
        places = pd.read_hdf(path, 'places')
        places.dropna(subset=['LIE_latitude', 'LIE_longitude'], inplace=True)
        nodes_in_places = places.LIE_id.unique().tolist()
        df = pd.read_hdf(path, 'main', start=0, stop=0)
        nodes_in_main = df.columns.get_level_values(0).unique().tolist()
        good_nodes = sorted(list(set(nodes_in_places) & set(nodes_in_main)))
        return good_nodes

    def build(self):
        self.maybe_download()
        # compute distances from latitude and longitude degrees
        if self.small:
            path = os.path.join(self.root_dir, 'mvmdts_20200101-20220101-t3000000-m20_small.h5')
        else:
            path = os.path.join(self.root_dir, 'mvmdts_20200101-20220101-t3000000-m20.h5')
        places = pd.DataFrame(pd.read_hdf(path, 'places')[['LIE_id', 'LIE_latitude',  'LIE_longitude']])\
            .rename(columns={'LIE_id': 'places_id', 'LIE_latitude': 'latitude',  'LIE_longitude': 'longitude'})\
            .set_index('places_id')
        good_nodes = self.give_good_nodes(path)
        if self.max_nodes:
            good_nodes = good_nodes[0:self.max_nodes+1]
        places = places[places.index.isin(good_nodes)]
        st_coord = places.loc[:, ['latitude', 'longitude']]
        dist = geographical_distance(st_coord, to_rad=True).values
        np.save(os.path.join(self.root_dir, 're_dist.npy'), dist)

    def load_raw(self):
        self.maybe_build()
        dist = np.load(os.path.join(self.root_dir, 're_dist.npy'))
        eval_mask = None
        if self.small:
            path = os.path.join(self.root_dir, 'mvmdts_20200101-20220101-t3000000-m20_small.h5')
        else:
            path = os.path.join(self.root_dir, 'mvmdts_20200101-20220101-t3000000-m20.h5')
        good_nodes = self.give_good_nodes(path)
        if self.max_nodes:
            good_nodes = good_nodes[0:self.max_nodes+1]
        df = pd.read_hdf(path, 'main')
        l_cols = df.columns[df.columns.get_level_values(0).isin(good_nodes)]
        df = df[l_cols]
        df = df.astype("float32")
        return pd.DataFrame(df), dist, eval_mask

    
    def load(self, impute_nans=True):
        # load readings and places metadata
        df, dist, eval_mask = self.load_raw()
        n_times = len(df)
        n_nodes = len(df.columns.get_level_values(0).unique())
        n_channels = len(df.columns.get_level_values(1).unique())
        # compute the masks:
        mask = (~np.isnan(df.values)).astype('uint8')  # 1 if value is valid
        if eval_mask is None:
            eval_mask = infer_mask(df, infer_from=self.infer_eval_from)
        # 1 if value is ground-truth for imputation
        eval_mask = eval_mask.values.astype('uint8')
        if len(self.masked_sensors):
            eval_mask[:, self.masked_sensors] = mask[:, self.masked_sensors]
        # eventually replace nans with weekly mean by hour
        if impute_nans:
            df = df.fillna(compute_mean(df))
        mask = mask.reshape(n_times, n_nodes, n_channels)
        eval_mask = eval_mask.reshape(n_times, n_nodes, n_channels)
        return df, mask, eval_mask, dist

    def get_splitter(self, method: Optional[str] = None, **kwargs):
        if method == 'air_quality':
            val_len = kwargs.get('val_len')
            return RealEstateSplitter(test_months=self.test_months,
                                      val_len=val_len)

    def compute_similarity(self, method: str, **kwargs):
        if method == "distance":
            # use same theta for both air and air36
            theta = np.std(self.dist[:36, :36])
            return gaussian_kernel(self.dist, theta=theta)
