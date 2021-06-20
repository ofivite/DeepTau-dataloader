#!/usr/bin/env python

import uproot
import awkward as ak
import numpy as np
import pandas as pd
import numba as nb
import matplotlib.pyplot as plt

import gc
import argparse
import yaml
from memory_profiler import profile

from dataloader.utils import add_vars_to_taus, sort_constituents_by_var, get_batch_yielder, derive_grid_mask
from dataloader.utils import fill_feature_tensor, fill_tensor

################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Path to yaml configuration file")
    parser.add_argument('--data', type=str, help="Path to input data files")
    parser.add_argument('--batch_size', type=int, default=1000, help="Batch size")
    parser.add_argument('--n_batches', type=int, default=1, help="Number of batches")
    args = parser.parse_args()
    with open(args.cfg) as f:
        input_cfg = yaml.load(f, Loader=yaml.FullLoader)
    # constituent info
    constituent_types = input_cfg['constituent_types']
    # branches to be stored
    fill_branches = input_cfg['fill_branches']
    # defining grids
    grid_types = input_cfg['grid_types']
    n_cells = input_cfg['n_cells']
    cell_size = input_cfg['cell_size']
    # get filled tensor
    grid_tensors = fill_tensor(args.data, args.batch_size, constituent_types, fill_branches, grid_types, n_cells, cell_size)
    # # release memory
    # for c_type in constituent_types:
    #     for grid_type in grid_types:
    #         del grid_tensors[grid_type][c_type]
    # gc.collect()
