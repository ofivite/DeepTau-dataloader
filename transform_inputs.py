#!/usr/bin/env python

import uproot
import awkward as ak
import numpy as np
import pandas as pd
import numba as nb
import matplotlib.pyplot as plt

import gc
from glob import glob
import argparse
import yaml
from memory_profiler import profile

from dataloader.utils import add_vars, sort_constituents_by_var, get_batch_yielder, derive_grid_mask
from dataloader.utils import fill_feature_tensor, fill_tensor

################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--cfg", type=str, help="Path to yaml configuration file")
    parser.add_argument('-d', '--data', type=str, help="Path to directory with input ROOT files")
    parser.add_argument('--n_files', type=int, default=1, help="Number of (first n after sorting) files to process from input data folder")
    parser.add_argument('--n_batches', type=int, default=10, help="Number of batches to create (for each file) and then terminate the program")
    parser.add_argument('--batch_size', type=int, default=1000, help="Number of tau tensors to be placed in one batch")
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
    file_names = sorted(glob(f'{args.data}/*.root'))[:args.n_files]
    for file_name in file_names:
        # get filled tensor
        grid_tensors = fill_tensor(file_name, args.batch_size, args.n_batches, constituent_types, fill_branches, grid_types, n_cells, cell_size)
        # release memory
        for c_type in constituent_types:
            for grid_type in grid_types:
                del grid_tensors[grid_type][c_type]
        gc.collect()
