import uproot
import awkward as ak
import numpy as np
import pandas as pd
import numba as nb
import matplotlib.pyplot as plt

import gc
import argparse
from memory_profiler import profile

################################################################################################

def add_vars_to_taus(taus, c_type):
    taus[f'n_{c_type}'] = ak.num(taus[c_type]) # counting number of constituents for each tau
    for dim in ['phi', 'eta']:
        taus[c_type, f'd{dim}'] = taus[c_type, dim] - taus[f'tau_{dim}'] # normalising constituent coordinates wrt. tau direction

def derive_grid_mask(taus, c_type, grid_left, grid_right):
    grid_eta_mask = (taus[c_type, 'deta'] > grid_left) & (taus[c_type, 'deta'] < grid_right)
    grid_phi_mask = (taus[c_type, 'dphi'] > grid_left) & (taus[c_type, 'dphi'] < grid_right)
    return grid_eta_mask * grid_phi_mask

def derive_cell_indices(taus, c_type, grid_left, cell_size, dim):
    # do this by affine transforming the grid to an array of grid indices and then flooring to the nearest integer
    return np.floor((taus[c_type, f'd{dim}'] - grid_left) / cell_size)

################################################################################################

def get_lazy_data(path, tree_name, step_size):
    taus = uproot.lazy(f'{path}:{tree_name}', step_size=step_size)
    # taus = uproot.concatenate(f'{path}:{tree_name}', library='ak')
    return taus

def get_batch_yielder(path, tree_name, step_size):
    f = uproot.open(path)
    batch_yielder = f[tree_name].iterate(library="ak", step_size=step_size, how="zip")
    return batch_yielder

def get_grid_mask(taus, c_type, grid_type):
    return taus[c_type, f'{grid_type}_grid_mask']

def get_fill_indices(taus, c_type, grid_type, grid_mask):
    indices_eta = taus[c_type, f'{grid_type}_grid_indices_eta'][grid_mask]
    indices_phi = taus[c_type, f'{grid_type}_grid_indices_phi'][grid_mask]
    indices_eta, indices_phi = ak.values_astype(indices_eta, 'int32'), ak.values_astype(indices_phi, 'int32')
    return indices_eta, indices_phi

def get_fill_values(tau, c_type, branches, grid_mask):
    values_to_fill = ak.to_numpy(tau[c_type, branches][grid_mask]).tolist()
    # values_to_fill = np.apply_along_axis(lambda v: list(v), 0, values_to_fill)
    # values_to_fill = [list(v) for v in values_to_fill]
    return values_to_fill
