#!/usr/bin/env python

import uproot
import awkward as ak
import numpy as np
import pandas as pd
import numba as nb
import matplotlib.pyplot as plt

import tensorflow as tf

import yaml
import time
import glob
# from memory_profiler import profile

################################################################################################

def get_data(path):
    return uproot.concatenate(f'{path}:taus', library='ak')

def get_grid_mask(i_tau, c_type, grid_type):
    grid_mask = taus[i_tau][f'{grid_type}_grid_{c_type}_mask']
    return grid_mask

def get_fill_indices(i_tau, c_type, grid_type, grid_mask):
    indices_eta = taus[i_tau][f'{grid_type}_grid_{c_type}_indices_eta'][grid_mask]
    indices_phi = taus[i_tau][f'{grid_type}_grid_{c_type}_indices_phi'][grid_mask]
    return indices_eta, indices_phi

def get_fill_values(i_tau, c_type, grid_mask):
    fill_values = taus[i_tau][fill_branches[c_type]][grid_mask]
    return fill_values

################################################################################################

# get data
taus = get_data('data/muon_*.root')

# some tau info
n_taus = len(taus)
tau_i = 7 # random tau index for illustrative purposes
constituent_types = ['ele', 'muon', 'pfCand']
fill_branches = {'pfCand': ['pfCand_pt', 'pfCand_deta', 'pfCand_dphi', 'pfCand_mass',]} # branches to be stored

# defining grids
grid_types = ['inner', 'outer']
grid_dim = ['eta', 'phi']
grid_size, grid_left, grid_right = {}, {}, {}

n_cells = {'inner': 11, 'outer': 21}
cell_size = {'inner': 0.02, 'outer': 0.05}

for grid_type in grid_types:
    grid_size[grid_type] = cell_size[grid_type] * n_cells[grid_type]
    grid_left[grid_type], grid_right[grid_type] = - grid_size[grid_type] / 2, grid_size[grid_type] / 2

# grid masks placeholder
grid_mask_dict = {key: {} for key in grid_types}

# initialize grid tensors
grid_tensors = {}
for grid_type in grid_types:
    grid_tensors[grid_type] = np.zeros((n_taus, n_cells[grid_type], n_cells[grid_type], len(fill_branches[c_type])))

################################################################################################

# loop over constituent types
for c_type in constituent_types:
    # counting number of constituents for each tau
    taus[f'n_{c_type}'] = ak.num(taus[f'{c_type}_pt'])

    # normalising constituent coordinates wrt. tau direction
    for dim in ['phi', 'eta']:
        taus[f'{c_type}_d{dim}'] = taus[f'{c_type}_{dim}'] - taus[f'tau_{dim}']

    # deriving grid masks
    for grid_type in grid_types:
        grid_eta_mask = (taus[f'{c_type}_deta'] > grid_left[grid_type]) & (taus[f'{c_type}_deta'] < grid_right[grid_type])
        grid_phi_mask = (taus[f'{c_type}_dphi'] > grid_left[grid_type]) & (taus[f'{c_type}_dphi'] < grid_right[grid_type])
        grid_mask_dict[grid_type][c_type] = grid_eta_mask * grid_phi_mask
    taus[f'inner_grid_{c_type}_mask'] = grid_mask_dict['inner'][c_type]
    taus[f'outer_grid_{c_type}_mask'] = grid_mask_dict['outer'][c_type] * (~grid_mask_dict['inner'][c_type])

    # deriving cell indices
    # do this by affine transforming the grid to an array of grid indices and then flooring to the nearest integer
    for grid_type in grid_types:
        for dim in grid_dim:
            taus[f'{grid_type}_grid_{c_type}_indices_{dim}'] = np.floor((taus[f'{c_type}_d{dim}'] - grid_left[grid_type]) / grid_size[grid_type] * n_cells[grid_type])

c_type = 'pfCand'
for i_tau, tau in enumerate(taus):
    for grid_type in grid_types:
        grid_mask = get_grid_mask(i_tau, c_type, grid_type)
        indices_eta, indices_phi = get_fill_indices(i_tau, c_type, grid_type, grid_mask)
        indices_eta, indices_phi = ak.values_astype(indices_eta, 'int32'), ak.values_astype(indices_phi, 'int32')
        values_to_fill = get_fill_values(i_tau, c_type, grid_mask)
        values_to_fill = ak.to_pandas(values_to_fill).values

        # put them in the tensor
        grid_tensors[grid_type][i_tau, indices_eta, indices_phi, :] = values_to_fill
