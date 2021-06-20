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

def _derive_grid_mask(taus, c_type, grid_left, grid_right):
    grid_eta_mask = (taus[c_type, 'deta'] > grid_left) & (taus[c_type, 'deta'] < grid_right)
    grid_phi_mask = (taus[c_type, 'dphi'] > grid_left) & (taus[c_type, 'dphi'] < grid_right)
    return grid_eta_mask * grid_phi_mask

@nb.njit
def derive_grid_mask(deta, dphi, grid_type, inner_grid_left, inner_grid_right, outer_grid_left, outer_grid_right):
    inner_mask = (deta > inner_grid_left) & (deta < inner_grid_right)
    inner_mask *= (dphi > inner_grid_left) & (dphi < inner_grid_right)
    if grid_type == 'inner':
        return inner_mask
    elif grid_type == 'outer':
        outer_mask = (deta > outer_grid_left) & (deta < outer_grid_right)
        outer_mask *= (dphi > outer_grid_left) & (dphi < outer_grid_right)
        outer_mask *= ~(inner_mask)
        return outer_mask
    else:
        raise ValueError("grid_type is expected to be either 'inner' or 'outer'")

def derive_cell_indices(taus, c_type, grid_left, cell_size, dim):
    # do this by affine transforming the grid to an array of grid indices and then flooring to the nearest integer
    return np.floor((taus[c_type, f'd{dim}'] - grid_left) / cell_size)

def sort_constituents_by_var(taus, c_type, var, ascending=True):
    """
    Sort constituents in array `taus[c_type]`` inplace according to the values of `var`.
    Filling function `fill_feature_tensor()` in case of multiple cell entries
    fills the last constituent in a row, which after calling this function would be the one with the highest `val`.
    """
    idx = ak.argsort(taus[c_type][var], ascending=ascending)
    taus[c_type] = taus[c_type][idx]

################################################################################################

def get_lazy_data(path, tree_name, step_size):
    taus = uproot.lazy(f'{path}:{tree_name}', step_size=step_size)
    # taus = uproot.concatenate(f'{path}:{tree_name}', library='ak')
    return taus

def get_batch_yielder(file_name, tree_name, step_size):
    f = uproot.open(file_name)
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

################################################################################################

@nb.njit
def fill_feature_tensor(tensor_to_fill, i_feature, taus_feature, c_deta, c_dphi, grid_type,
                        inner_grid_left, inner_grid_right, inner_cell_size,
                        outer_grid_left, outer_grid_right, outer_cell_size):
    """
    For a single-feature batch `taus_feature`
    loop over all taus and constituents wherein,
    derive indices on the grid (if fall onto the grid area)
    and fill the final tensor `tensor_to_fill`
    """
    if grid_type == 'inner':
        grid_left, grid_right, cell_size = inner_grid_left, inner_grid_right, inner_cell_size
    elif grid_type == 'outer':
        grid_left, grid_right, cell_size = outer_grid_left, outer_grid_right, outer_cell_size
    else:
        raise ValueError("grid_type should be either 'inner' or 'outer'")
    for i_tau in range(len(taus_feature)):
        for i_const in range(len(taus_feature[i_tau])):
            mask = derive_grid_mask(c_deta[i_tau][i_const], c_dphi[i_tau][i_const], grid_type,
                                    inner_grid_left, inner_grid_right, outer_grid_left, outer_grid_right)
            if mask:
                i_eta = np.int(np.floor((c_deta[i_tau][i_const] - grid_left) / cell_size))
                i_phi = np.int(np.floor((c_dphi[i_tau][i_const] - grid_left) / cell_size))
                tensor_to_fill[i_tau, i_eta, i_phi, i_feature] = taus_feature[i_tau][i_const]

# @profile
def fill_tensor(file_name, batch_size, constituent_types, fill_branches, grid_types, n_cells, cell_size):
    """
    Main function which loops over
    batches of taus -> types of constituents -> types of grid ->  feature list
    and fills the output tensor via `fill_feature_tensor()`
    """
    grid_size, grid_left, grid_right = {}, {}, {}
    for grid_type in grid_types:
        grid_size[grid_type] = cell_size[grid_type] * n_cells[grid_type]
        grid_left[grid_type], grid_right[grid_type] = - grid_size[grid_type] / 2, grid_size[grid_type] / 2
    #
    grid_tensors = {key: {} for key in grid_types} # dictionary to store final tensors
    batch_yielder = get_batch_yielder(file_name, 'taus', batch_size)
    for i_batch, taus in enumerate(batch_yielder):
        print('\n' + 30*'*')
        print(f'{i_batch}th batch goes in:')
        for c_type in constituent_types:
            print(f'\n  {c_type} constituents')
            add_vars_to_taus(taus, c_type) # at the moment minor preprocessing and feature engineering
            sort_constituents_by_var(taus, c_type, 'pt', ascending=True)
            for grid_type in grid_types:
                print(f'      {grid_type} grid')
                grid_tensors[grid_type][c_type] = np.zeros((batch_size, n_cells[grid_type], n_cells[grid_type], len(fill_branches[c_type])))
                for i_feature, feature in enumerate(fill_branches[c_type]):
                    fill_feature_tensor(grid_tensors[grid_type][c_type], i_feature, taus[c_type, feature],
                                        taus[c_type, 'deta'], taus[c_type, 'dphi'], grid_type,
                                        grid_left['inner'], grid_right['inner'], cell_size['inner'],
                                        grid_left['outer'], grid_right['outer'], cell_size['outer'])
    return grid_tensors
