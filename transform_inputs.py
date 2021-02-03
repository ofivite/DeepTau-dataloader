#!/usr/bin/env python

import uproot
import awkward as ak
import numpy as np
import pandas as pd
import numba as nb
import matplotlib.pyplot as plt

import yaml
import time
import glob
import gc
import argparse
from memory_profiler import profile

################################################################################################

# constituent info
constituent_types = ['ele', 'muon', 'pfCand']
fill_branches = {'ele': ['pt',
                         'eta',
                         'phi',
                         'mass',
                         'cc_ele_energy',
                         'cc_gamma_energy',
                         'cc_n_gamma',
                         'dxy',
                         'dxy_error',
                         'ip3d',
                         'trackMomentumAtVtx',
                         'trackMomentumAtCalo',
                         'trackMomentumOut',
                         'trackMomentumAtEleClus',
                         'trackMomentumAtVtxWithConstraint',
                         'ecalEnergy',
                         'ecalEnergy_error',
                         'eSuperClusterOverP',
                         'eSeedClusterOverP',
                         'eSeedClusterOverPout',
                         'eEleClusterOverPout',
                         'deltaEtaSuperClusterTrackAtVtx',
                         'deltaEtaSeedClusterTrackAtCalo',
                         'deltaEtaEleClusterTrackAtCalo',
                         'deltaEtaSeedClusterTrackAtVtx',
                         'deltaPhiEleClusterTrackAtCalo',
                         'deltaPhiSuperClusterTrackAtVtx',
                         'deltaPhiSeedClusterTrackAtCalo',
                         'mvaInput_earlyBrem',
                         'mvaInput_lateBrem',
                         'mvaInput_sigmaEtaEta',
                         'mvaInput_hadEnergy',
                         'mvaInput_deltaEta',
                         'gsfTrack_normalizedChi2',
                         'gsfTrack_numberOfValidHits',
                         'gsfTrack_pt',
                         'gsfTrack_pt_error'],
                 'muon': ['pt',
                         'eta',
                         'phi',
                         'mass',
                         'dxy',
                         'dxy_error',
                         'normalizedChi2',
                         'numberOfValidHits',
                         'segmentCompatibility',
                         'caloCompatibility',
                         'pfEcalEnergy',
                         'type',
                         'n_matches_DT_1',
                         'n_matches_DT_2',
                         'n_matches_DT_3',
                         'n_matches_DT_4',
                         'n_matches_CSC_1',
                         'n_matches_CSC_2',
                         'n_matches_CSC_3',
                         'n_matches_CSC_4',
                         'n_matches_RPC_1',
                         'n_matches_RPC_2',
                         'n_matches_RPC_3',
                         'n_matches_RPC_4',
                         'n_matches_GEM_1',
                         'n_matches_GEM_2',
                         'n_matches_GEM_3',
                         'n_matches_GEM_4',
                         'n_matches_ME0_1',
                         'n_matches_ME0_2',
                         'n_matches_ME0_3',
                         'n_matches_ME0_4',
                         'n_hits_DT_1',
                         'n_hits_DT_2',
                         'n_hits_DT_3',
                         'n_hits_DT_4',
                         'n_hits_CSC_1'],
                 'pfCand': [
                     'jetDaughter',
                     'tauSignal',
                     'leadChargedHadrCand',
                     'tauIso',
                     'pt',
                     'eta',
                     'phi',
                     'mass',
                     'pvAssociationQuality',
                     'fromPV',
                     'puppiWeight',
                     'puppiWeightNoLep',
                     'pdgId',
                     'charge',
                     'lostInnerHits',
                     'numberOfPixelHits',
                     'numberOfHits',
                     'vertex_x',
                     'vertex_y',
                     'vertex_z',
                     'vertex_t',
                     'time',
                     'timeError',
                     'hasTrackDetails',
                     'dxy',
                     'dxy_error',
                     'dz',
                     'dz_error',
                     'track_chi2',
                     'track_ndof',
                     'caloFraction',
                     'hcalFraction',
                     'rawCaloFraction',
                     'rawHcalFraction',
                 ]
} # branches to be stored

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

################################################################################################

def add_vars_to_taus(taus, c_type):
    taus[f'n_{c_type}'] = ak.num(taus[c_type]) # counting number of constituents for each tau
    for dim in ['phi', 'eta']:
        taus[c_type, f'd{dim}'] = taus[c_type, dim] - taus[f'tau_{dim}'] # normalising constituent coordinates wrt. tau direction

def derive_grid_mask(taus, c_type, grid_type):
    grid_eta_mask = (taus[c_type, 'deta'] > grid_left[grid_type]) & (taus[c_type, 'deta'] < grid_right[grid_type])
    grid_phi_mask = (taus[c_type, 'dphi'] > grid_left[grid_type]) & (taus[c_type, 'dphi'] < grid_right[grid_type])
    return grid_eta_mask * grid_phi_mask

def derive_cell_indices(taus, c_type, grid_type, dim):
    # do this by affine transforming the grid to an array of grid indices and then flooring to the nearest integer
    return np.floor((taus[c_type, f'd{dim}'] - grid_left[grid_type]) / grid_size[grid_type] * n_cells[grid_type])

################################################################################################

def get_data(path, tree_name, step_size):
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

################################################################################################

# @profile
def fill_tensor(path_to_data, batch_size, n_batches):
    # initialize grid tensors dictionary
    grid_tensors = {key: {} for key in grid_types}
    # get data
    batch_yielder = get_batch_yielder(path_to_data, 'taus', batch_size)
    # looping over batches of taus
    for i_batch, taus in enumerate(batch_yielder):
        print('\n' + 30*'*')
        print(f'{i_batch}th batch goes in:\n')
        # derive grid masks and indices therein
        for c_type in constituent_types:
            add_vars_to_taus(taus, c_type)
            for grid_type in grid_types:
                grid_mask_dict[grid_type][c_type] = derive_grid_mask(taus, c_type, grid_type)
                for dim in grid_dim:
                    taus[c_type, f'{grid_type}_grid_indices_{dim}'] = derive_cell_indices(taus, c_type, grid_type, dim)
            # store grid masks as branches
            taus[c_type, 'inner_grid_mask'] = grid_mask_dict['inner'][c_type]
            taus[c_type, 'outer_grid_mask'] = grid_mask_dict['outer'][c_type] * (~grid_mask_dict['inner'][c_type])

        # looping over constituents
        for c_type in constituent_types:
            print(f'  {c_type} constituents:\n')
            # looping over grids
            for grid_type in grid_types:
                print(f'      {grid_type} grid:\n')
                # init grid tensors with 0
                grid_tensors[grid_type][c_type] = np.zeros((batch_size, n_cells[grid_type], n_cells[grid_type], len(fill_branches[c_type])))
                # fetch grid_mask
                grid_mask = get_grid_mask(taus, c_type, grid_type)
                # fetch grid indices to be filled
                indices_eta, indices_phi = get_fill_indices(taus, c_type, grid_type, grid_mask)
                # loop over taus in the batch
                for i_tau, tau in enumerate(taus):
                    if i_tau%100 == 0:
                        print(f'        - {i_tau}th tau')
                    if ak.sum(grid_mask[i_tau]) == 0:
                        continue
                    # fetch indices
                    i_eta, i_phi = indices_eta[i_tau], indices_phi[i_tau]
                    # fetch values to be filled
                    values_to_fill = get_fill_values(tau, c_type, fill_branches[c_type], grid_mask[i_tau])
                    # put them in the tensor
                    grid_tensors[grid_type][c_type][i_tau, i_eta, i_phi, :] = values_to_fill
        if (i_batch == n_batches - 1) and (n_batches > 0):
            break

    # release memory
    for c_type in constituent_types:
        for grid_type in grid_types:
            del grid_tensors[grid_type][c_type]
    gc.collect()

################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', action="store", dest="path_to_data", type=str)
    parser.add_argument('--batch_size', action="store", dest="batch_size", type=int, default=1000)
    parser.add_argument('--n_batches', action="store", dest="n_batches", type=int, default=1)
    args = parser.parse_args()
    fill_tensor(args.path_to_data, args.batch_size, args.n_batches)
