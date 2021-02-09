#!/usr/bin/env python

import uproot
import awkward as ak
import numpy as np
import pandas as pd
import numba as nb
import matplotlib.pyplot as plt

import gc
import argparse
from memory_profiler import profile

from utils import add_vars_to_taus, sort_constituents_by_var, get_batch_yielder, derive_grid_mask

################################################################################################

# constituent info
constituent_types = ['ele', 'muon', 'pfCand']

# branches to be stored
# note a possible multiplicative factor in front of the list
fill_branches = {'ele': 2*['pt',
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
                 'muon': 2*['pt',
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
}

# defining grids
grid_types = ['inner', 'outer']
grid_dim = ['eta', 'phi']
grid_size, grid_left, grid_right = {}, {}, {}

n_cells = {'inner': 11, 'outer': 21}
cell_size = {'inner': 0.02, 'outer': 0.05}

for grid_type in grid_types:
    grid_size[grid_type] = cell_size[grid_type] * n_cells[grid_type]
    grid_left[grid_type], grid_right[grid_type] = - grid_size[grid_type] / 2, grid_size[grid_type] / 2

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
def fill_tensor(path_to_data, batch_size, n_batches):
    """
    Main function which loops over
    batches of taus -> types of constituents -> types of grid ->  feature list
    and fills the output tensor via `fill_feature_tensor()`
    """
    grid_tensors = {key: {} for key in grid_types} # dictionary to store final tensors
    batch_yielder = get_batch_yielder(path_to_data, 'taus', batch_size)
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
