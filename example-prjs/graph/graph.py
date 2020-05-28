"""
This module contains the functionality to construct and work with hit graphs.
"""

import logging

from collections import namedtuple

import numpy as np
import pandas as pd


# Global feature details
#feature_names = ['r', 'phi', 'z']
#feature_scale = np.array([1000., np.pi, 1000.])

# Graph is a namedtuple of (X, Ri, Ro, y) for convenience
Graph = namedtuple('Graph', ['X', 'Ri', 'Ro', 'y'])
# Sparse graph uses the indices for the Ri, Ro matrices
SparseGraph = namedtuple('SparseGraph',
        ['X', 'Ri_rows', 'Ri_cols', 'Ro_rows', 'Ro_cols', 'y'])

def make_sparse_graph(X, Ri, Ro, y):
    Ri_rows, Ri_cols = Ri.nonzero()
    Ro_rows, Ro_cols = Ro.nonzero()
    return SparseGraph(X, Ri_rows, Ri_cols, Ro_rows, Ro_cols, y)

def graph_from_sparse(sparse_graph, dtype=np.uint8):
    n_nodes = sparse_graph.X.shape[0]
    n_edges = sparse_graph.Ri_rows.shape[0]
    Ri = np.zeros((n_nodes, n_edges), dtype=dtype)
    Ro = np.zeros((n_nodes, n_edges), dtype=dtype)
    Ri[sparse_graph.Ri_rows, sparse_graph.Ri_cols] = 1
    Ro[sparse_graph.Ro_rows, sparse_graph.Ro_cols] = 1
    return Graph(sparse_graph.X, Ri, Ro, sparse_graph.y)

def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2*np.pi
    dphi[dphi < -np.pi] += 2*np.pi
    return dphi

def select_segments(hits1, hits2, phi_slope_max, phi_slope_mid_max, phi_slope_outer_max, z0_max):
    """
    Construct a list of selected segments from the pairings
    between hits1 and hits2, filtered with the specified
    phi slope and z0 criteria.

    Returns: pd DataFrame of (index_1, index_2), corresponding to the
    DataFrame hit label-indices in hits1 and hits2, respectively.
    """
    # Start with all possible pairs of hits
    keys = ['evtid', 'layer', 'r', 'phi', 'z']
    hit_pairs = hits1[keys].reset_index().merge(
        hits2[keys].reset_index(), on='evtid', suffixes=('_1', '_2'))
    # Compute line through the points
    dphi = calc_dphi(hit_pairs.phi_1, hit_pairs.phi_2)
    dz = hit_pairs.z_2 - hit_pairs.z_1
    dr = hit_pairs.r_2 - hit_pairs.r_1
    phi_slope = dphi / dr
    z0 = hit_pairs.z_1 - hit_pairs.r_1 * dz / dr
    # Filter segments according to criteria
    #good_seg_mask = (phi_slope.abs() < phi_slope_max) & (z0.abs() < z0_max)
    good_seg_mask = (((phi_slope.abs() < phi_slope_max) & (hit_pairs.layer_1[0] < 5)) | ((phi_slope.abs() < phi_slope_outer_max) & (hit_pairs.layer_1[0] >= 5))) & (z0.abs() < z0_max)  
    return hit_pairs[['index_1', 'index_2']][good_seg_mask]

def construct_segments(hits, layer_pairs,
                       phi_slope_max, phi_slope_mid_max, phi_slope_outer_max, z0_max):
    """
    Construct a list of selected segments from hits DataFrame using
    the specified layer pairs and selection criteria.

    Returns: DataFrame of (index_1, index_2) corresponding to the
    hit indices of the selected segments.
    """
    # Loop over layer pairs and construct segments
    layer_groups = hits.groupby('layer')
    segments = []
    for (layer1, layer2) in layer_pairs:
        # Find and join all hit pairs
        try:
            hits1 = layer_groups.get_group(layer1)
            hits2 = layer_groups.get_group(layer2)
        # If an event has no hits on a layer, we get a KeyError.
        # In that case we just skip to the next layer pair
        except KeyError as e:
            logging.info('skipping empty layer: %s' % e)
            continue
        # Construct the segments
        segments.append(select_segments(hits1, hits2, phi_slope_max, phi_slope_mid_max, phi_slope_outer_max, z0_max))
    # Combine segments from all layer pairs
    return pd.concat(segments)

def construct_graph(hits, layer_pairs,
                    phi_slope_max, phi_slope_mid_max, phi_slope_outer_max, z0_max,
                    feature_names, feature_scale,
                    max_tracks=None,
                    no_missing_hits=False):
    """Construct one graph (e.g. from one event)"""

    if no_missing_hits:
        hits = (hits.groupby(['particle_id'])
                .filter(lambda x: len(x.layer.unique()) == 10))
    if max_tracks is not None:           
        particle_keys = hits['particle_id'].drop_duplicates().values
        np.random.shuffle(particle_keys)
        sample_keys = particle_keys[0:max_tracks]
        hits = hits[hits['particle_id'].isin(sample_keys)]

    # Construct segments
    segments = construct_segments(hits, layer_pairs,
                                  phi_slope_max=phi_slope_max,
                                  phi_slope_mid_max=phi_slope_mid_max,
                                  phi_slope_outer_max=phi_slope_outer_max,
                                  z0_max=z0_max)
    n_hits = hits.shape[0]
    n_edges = segments.shape[0]
    #evtid = hits.evtid.unique()
    # Prepare the tensors
    X = (hits[feature_names].values / feature_scale).astype(np.float32)
    Ri = np.zeros((n_hits, n_edges), dtype=np.uint8)
    Ro = np.zeros((n_hits, n_edges), dtype=np.uint8)
    y = np.zeros(n_edges, dtype=np.float32)
    # We have the segments' hits given by dataframe label,
    # so we need to translate into positional indices.
    # Use a series to map hit label-index onto positional-index.
    hit_idx = pd.Series(np.arange(n_hits), index=hits.index)
    seg_start = hit_idx.loc[segments.index_1].values
    seg_end = hit_idx.loc[segments.index_2].values
    # Now we can fill the association matrices.
    # Note that Ri maps hits onto their incoming edges,
    # which are actually segment endings.
    Ri[seg_end, np.arange(n_edges)] = 1
    Ro[seg_start, np.arange(n_edges)] = 1
    # Fill the segment labels
    pid1 = hits.particle_id.loc[segments.index_1].values
    pid2 = hits.particle_id.loc[segments.index_2].values
    y[:] = (pid1 == pid2)
    # Return a tuple of the results
    #print("X:", X.shape, ", Ri:", Ri.shape, ", Ro:", Ro.shape, ", y:", y.shape)
    return make_sparse_graph(X, Ri, Ro, y), segments 
    #return Graph(X, Ri, Ro, y)

def construct_graphs(hits, layer_pairs,
                     phi_slope_max, phi_slope_mid_max, phi_slope_outer_max, z0_max_inner, z0_max_outer,
                     max_events=None, max_tracks=None):
    """
    Construct the full graph representation from the provided hits DataFrame.
    Returns: A list of (X, Ri, Ro, y)
    """
    # Organize hits by event 
    evt_hit_groups = hits.groupby('evtid')
    # Organize hits by event and barcode
    evt_barcode_hit_groups = hits.groupby(['evtid', 'barcode'])
    evtids = hits.evtid.unique()
    if max_events is not None:
        evtids = evtids[:max_events]

    # Loop over events and construct graphs
    graphs = []
    for evtid in evtids:
        # Get all the hits for this event
        evt_hits = evt_hit_groups.get_group(evtid)
        if max_tracks is not None:
            particle_keys = evt_hits['barcode'].drop_duplicates().values
            np.random.shuffle(particle_keys)
            sample_keys = particle_keys[0:max_tracks]
            evt_hits = evt_hits[evt_hits['barcode'].isin(sample_keys)]
            #print('max tracks:', len(sample_keys))
            #print('number of hits:', len(evt_hits))
        graph = construct_graph(evt_hits, layer_pairs,
                                phi_slope_max, phi_slope_mid_max, phi_slope_outer_max, z0_max_inner, z0_max_outer)
        graphs.append(graph)

    # Return the results
    return graphs

def save_graph(graph, filename):
    """Write a single graph to an NPZ file archive"""
    np.savez(filename, **graph[0]._asdict())
    #np.savez(filename, X=graph.X, Ri=graph.Ri, Ro=graph.Ro, y=graph.y)

def save_graphs(graphs, filenames):
    for graph, filename in zip(graphs, filenames):
        save_graph(graph, filename)

def load_graph(filename, graph_type=Graph):
    """Reade a single graph NPZ"""
    with np.load(filename) as f:
        return graph_type(**dict(f.items()))

def load_graphs(filenames, graph_type=Graph):
    return [load_graph(f, graph_type) for f in filenames]
 

