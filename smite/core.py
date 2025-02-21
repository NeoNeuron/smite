# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import hankel
from itertools import permutations
from sklearn.utils.parallel import Parallel, delayed
from typing import List, Union

def generate_permutations(sequence):
    """
    Generate all permutations of the given sequence

    """
    return [''.join(np.asarray(item).astype(str)) for item in permutations(sequence, len(sequence))]

def sym2int(sym):
    """
    Convert a symbolic str representation to an integer representation.

    Args:
        sym (numpy.ndarray): The symbolic representation to be converted.

    Returns:
        tuple: A tuple containing the size of the integer representation and the converted integer representation.

    """
    perms = generate_permutations(np.arange(len(sym[0])))
    size_X = len(perms)
    mapping_dict = {key: value for key, value in zip(perms, np.arange(len(perms)))}
    mapper = np.vectorize(lambda x: mapping_dict.get(x, -1))
    mappedX = mapper(sym)
    return size_X, mappedX

def symbolize(X:np.ndarray, m:int=3):
    """
    Converts numeric values of the series to a symbolic version of it based
    on the m consecutive values.
    
    Parameters
    ----------
    X : 1-d numpy array to symbolize.
    m : length of the symbolic subset.
    
    Returns
    ----------
    List of symbolized X

    """
    
    if m >= len(X):
        raise ValueError("Length of the series must be greater than m")
    
    dummy = hankel(X[:m],X[m-1:]).T

    yy = np.argsort(dummy, axis=1)
    symX = np.zeros_like(dummy).astype(int)
    indices = np.tile(np.arange(m), (dummy.shape[0], 1))
    xx = np.tile(np.arange(0, dummy.shape[0]), (dummy.shape[1], 1)).astype(int).T
    symX[xx.flatten(), yy.flatten()] = indices.flatten()
    symX = np.asarray([''.join(line) for line in symX.astype(str)])

    return symX

def symbolic_mutual_information(symX, symY):
    """
    Computes the symbolic mutual information between symbolic series X and 
    symbolic series Y.
    
    Parameters
    ----------
    symX : Symbolic series X.
    symY : Symbolic series Y.
    
    Returns
    ----------
    Value for mutual information

    """

    if len(symX) != len(symY):
        raise ValueError('All arrays must have same length')
        
    symX = np.array(symX)
    symY = np.array(symY)

    # mapping symbols to integers
    size_X, mappedX = sym2int(symX)
    size_Y, mappedY = sym2int(symY)

    # P(x[n], y[n])
    pxy,_,_ = np.histogram2d(
        mappedX, mappedY,
        bins=(size_X, size_Y),
        range=((-0.5,size_X-0.5),(-0.5,size_Y-0.5)),
        density=True)

    px = pxy.sum(axis=1)  # P(x[n])
    py = pxy.sum(axis=0)  # P(y[n])
    
    MI = 0
    
    for i in range(pxy.shape[0]):
        for j in range(pxy.shape[1]):
            if pxy[i,j] > 0:
                MI += pxy[i,j] * np.log(pxy[i,j] / (px[i] * py[j]))
    return MI


def symbolic_transfer_entropy_matrix(X:np.ndarray, m:int=1, n_jobs:int=1) -> np.ndarray:
    """
    Computes T(Y->X), the transfer of entropy from symbolic series Y to X.
    
    Parameters
    ----------
    X : time series, shape (n_time_points, n_nodes).
    m : order of time delayed embedding.
    n_jobs: number of parallel jobs.
    
    Returns
    ----------
    matrix of symbolic transfer entropy

    """

    if n_jobs > 1:
        # symbolize the data
        if n_jobs >= X.shape[1]:
            job_seq = Parallel(n_jobs=X.shape[1])
        else:
            job_seq = Parallel(n_jobs=n_jobs)
        symX = np.asarray(job_seq(delayed(symbolize)(row, m=m) for row in X.T))
        # mapping symbols to integers
        results = job_seq(delayed(sym2int)(row) for row in symX)
        size_X = np.asarray([item[0] for item in results])
        mappedX = [item[1] for item in results]

        N = X.shape[1]
        x_ids, y_ids = np.meshgrid(np.arange(N), np.arange(N))
        size_X, size_Y = np.meshgrid(size_X, size_X)
        mask = (1-np.eye(N)).astype(bool)
        job_seq = Parallel(n_jobs=n_jobs)
        ste_list = np.asarray(
            job_seq(delayed(_symbolic_transfer_entropy)(
                    x_s, y_s, mappedX[x_id], mappedX[y_id]
                ) for x_id, y_id, x_s, y_s in zip(
                    x_ids[mask], y_ids[mask], size_X[mask], size_Y[mask])
            )
        )
        ste = np.zeros((N,N))
        ste[mask] = ste_list
    else:
        # symbolize the data
        symX = np.asarray([symbolize(row, m=m) for row in X.T])
        # mapping symbols to integers
        results = [sym2int(row) for row in symX]
        size_X = np.asarray([item[0] for item in results])
        mappedX = [item[1] for item in results]

        N = X.shape[1]
        x_ids, y_ids = np.meshgrid(np.arange(N), np.arange(N))
        size_X, size_Y = np.meshgrid(size_X, size_X)
        mask = (1-np.eye(N)).astype(bool)
        ste_list = np.asarray(
            [_symbolic_transfer_entropy(
                    x_s, y_s, mappedX[x_id], mappedX[y_id]
                ) for x_id, y_id, x_s, y_s in zip(
                    x_ids[mask], y_ids[mask], size_X[mask], size_Y[mask])
            ]
        )
        ste = np.zeros((N,N))
        ste[mask] = ste_list

    return ste


def symbolic_transfer_entropy(symX:Union[List, np.ndarray], symY:Union[List, np.ndarray]):
    """
    Computes T(Y->X), the transfer of entropy from symbolic series Y to X.
    
    Parameters
    ----------
    symX : 1-d Symbolic series X.
    symY : 1-d Symbolic series Y.
    
    Returns
    ----------
    Value for symbolic transfer entropy

    """

    if len(symX) != len(symY):
        raise ValueError('All arrays must have same length')
        
    symX = np.array(symX)
    symY = np.array(symY)

    # mapping symbols to integers
    size_X, mappedX = sym2int(symX)
    size_Y, mappedY = sym2int(symY)
    return _symbolic_transfer_entropy(size_X, size_Y, mappedX, mappedY)


def _symbolic_transfer_entropy(size_X, size_Y, mappedX, mappedY):
    """
    Computes T(Y->X), the transfer of entropy from symbolic series Y to X.
    
    Parameters
    ----------
    size_X : size of X's symbolic space.
    size_Y : size of Y's symbolic space.
    mappedX : mapped symbolic series X.
    mappedY : mapped symbolic series Y.
    
    Returns
    ----------
    Value for mutual information

    """

    # P(x[n+1], x[n], y[n])
    pxxy,_ = np.histogramdd(
        np.vstack((mappedX[1:], mappedX[:-1], mappedY[:-1])).T,
        bins=(size_X, size_X, size_Y),
        range=((-0.5,size_X-0.5), (-0.5,size_X-0.5), (-0.5,size_Y-0.5)),
        density=True)

    pxy = pxxy.sum(axis=0)  # P(x[n], y[n])
    pxx = pxxy.sum(axis=2)  # P(x[n+1], x[n])
    px  = pxx.sum(axis=0)   # P(x[n])
    
    TE = 0
    
    for i in range(pxxy.shape[0]):
        for j in range(pxxy.shape[1]):
            for k in range(pxxy.shape[2]):
                if pxxy[i,j,k] > 0:
                    TE += pxxy[i,j,k] * np.log2(pxxy[i,j,k] * px[j] 
                                                / (pxx[i,j] * pxy[j,k]))
    return TE
