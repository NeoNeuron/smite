# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import hankel
from itertools import permutations

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

def symbolize(X, m):
    """
    Converts numeric values of the series to a symbolic version of it based
    on the m consecutive values.
    
    Parameters
    ----------
    X : Series to symbolize.
    m : length of the symbolic subset.
    
    Returns
    ----------
    List of symbolized X

    """
    
    X = np.asarray(X)

    if m >= len(X):
        raise ValueError("Length of the series must be greater than m")
    
    dummy = hankel(X[:m],X[m-1:]).T

    yy = np.argsort(dummy, axis=1)
    symX = np.zeros_like(dummy).astype(int)
    indices = np.tile(np.arange(m), (dummy.shape[0], 1))
    xx = np.tile(np.arange(0, dummy.shape[0]), (dummy.shape[1], 1)).astype(int).T
    symX[xx.flatten(), yy.flatten()] = indices.flatten()
    symX = [''.join(line) for line in symX.astype(str)]

    return symX

def symbolic_mutual_information_old(symX, symY):
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
    
    symbols = np.unique(np.concatenate((symX,symY))).tolist()
        
    jp = symbolic_joint_probabilities(symX, symY)
    pX = symbolic_probabilities(symX)
    pY = symbolic_probabilities(symY)
    
    MI = 0

    for yi in list(pY.keys()):
        for xi in list(pX.keys()):         
            a = pX[xi]
            b = pY[yi]
            
            try:
                c = jp[yi][xi]
                MI += c * np.log(c /(a * b))
            except KeyError:
                continue
            except:
                print("Unexpected Error")
                raise
        
    return MI

def symbolic_transfer_entropy_old(symX, symY):
    """
    Computes T(Y->X), the transfer of entropy from symbolic series Y to X.
    
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
        
    cp = symbolic_conditional_probabilities_consecutive(symX)
    cp2 = symbolic_conditional_probabilities_consecutive_external(symX, symY)
    jp = symbolic_joint_probabilities_consecutive_external(symX, symY)
    
    TE = 0
    
    for yi in list(jp.keys()):
        for xi in list(jp[yi].keys()):
            for xii in list(jp[yi][xi].keys()):

                try:
                    a = cp[xi][xii]
                    b = cp2[yi][xi][xii]
                    c = jp[yi][xi][xii]
                    TE += c * np.log(b / a) / np.log(2.);
                except KeyError:
                    continue
                except:
                    print("Unexpected Error")
                    raise
    del cp
    del cp2
    del jp
    
    return TE

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

def symbolic_transfer_entropy(symX, symY):
    """
    Computes T(Y->X), the transfer of entropy from symbolic series Y to X.
    
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

    # P(x[n+1], x[n], y[n])
    pxxy,_,_ = np.histogramdd(
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

def symbolic_probabilities(symX):
    """
    Computes the conditional probabilities where M[A][B] stands for the
    probability of getting B after A.
    
    Parameters
    ----------
    symX : Symbolic series X.
    symbols: Collection of symbols. If "None" calculated from symX
    
    Returns
    ----------
    Matrix with conditional probabilities

    """

    symX = np.array(symX)
    
    # initialize
    p = {}
    n = len(symX)

    for xi in symX:
        if xi in p: 
            p[xi] += 1.0 / n
        else:
            p[xi] = 1.0 / n
    
    return p

def symbolic_joint_probabilities(symX, symY):
    """
    Computes the joint probabilities where M[yi][xi] stands for the
    probability of ocurrence yi and xi.
    
    Parameters
    ----------
    symX : Symbolic series X.
    symY : Symbolic series Y.
    symbols: Collection of symbols. If "None" calculated from symX
    
    Returns
    ----------
    Matrix with joint probabilities

    """

    if len(symX) != len(symY):
        raise ValueError('All arrays must have same length')
        
    symX = np.array(symX)
    symY = np.array(symY)
    
    # initialize
    jp = {}
    n = len(symX)

    for yi, xi in zip(symY,symX):
        if yi in jp:
            if xi in jp[yi]:
                jp[yi][xi] += 1.0 / n
            else:
                jp[yi][xi] = 1.0 / n
        else:
            jp[yi] = {}
            jp[yi][xi] = 1.0 / n
    
    return jp

def symbolic_conditional_probabilities(symX, symY):
    """
    Computes the conditional probabilities where M[A][B] stands for the
    probability of getting "B" in symX, when we get "A" in symY.
    
    Parameters
    ----------
    symX : Symbolic series X.
    symY : Symbolic series Y.
    
    Returns
    ----------
    Matrix with conditional probabilities

    """
    
    if len(symX) != len(symY):
        raise ValueError('All arrays must have same length')
        
    symX = np.array(symX)
    symY = np.array(symY)
    
    # initialize
    cp = {}
    n = {}

    for xi, yi in zip(symX,symY):
        if yi in cp:
            n[yi] += 1
            if xi in cp[yi]:
                cp[yi][xi] += 1.0
            else:
                cp[yi][xi] = 1.0
        else:
            cp[yi] = {}
            cp[yi][xi] = 1.0
            
            n[yi] = 1

        
    for yi in list(cp.keys()):
        for xi in list(cp[yi].keys()):
            cp[yi][xi] /= n[yi]
    
    return cp

def symbolic_conditional_probabilities_consecutive(symX):
    """
    Computes the conditional probabilities where M[A][B] stands for the
    probability of getting B after A.
    
    Parameters
    ----------
    symX : Symbolic series X.
    symbols: Collection of symbols. If "None" calculated from symX
    
    Returns
    ----------
    Matrix with conditional probabilities

    """

    symX = np.array(symX)

    cp = symbolic_conditional_probabilities(symX[1:],symX[:-1])
    
    return cp

def symbolic_double_conditional_probabilities(symX, symY, symZ):
    """
    Computes the conditional probabilities where M[y][z][x] stands for the
    probability p(x|y,z).
    
    Parameters
    ----------
    symX : Symbolic series X.
    symY : Symbolic series Y.
    symZ : Symbolic series Z.
    
    Returns
    ----------
    Matrix with conditional probabilities

    """

    if (len(symX) != len(symY)) or (len(symY) != len(symZ)):
        raise ValueError('All arrays must have same length')
        
    symX = np.array(symX)
    symY = np.array(symY)
    symZ = np.array(symZ)
    
    # initialize
    cp = {}
    n = {}

    for x, y, z in zip(symX,symY,symZ):
        if y in cp:
            if z in cp[y]:
                n[y][z] += 1.0
                if x in cp[y][z]:
                    cp[y][z][x] += 1.0
                else:
                    cp[y][z][x] = 1.0
            else:
                cp[y][z] = {}
                cp[y][z][x] = 1.0
                n[y][z] = 1.0
        else:
            cp[y] = {}
            n[y] = {}
            
            cp[y][z] = {}
            n[y][z] = 1.0
            
            cp[y][z][x] = 1.0

        
    for y in list(cp.keys()):
        for z in list(cp[y].keys()):
            for x in list(cp[y][z].keys()):
                cp[y][z][x] /= n[y][z]
    
    return cp

def symbolic_conditional_probabilities_consecutive_external(symX, symY):
    """
    Computes the conditional probabilities where M[yi][xi][xii] stands for the
    probability p(xii|xi,yi), where xii = x(t+1), xi = x(t) and yi = y(t). 
    
    Parameters
    ----------
    symX : Symbolic series X.
    symY : Symbolic series Y.
    symbols: Collection of symbols. If "None" calculated from symX
    
    Returns
    ----------
    Matrix with conditional probabilities

    """

    if len(symX) != len(symY):
        raise ValueError('All arrays must have same length')
        
    symX = np.array(symX)
    symY = np.array(symY)

    cp = symbolic_double_conditional_probabilities(symX[1:],symY[:-1],symX[:-1])
    
    return cp

def symbolic_joint_probabilities_triple(symX, symY, symZ):
    """
    Computes the joint probabilities where M[y][z][x] stands for the
    probability of coocurrence y, z and x p(y,z,x).
    
    Parameters
    ----------
    symX : Symbolic series X.
    symY : Symbolic series Y.
    symZ : Symbolic series Z.
    
    Returns
    ----------
    Matrix with joint probabilities

    """
    
    if (len(symX) != len(symY)) or (len(symY) != len(symZ)):
        raise ValueError('All arrays must have same length')

    symX = np.array(symX)
    symY = np.array(symY)
    symZ = np.array(symZ)
    
    # initialize
    jp = {}
    n = len(symX)

    for x, y, z in zip(symX,symY,symZ):
        if y in jp:
            if z in jp[y]:
                if x in jp[y][z]:
                    jp[y][z][x] += 1.0 / n
                else:
                    jp[y][z][x] = 1.0 / n
            else:                
                jp[y][z] = {}
                jp[y][z][x] = 1.0 / n
        else:
            jp[y] = {}
            jp[y][z] = {}
            jp[y][z][x] = 1.0 / n
    
    return jp

def symbolic_joint_probabilities_consecutive_external(symX, symY):
    """
    Computes the joint probabilities where M[yi][xi][xii] stands for the
    probability of ocurrence yi, xi and xii.
    
    Parameters
    ----------
    symX : Symbolic series X.
    symY : Symbolic series Y.
    symbols: Collection of symbols. If "None" calculated from symX
    
    Returns
    ----------
    Matrix with joint probabilities

    """
    
    if len(symX) != len(symY):
        raise ValueError('All arrays must have same length')

    symX = np.array(symX)
    symY = np.array(symY)
    
    jp = symbolic_joint_probabilities_triple(symX[1:],symY[:-1],symX[:-1])
    
    return jp
