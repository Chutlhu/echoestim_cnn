"""
A little collection of function used for evaluation
"""
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision = 2, suppress=True) 

def nrmse(x, x_ref):

    return np.sqrt(np.sum((x - x_ref)**2, axis = 0))\
           /np.sqrt(np.sum((x_ref - np.mean(x_ref, axis = 0))**2, axis = 0))

def dist(x, x_ref):
    return np.abs(x-x_ref)

def hmean(a):
    return len(a) / np.sum(1.0/a) 

def accuracy(x, x_ref, tollerance):
    if x.shape[2] > 1:
        # if x and x_ref are a multimensional variables
        N = np.max(np.shape(x_ref))
        return np.mean(np.sum(np.abs(x-x_ref) < tollerance, axis=2)/N)
    # otherwise...
    N = np.max(np.shape(x_ref))
    return np.sum(np.abs(x-x_ref) < tollerance)/N

def plot_regression_scatter(x,x_ref, axes = None):
    plt.cla()
    if axes is None:
        axes = plt.gca()
    plt.scatter(x_ref, x)
    axes.set_xlim([np.min(x_ref),np.max(x_ref)])
    axes.set_ylim([np.min(x_ref),np.max(x_ref)])
    plt.plot(x_ref, x_ref, color='red')

def ordinal_accuracy(x, x_ref, tol=0.5, tol_class = 0):
    N, C = x.shape
    results = np.zeros(N)
    x = x > tol
    x_ref = x_ref > tol

    print(x.shape)

    r = np.logical_not(np.logical_xor(x,x_ref));
    r = np.sum(r, axis = 1)
    r = r > C - tol_class-1
    acc = np.sum(r, axis = 0)/N
    return acc


