#!/bin/env python

###Use this template to create the functions for Assignemt 4.
### You can later import them into your project if you wish!
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def load_data_from_csv(csv_file='../../Datasets/x-vs-y-for-PCA.csv'):
    '''Uses pandas to read a csv file into a data object
    '''
    data=None
    return data

def standardize_data(data):
    """Standardizes a 2D dataset using the 'standardize()' function 
    that you need to implement!
    """
    standard_data=None
    return standard_data

def standardize(x):
    """Standardizes the data by subtracting mean and dividing by std
    """
    weighed_x=None
    return weighed_x

def get_cov_data(data):
    '''Returns the covariance matrix from a 2D dataset
    '''
    cov=None
    return cov


def get_covar_eigenvectors(cov_mtrx):
    '''Returns the eigen_vectors and eigenvals of 
       the covariance matrix
    '''
    lam, v = None
    return {'lam':lam,'v':v}

def get_PCS_vectors(xw,yw,v):
    '''Projects weighed x and y values (xw,yw) onto the eigenvectors

    '''
    pcs = np.zeros((2, len(xw)))

    return pcs

def get_1_projection(data,ev1):
    """Projects data onto one eigenvalue
    """
    xw=data['x']
    yw=data['y']
    pc1 = np.zeros((len(xw)))

    return pc1
