# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 00:49:45 2025

@author: adi2l
"""
#from pyDOE2 import lhs

import numpy as np
from scipy.stats import qmc # for LHS implementation



class SurrogateModel:
    # defining the 1d design space
    stp = 0  # start point   
    endp = 1  # end point 
    
    # defining the true one-dimensional non-linear function

    @staticmethod
    def true_func(x): 
        y = (6*x - 2)**2*np.sin(12*x - 4)
        return y

    """
    sampling the design space -
    input: ms, sampler_case -  number of sample points, sampling case
    output: y,x according to sampling requirements  
    """

    @staticmethod
    def sampler_func(ns,sampler_case):
        if sampler_case==1:
            # implementing equi-spaced sampling 
            x = np.linspace(SurrogateModel.stp, SurrogateModel.endp, ns) # sampling the design space for given number of sample points

        elif sampler_case==2:
            """ 
            implementing random sampling - this samples design points for each design variable (in this case 1) from a distribution - 
            repeat for required number of sample points 
            """
            x = np.random.uniform(0,1,ns) # this randomly samples ns points from a uniform distribution between 0 and 1

        elif sampler_case==3:
            # latin hypercube sampling
            x = SurrogateModel.lhs_sampling(ns)

        else: # incorrect option handling
            raise ValueError("Incorrect sampling case provided. Valid cases are 1 (Equi-spaced), 2 (Random), 3 (LHS).")

        
        y = SurrogateModel.true_func(x)
        return y, x

    # implementing LHS
    @staticmethod
    def lhs_sampling(ns):
        # object called sampler created
        sampler = qmc.LatinHypercube(d=1)  # 1D LHS
        sample = sampler.random(ns)  # Generate ns samples
        return np.sort(sample.flatten())  # Flatten to 1D array and sort the samples






