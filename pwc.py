# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 00:54:59 2025

@author: adi2l
"""
import os # utilized in plotting section
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from main import SurrogateModel
from scipy.interpolate import CubicSpline


surr_model = SurrogateModel()  # object definition for calling class defined functions



def main():

    # Output directories for saving plots
    output_dir = "outputs/Plots_pw_cubic_spline/"

    # Create subplots
    fig, axs = plt.subplots(3,3, figsize=(18, 12), sharex=False, sharey=False)
    fig.suptitle('Piecewise Cubic Spline for Different Sampling Strategies and Sizes', fontsize=16) # common title of all the plots

    # obtain data for model development
    case = 0
    for sampler_case in [1,2,3]: # loop through the various sampling methods
        size = 0
        for ns in [5,10,15]: # loop through sample sizes
            print(f"Sampling Case: {sampler_case}, Sample Size: {ns}")

            y_sample, x_sample = surr_model.sampler_func(ns,sampler_case) # obtain sampling points to fit model
            # print(x_sample)
            # print(y_sample)
            x_dense = np.linspace(0, 1, 100) # dense vector to test model

       # implement cs algo

            # intervals and spline coefficients
            n_interv = ns - 1
            interv = np.diff(x_sample) # obtaining intervals between subsequent x points 

            """
                alpha for natural spline - 
                alpha signifies the difference in slopes at the boundaries of the intervals  
            """
            alpha = np.zeros(n_interv - 1) # initialize alpha vector - 2 adjacent intervals share an alpha
            for i in range(1, n_interv):
                alpha[i - 1] = (3/interv[i])*(y_sample[i+1] - y_sample[i]) - (3/interv[i-1])*(y_sample[i] - y_sample[i-1])
            # slope difference calculation
            """
            for two adjacent intervals i, i-1
            """

            # Tridiagonal matrix to solve for c coefficients
            """
                the c_i are the coefficients of the quadratic term...
            ...and they are important because they are related to the second derivative of the spline
            """
            A = np.zeros((n_interv - 1, n_interv - 1))
            for i in range(n_interv - 1):
                A[i, i] = 2 * (interv[i] + interv[i+1])
                if i > 0:
                    A[i, i-1] = interv[i]
                if i < n_interv - 2:
                    A[i, i+1] = interv[i+1]

            c_inner = np.linalg.solve(A, alpha)
            c = np.concatenate(([0], c_inner, [0])) # second derivatives = 0 at the ends of the spline

            # solving for coefficients of the cubic and linear terms
            a = y_sample[:-1]

            # initialize b and d arrays
            b = np.zeros(n_interv)
            d = np.zeros(n_interv)

            for i in range(n_interv):
                b[i] = (y_sample[i+1] - y_sample[i])/interv[i] - interv[i]*(2*c[i] + c[i+1])/3
                d[i] = (c[i+1] - c[i]) / (3*interv[i])

            # evaluating using fit curve

            y_dense_cs = np.array([evaluate_spline(xi, x_sample, a, b, c, d) for xi in x_dense])

            # true function evaluation
            y_dense_true = surr_model.true_func(x_dense)

            # calculation of MSO to determine fit accuracy
            mse = round(np.mean((y_dense_true - y_dense_cs)**2),4)

            # Plotting the results
            axs[case,size].plot(x_sample, y_sample, '*', markersize=4, label='sampled points') # scatter of samples
            axs[case,size].plot(x_dense, y_dense_true, label='True Function')
            axs[case,size].plot(x_dense, y_dense_cs, label=f'Piece-wise cubic spline(MSE = {mse})')

            # plt.plot(x_sample, y_pred2, label='Model 2 Fit')
            axs[case,size].legend()
            axs[case,size].set_xlabel('x')
            axs[case,size].set_ylabel('y')
            axs[case,size].set_title(f'Sample Size: {ns}, Sampling Method: {sampler_case}')
            axs[case,size].grid(True)


            size+=1
            if size>=3:
                break # finished looping through sample sizes
        case+=1
        if case>=3:
            break # finished looping through sampling methods

    fig.show()
    # # Saving the plot to file
    file_name = "piecewise_cubic.png"
    file_path = os.path.join(output_dir, file_name) # creating full paths name
    fig.savefig(file_path, bbox_inches='tight', dpi=300)


def evaluate_spline(xi, x_sample, a, b, c, d):
    i = np.searchsorted(x_sample, xi) - 1
    i = np.clip(i, 0, len(a) - 1)
    dx = xi - x_sample[i]
    return a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3



if __name__ == "__main__":
    main()


#
#