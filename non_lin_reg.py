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

surr_model = SurrogateModel()  # object definition for calling class defined functions


# Model 1 for non linear regression
def mod1(x, b1, b2, b3, b4):
    y_ = b1 + b2 * x**2 * np.sin(b3 * x + b4)
    return y_
# Model 2 for non linear regression
def mod2(x, b1, b2, b3, b4):
    y_ = (b1 * x + b2)**2 * np.sin(b3 * x + b4)
    return y_

def main():

    # Output directories for saving plots
    output_dir = "outputs/Plots_non_lin_reg/"

    # Create subplots
    fig, axs = plt.subplots(3,3, figsize=(18, 12), sharex=False, sharey=False)
    fig.suptitle('Non-Linear Regression for Different Sampling Strategies and Sizes', fontsize=16) # common title of all the plots

    # obtain data for model development
    i = 0
    for sampler_case in [1,2,3]: # loop through the various sampling methods
        j = 0
        for ns in [5,10,15]: # loop through sample sizes
            print(f"Sampling Case: {sampler_case}, Sample Size: {ns}")
            y_sample,x_sample = surr_model.sampler_func(ns,sampler_case)
            """
            obtain sampling points to fit model 
            """

            # Fit using Model 1
            popt1, _ = curve_fit(mod1, x_sample, y_sample, p0=[1, 1, 10, -1] , maxfev=100000)
            # Fit using Model 2
            popt2, _ = curve_fit(mod2, x_sample, y_sample, p0=[1, 1, 10, -1] , maxfev=100000)

            x_dense = np.linspace(0, 1, 100)

            print(popt1)
            print(popt2)
            # true function
            y_dense_true = surr_model.true_func(x_dense)
            # Predictions using fit model
            y_dense_pred1 = mod1(x_dense, *popt1)
            y_dense_pred2 = mod2(x_dense, *popt2)

            MSE1 = round(np.mean((y_dense_true - y_dense_pred1)**2),4)
            MSE2 = round(np.mean((y_dense_true - y_dense_pred2)**2),4)

            # Plotting the results
            axs[i,j].plot(x_sample, y_sample, '*', markersize=4, label='sampled points') # scatter of samples
            axs[i,j].plot(x_dense, y_dense_true, label='True Function')
            axs[i,j].plot(x_dense, y_dense_pred1, label=f'Model 1(MSE = {MSE1})')
            axs[i,j].plot(x_dense, y_dense_pred2, label=f'Model 2(MSE = {MSE2})')
            # plt.plot(x_sample, y_pred2, label='Model 2 Fit')
            axs[i,j].legend()
            axs[i,j].set_xlabel('x')
            axs[i,j].set_ylabel('y')
            axs[i,j].set_title(f'Sample Size: {ns}, Sampling Method: {sampler_case}')
            axs[i,j].grid(True)


            j+=1
            if j==3:
                break # finished looping through sample sizes
        i+=1
        if i==3:
            break # finished looping through sampling methods

    fig.show()
    # # Saving the plot to file
    file_name = "non__lin_reg.png"
    file_path = os.path.join(output_dir, file_name) # creating full paths name
    fig.savefig(file_path, bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    main()

