# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 00:54:44 2025

@author: adi2l
"""
import os # utilized in plotting section
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from main import SurrogateModel

plt.close('all') # close all plot objects

surr_model = SurrogateModel()  # object definition for using the functions defined within SurrogateModel class

# poly basis - this creates a vector with the terms of the polynomial of required degree 
def poly_basis(x_sample,degree): # input x and the highest degree of the polynomial to fit to
    # poly = np.array([]) # array saving polynomial terms (initalized with 1)
    poly = [1]
    m = 1
    while m<=degree:
        poly.append(x_sample**m) # append the polynomial term of current degree in array of polynomial terms
        m+=1
    return np.array(poly)

def lin_reg_poly(x_sample,y_sample,degree):
        """""
             polynomial basis fitting - this function obtains the vector of coefficients for fitting the sample points
             to a polynomial basis of given degree
             input - x, y coordinates of sample points
             output - retturn vector of coefficients
        """
        # construction of the basis matrix - size of 2 d matrix phi = (ns,degree)
        phi = np.array([poly_basis(xi,degree) for xi in x_sample]) # create a polynomial basis for each x coordinate from sampled lot
        """"" 
            phi is multiplied with the weights vector to be solved for as x in -
             phi@cheb = y_sample
        """
        phi_pinv = np.linalg.pinv(phi)
        # pseudo inverse matrix 
        """"
         using the Moore-Penrose psudo-inverse 
         is the same as obtaining the least-square solutuion
        """
        # solving for the weights
        poly = phi_pinv @ y_sample # @ signifies matrix multiplication
        return poly # return corresponding weight vector

# chebyshev basis - this creates a vector with the terms of the chebyshev function of required degree
def chebyshev_basis_recursive(x_sample, degree):
    cheby = [1, x_sample]
    for n in range(2, degree + 1): # recursion
        temp = 2 * x_sample * cheby[-1] - cheby[-2] # using last two elements of basis
        cheby.append(temp)
    return np.array(cheby)

# Chebyshev polynomial basis
def lin_reg_cheby(x_sample,y_sample,degree):
        """
             Chebyshev basis fitting - this function obtains the vector of coefficients for fitting the sample points
             to a Chebyshev basis of given degree
             input - x, y coordinates of sample points
             output - retturn vector of coefficients
        """
        # cheb = [] # vector of weights/term coefficients

        # construction of the basis matrix
        phi = np.array([chebyshev_basis_recursive(xi,degree) for xi in x_sample])    # size of 2 d matrix phi = (ns,degree)

        """"" 
            phi is multiplied with the weights vector to be solved for as x in -
             phi@cheb = y_sample
        """

        # obtaining the coefficient matrix (beta)
        phi_pinv = np.linalg.pinv(phi)
        """" 
            using the Moore-Penrose psudo-inverse
            is the same as obtaining least sqauare solution
        """""

        # solving for the weights by obtaining the least norm
        cheb = phi_pinv @ y_sample # @ signifies matrix multipication
        return cheb # return corresponding weight vector

def sine_basis(x_sample,degree):# this creates a vector with the terms of the sine series of required harmonic
    sine = [1]# # the array of sine basis terms
    m = 1
    while m<=degree:
        sine.append(np.sin(np.pi * m * x_sample))  # append sine basis array with the current sine term#         m+=1
        m+=1
    return np.array(sine)


def lin_reg_sine(x_sample,y_sample,degree):
        """
             sine basis fitting - this function obtains the vector of coefficients for fitting the sample points
             to a polynomial basis of given degree
             input - x, y coordinates of sample points
             output - retturn vector of coefficients
        """
        # construction of the basis matrix
        phi = np.array([sine_basis(xi,degree) for xi in x_sample]) # size of 2 d matrix phi = (ns,degree)
        # phi is multiplied with the weights vector to be solved for as x in - phi@cheb = y_sample
        phi_pinv = np.linalg.pinv(phi) # using the Moore-Penrose psudo-inverse - same as obtaining least sqaure solution
        # solving for the weights by obtaining the least norm
        sine = phi_pinv @ y_sample
        return sine # return corresponding weight/coefficient vector

def poly_b():
    # Output directories for saving plots
    output_dir = "outputs/Plots_lin_reg/Poly_basis/"

    # Create subplots
    fig, axs = plt.subplots(3,3, figsize=(18, 12), sharex=False, sharey=False)
    fig.suptitle('Linear Regression using Polynomial Bases', fontsize=16) # common title of all the plots

    # obtain data for model development
    i = 0
    for sampler_case in [1,2,3]: # loop through the various sampling methods
        j = 0
        for ns in [5,10,15]: # loop through sample sizes
            print(f"Sampling Case: {sampler_case}, Sample Size: {ns}") # Display current combination of sampling size and method

            # obtain sample points for the given combination of sample size and method
            y_sample,x_sample = surr_model.sampler_func(ns,sampler_case)

            # fit  chebyshev bases using sampled points

            poly2 = lin_reg_poly(x_sample,y_sample,2) # Fit using 2 degree Polynomial basis
            poly3 = lin_reg_poly(x_sample,y_sample,3) # Fit using 3 degree polynomial basis
            poly4 = lin_reg_poly(x_sample,y_sample,4) # Fit using 4 degree polynomial basis

            print(poly2) # print weight vector for 2 degree polynomial function
            print(poly3) # "" 3
            print(poly4) # "" 4

            x_dense = np.linspace(0, 1, 100) # create a dense vector of x-coordinates for plotting

            # true function
            y_dense_true = surr_model.true_func(x_dense) # plot the true function
            # Predictions using fit bases

            y_dense_poly2 = np.array([np.sum(poly_basis(xi,2)*poly2) for xi in x_dense]) # size of 2 d matrix phi = (ns,degree)
            y_dense_poly3 = np.array([np.sum(poly_basis(xi,3)*poly3) for xi in x_dense])
            y_dense_poly4 = np.array([np.sum(poly_basis(xi,4)*poly4) for xi in x_dense])


            # compute mean sqaure error for each model
            mse2 = round(np.mean((y_dense_true - y_dense_poly2)**2),4)
            mse3 = round(np.mean((y_dense_true - y_dense_poly3)**2),4)
            mse4 = round(np.mean((y_dense_true - y_dense_poly4)**2),4)

            # Plotting the results
            axs[i,j].plot(x_sample, y_sample, '*', markersize=4, label='sampled points') # scatter of samples
            axs[i,j].plot(x_dense, y_dense_true, label='True Function')
            axs[i,j].plot(x_dense, y_dense_poly2, label=f'2 degree polynomial (MSE = {mse2})')
            axs[i,j].plot(x_dense, y_dense_poly3, label=f'3 degree polynomial(MSE = {mse3})')
            axs[i,j].plot(x_dense, y_dense_poly4, label=f'4 degree polynomial(MSE = {mse4})')

            axs[i,j].legend()
            axs[i,j].set_xlabel('x')
            axs[i,j].set_ylabel('y')
            axs[i,j].set_title(f'Sample Size: {ns}, Sampling Method: {sampler_case}')
            axs[i,j].grid(True)

            j+=1
            if j==3:
                break # finished looping through sample size
        i+=1
        if i==3:
            break # finished looping through sampling methods

    fig.show()
    # # Saving the plot to file
    file_name = "Poly_lin_reg.png"
    file_path = os.path.join(output_dir, file_name) # creating full paths name
    fig.savefig(file_path, bbox_inches='tight', dpi=300)


def cheby_b():
    # Output directories for saving plots
    output_dir = "outputs/Plots_lin_reg/Cheby_basis/"

    # Create subplots
    fig, axs = plt.subplots(3,3, figsize=(18, 12), sharex=False, sharey=False)
    fig.suptitle('Linear Regression using Chebyshev Bases', fontsize=16) # common title of all the plots

    # obtain data for model development
    i = 0
    for sampler_case in [1,2,3]: # loop through the various sampling methods
        j = 0
        for ns in [5,10,15]: # loop through sample sizes
            print(f"Sampling Case: {sampler_case}, Sample Size: {ns}") # Display current combination of sampling size and method

            # obtain sample points for the given combination of sample size and method
            y_sample,x_sample = surr_model.sampler_func(ns,sampler_case)

            # fit  cheby bases using sampled points

            cheby2 = lin_reg_cheby(x_sample,y_sample,2) # Fit using 2 degree chebyyshev basis
            cheby4 = lin_reg_cheby(x_sample,y_sample,4) # Fit using 3 degree chebyshev basis
            cheby5 = lin_reg_cheby(x_sample,y_sample,5) # Fit using 4 degree chebyshev basis

            print(cheby2) # print weight vector for 2 degree chebyshev function
            print(cheby4) # "" 4
            print(cheby5) # "" 5

            x_dense = np.linspace(0, 1, 100) # create a dense vector of x-coordinates for plotting

            # true function
            y_dense_true = surr_model.true_func(x_dense) # plot the true function
            # Predictions using fit bases

            y_dense_cheby2 = np.array([np.sum(chebyshev_basis_recursive(xi,2)*cheby2) for xi in x_dense]) # size of 2 d matrix phi = (ns,degree)
            y_dense_cheby4 = np.array([np.sum(chebyshev_basis_recursive(xi,4)*cheby4) for xi in x_dense]) # size of 2 d matrix phi = (ns,degree)
            y_dense_cheby5 = np.array([np.sum(chebyshev_basis_recursive(xi,5)*cheby5) for xi in x_dense]) # size of 2 d matrix phi = (ns,degree)


            # compute mean sqaure error for each model
            mse2 = round(np.mean((y_dense_true - y_dense_cheby2)**2),4)
            mse4 = round(np.mean((y_dense_true - y_dense_cheby4)**2),4)
            mse5 = round(np.mean((y_dense_true - y_dense_cheby5)**2),4)

            # Plotting the results
            axs[i,j].plot(x_sample, y_sample, '*', markersize=4, label='sampled points') # scatter of samples
            axs[i,j].plot(x_dense, y_dense_true, label='True Function')
            axs[i,j].plot(x_dense, y_dense_cheby2, label=f'2 degree chebyshev (MSE = {mse2})')
            axs[i,j].plot(x_dense, y_dense_cheby4, label=f'4 degree chebyshev (MSE = {mse4})')
            axs[i,j].plot(x_dense, y_dense_cheby5, label=f'5 degree chebyshev (MSE = {mse5})')

            axs[i,j].legend()
            axs[i,j].set_xlabel('x')
            axs[i,j].set_ylabel('y')
            axs[i,j].set_title(f'Sample Size: {ns}, Sampling Method: {sampler_case}')
            axs[i,j].grid(True)

            j+=1
            if j==3:
                break # finished looping through sample size
        i+=1
        if i==3:
            break # finished looping through sampling methods

    fig.show()
    # # Saving the plot to file
    file_name = "Chebyshev_lin_reg.png"
    file_path = os.path.join(output_dir, file_name) # creating full paths name
    fig.savefig(file_path, bbox_inches='tight', dpi=300)


def sine_b():
    # Output directories for saving plots
    output_dir = "outputs/Plots_lin_reg/Sine_basis/"

    # Create subplots
    fig, axs = plt.subplots(3,3, figsize=(18, 12), sharex=False, sharey=False)
    fig.suptitle('Linear Regression using Sine Bases', fontsize=16) # common title of all the plots

    # obtain data for model development
    i = 0
    for sampler_case in [1,2,3]: # loop through the various sampling methods
        j = 0
        for ns in [5,10,15]: # loop through sample sizes
            print(f"Sampling Case: {sampler_case}, Sample Size: {ns}") # Display current combination of sampling size and method

            # obtain sample points for the given combination of sample size and method
            y_sample,x_sample = surr_model.sampler_func(ns,sampler_case)

            # fit  cheby bases using sampled points

            sine2 = lin_reg_sine(x_sample,y_sample,2) # Fit using 2 degree sine basis
            sine3 = lin_reg_sine(x_sample,y_sample,3) # Fit using 3 degree sine basis
            sine4 = lin_reg_sine(x_sample,y_sample,4) # Fit using 4 degree sine basis

            print(sine2) # print weight vector for 2 degree sine function
            print(sine3) # "" 3
            print(sine4) # "" 4

            x_dense = np.linspace(0, 1, 100) # create a dense vector of x-coordinates for plotting

            # true function
            y_dense_true = surr_model.true_func(x_dense) # plot the true function
            # Predictions using fit bases

            y_dense_sine2 = np.array([np.sum(sine_basis(xi,2)*sine2) for xi in x_dense]) # size of 2 d matrix phi = (ns,degree)
            y_dense_sine3 = np.array([np.sum(sine_basis(xi,3)*sine3) for xi in x_dense]) # size of 2 d matrix phi = (ns,degree)
            y_dense_sine4 = np.array([np.sum(sine_basis(xi,4)*sine4) for xi in x_dense]) # size of 2 d matrix phi = (ns,degree)


            # compute mean sqaure error for each model
            mse2 = round(np.mean((y_dense_true - y_dense_sine2)**2),4)
            mse3 = round(np.mean((y_dense_true - y_dense_sine3)**2),4)
            mse4 = round(np.mean((y_dense_true - y_dense_sine4)**2),4)

            # Plotting the results
            axs[i,j].plot(x_sample, y_sample, '*', markersize=4, label='sampled points') # scatter of samples
            axs[i,j].plot(x_dense, y_dense_true, label='True Function')
            axs[i,j].plot(x_dense, y_dense_sine2, label=f'2 degree sine (MSE = {mse2})')
            axs[i,j].plot(x_dense, y_dense_sine3, label=f'3 degree sine (MSE = {mse3})')
            axs[i,j].plot(x_dense, y_dense_sine4, label=f'4 degree sine (MSE = {mse4})')

            axs[i,j].legend()
            axs[i,j].set_xlabel('x')
            axs[i,j].set_ylabel('y')
            axs[i,j].set_title(f'Sample Size: {ns}, Sampling Method: {sampler_case}')
            axs[i,j].grid(True)

            j+=1
            if j==3:
                break # finished looping through sample size
        i+=1
        if i==3:
            break # finished looping through sampling methods

    fig.show()
    # # Saving the plot to file
    file_name = "Sine_lin_reg.png"
    file_path = os.path.join(output_dir, file_name) # creating full paths name
    fig.savefig(file_path, bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    poly_b()
    cheby_b()
    sine_b()
