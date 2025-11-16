Welcome!

This project aims to implement surrogate modelling using various basis
functions and modelling techniques as well as different number of samples to approximate
the given non-linear analytical function in 1 variable. The results obtained are compared
amongst the various approaches and the best performing approach is identified.

The non linear function selected is : y = (6x − 2)2 sin(12x − 4), xϵ[0, 1]

The following methods have been applied for surrogate modelling - 
  1. Linear regression - Implementation using the following basis functions -
  (a) polynomial basis functions
  (b) Chebyshev basis functions
  (c) Sine basis functions
2. Implementation of non-linear regression
3. Implementation of Piece-wise Cubic spline interpolation

   The following sampling approaches have been applied  -
(a) Random sampling (uniform) with 5, 10 and 15 samples
(b) Equi-spaced sampling with 5, 10 and 15 samples
(c) Latin hypercube sampling with 5, 10 and 15 samples

For all of the above, the performance of each technique in approximating the function is re-
ported via the mean squared error of the model with respect to the true function.
