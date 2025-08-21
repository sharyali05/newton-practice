def derivative(func, x, h=1e-5):
    #First derivative
    return (func(x + h) - func(x - h)) / (2 * h)

def second_derivative(func, x, h=1e-5):
    #second derivative
    return (func(x + h) - 2*func(x) + func(x - h)) / (h**2)

def newton_method(func, x0, tol=1e-6, max_iter=100):
    #Newton's method for univariate optimization.
    x = x0
    iteration = 0
    
    while abs(derivative(func, x)) > tol and iteration < max_iter:
        g = derivative(func, x)
        H = second_derivative(func, x)
        if H == 0:   #no division by zero
            break
        x = x - g / H
        iteration += 1
    
    return x

import numpy as np
def newton_multi_var(grad_F, H, x0, tol=1e-8, max_iter=100):
    x = np.array(x0, dtype=float)
    for i in range(max_iter):
        g = grad_F(x)
        Hx = H(x)
        
        delta = np.linalg.solve(Hx, g)

        x_new = x - delta
        if np.linalg.norm(delta) < tol:
            return x_new
        x = x_new
    return x
 
