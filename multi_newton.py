def derivative(func, x, h=1e-5):
    """Numerical first derivative."""
    return (func(x + h) - func(x - h)) / (2 * h)

def second_derivative(func, x, h=1e-5):
    """Numerical second derivative."""
    return (func(x + h) - 2*func(x) + func(x - h)) / (h**2)

def newton_method(func, x0, tol=1e-6, max_iter=100):
    """Newton's method for univariate optimization."""
    x = x0
    iteration = 0
    
    while abs(derivative(func, x)) > tol and iteration < max_iter:
        g = derivative(func, x)
        H = second_derivative(func, x)
        if H == 0:   #no division by zero
            print("Hessian is zero, stopping.")
            break
        x = x - g / H
        iteration += 1
    
    return x
