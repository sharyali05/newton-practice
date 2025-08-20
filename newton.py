def gradient(func, x, h=1e-5):
    #calculates gradient
    return (func(x + h) - func(x - h)) / (2 * h)


#newton method function
def newton_method(start, func, tol=1e-7, max_iter=1000, h=1e-5):
    x = start
    for i in range(max_iter):
        grad = gradient(func, x, h)
        
        x_new = x - func(x) / grad
        if abs(x_new - x) < tol:  #convergence check
            return x_new
        
        x = x_new