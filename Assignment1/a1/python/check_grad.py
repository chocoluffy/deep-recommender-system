import numpy as np
import numpy.linalg as LA

def check_grad(func, X, epsilon, *args):
    """
    checkgrad checks the derivatives in a function, by comparing them to finite
    differences approximations. The partial derivatives and the approximation
    are printed and the norm of the diffrence divided by the norm of the sum is
    returned as an indication of accuracy.

    usage: checkgrad(func, X, epsilon, P1, P2, ...)

    where X is the argument and epsilon is the small perturbation used for the finite
    differences. and the P1, P2, ... are optional additional parameters which
    get passed to f. The function f should be of the type 

    (fX, dfX) = func(X, P1, P2, ...)

    where fX is the function value and dfX is a vector of partial derivatives.

    Original Author: Carl Edward Rasmussen, 2001-08-01.

    Ported to Python 2.7 by JCS (9/21/2013).
    """

    if len(X.shape) != 2 or X.shape[1] != 1:
        raise ValueError("X must be a vector")

    y, dy, = func(X, *args)[:2]         # get the partial derivatives dy

    dh = np.zeros((len(X), 1))

    for j in xrange(len(X)):
        dx = np.zeros((len(X), 1))
        dx[j] += epsilon
        y2 = func(X+dx, *args)[0]
        dx = -dx
        y1 = func(X+dx, *args)[0]
        dh[j] = (y2 - y1)/(2*epsilon)

    print np.hstack((dy, dh))          # print the two vectors
    d = LA.norm(dh-dy)/LA.norm(dh+dy)  # return norm of diff divided by norm of sum

    return d
