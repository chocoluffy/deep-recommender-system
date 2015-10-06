import numpy as np

def l2_distance(a, b):
    """Computes the Euclidean distance matrix between a and b.

    Inputs:
        A: D x M array.
        B: D x N array.

    Returns:
        E: M x N Euclidean distances between vectors in A and B.


    Author   : Roland Bunschoten
               University of Amsterdam
               Intelligent Autonomous Systems (IAS) group
               Kruislaan 403  1098 SJ Amsterdam
               tel.(+31)20-5257524
               bunschot@wins.uva.nl
    Last Rev : Wed Oct 20 08:58:08 MET DST 1999
    Tested   : PC Matlab v5.2 and Solaris Matlab v5.3

    Copyright notice: You are free to modify, extend and distribute 
       this code granted that the author of the original code is 
       mentioned as the original author of the code.

    Fixed by JBT (3/18/00) to work for 1-dimensional vectors
    and to warn for imaginary numbers.  Also ensures that 
    output is all real, and allows the option of forcing diagonals to
    be zero.  

    Basic functionality ported to Python 2.7 by JCS (9/21/2013).
    """

    if a.shape[0] != b.shape[0]:
        raise ValueError("A and B should be of same dimensionality")

    aa = np.sum(a**2, axis=0)
    bb = np.sum(b**2, axis=0)
    ab = np.dot(a.T, b)

    return np.sqrt(aa[:, np.newaxis] + bb[np.newaxis, :] - 2*ab)
