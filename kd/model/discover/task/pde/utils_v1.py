"""Compatibility exports for the DISCOVER PINN finite-difference helpers.

Shared stencils live in utils_fd; the distinct 3D boundary stencil stays here.
"""

import numpy as np

from .utils_fd import (
    FiniteDiff,
    FiniteDiff2,
    Diff,
    Diff2,
    Diff3,
    Diff4,
    Diff_2,
    Diff2_2,
    Laplace,
    Diff_3,
)


def Diff2_3(u, dxt, name=1):
    """
    Here dx is a scalar, name is a str indicating what it is
    """


    if u.shape == dxt.shape:
        return u/dxt
    t,n,m,p = u.shape
    uxt = np.zeros((t, n, m,p))
    dxt = dxt.ravel()
    dxt = dxt[2]-dxt[1]
    if name == 1:

        uxt[:,1:n-1,:,:]= (u[:,2:n,:,:] - 2 * u[:,1:n-1,:,:] + u[:,0:n-2,:,:]) / dxt ** 2
        uxt[:,0,:,:] = (2 * u[:,0,:,:] - 5 * u[:,1,:,:] + 4 * u[:,2,:,:] - u[:,3,:,:]) / dxt ** 2
        uxt[:,n - 1,:,:] = (2 * u[:,n - 1,:,:] - 5 * u[:,n - 2,:,:] + 4 * u[:,n - 3,:,:] - u[:,n - 4,:,:]) / dxt ** 2
    elif name == 2:

        uxt[:,:,1:m-1,:]= (u[:,:,2:m,:] - 2 * u[:,:,1:m-1,:] + u[:,:,0:m-2,:]) / dxt ** 2
        uxt[:,:,0,:] = (2 * u[:,:,0,:] - 5 * u[:,:,1,:] + 4 * u[:,:,2,:] - u[:,:,3,:]) / dxt ** 2
        uxt[:,:,m - 1,:] = (2 * u[:,:,m - 1,:] - 5 * u[:,:,m - 2,:] + 4 * u[:,:,m - 3,:] - u[:,:,m - 4,:]) / dxt ** 2
    elif name == 3:
        uxt[:,:,:,1:m-1]= (u[:,:,:,2:m] - 2 * u[:,:,:,1:m-1] + u[:,:,:,0:m-2]) / dxt ** 2
        uxt[:,:,0] = (2 * u[:,:,0] - 5 * u[:,:,1] + 4 * u[:,:,2] - u[:,:,3]) / dxt ** 2
        uxt[:,:,:,p - 1] = (2 * u[:,:,:,p - 1] - 5 * u[:,:,:,p - 2] + 4 * u[:,:,:,p - 3] - u[:,:,:,p- 4]) / dxt ** 2
    else:
        NotImplementedError()

    return uxt
