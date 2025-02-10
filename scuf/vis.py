import numpy as np


def plot(ellipsoid, npoints = 20):
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
        "The 'open3d' package is required for this functionality. "
        "Install it with: pip install my_library[visualization]"
    )
    ell = ellipsoid[1]
    """Plot an ellipsoid"""
    u = np.linspace(0.0, 2.0 * np.pi, npoints)
    v = np.linspace(0.0, np.pi, npoints)
    a, b, c = ell[1]
    # cartesian coordinates that correspond to the spherical angles:
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones_like(u), np.cos(v))
    xyz = np.stack((x.flatten(), y.flatten(), z.flatten()))
    xyz = xyz.T
    for i in range(npoints*npoints):
        
        xyz[i] = np.dot(xyz[i], ell[2].T)
        xyz[i] = np.add(xyz[i], ell[0])

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(xyz)

    new_pcd.paint_uniform_color([1, 0, 0])
    return new_pcd