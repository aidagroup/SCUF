import logging

import numpy as np
import open3d as o3d
from scuf.polynomial_rs import PolynomialRS


RANSAC_CLASSES = {
    "ellipsoid": PolynomialRS,
    "ellipse": PolynomialRS,
    "line2d": PolynomialRS,
    "plane": PolynomialRS
}
logger = logging.getLogger(__name__)


class RANSAC:
    def __init__(self, figure="ellipsoid"):
        self.__figure = RANSAC_CLASSES[figure](figure)
        self.best_model = None

    def fit(self, points, iterations=1000, threshold=0.01, params=[10, 2.0], method="count"):
        self.__figure.set_params(params)
        points_samples = self.__figure.get_samples(points, nsamples=iterations)

        hypotheses = self.__figure.get_models(points_samples)
        logger.info(f"Number of hypotheses {len(hypotheses)} ")
        models = []
        for __model in hypotheses:
            try:
                model = self.__figure.build_model(__model)
            except Exception as inst:
                logger.info(f"a certain set of points is assembled into a singular matrix")
            if not self.__figure.check_model(model):
                continue
            else:
                models.append(model)
        logger.info(f"Number of models {len(models)} ")
        best_model = self.__figure.check_inliers(
            models, points, threshold, method) if models else None
        self.best_model = best_model
        if best_model != None:
            if best_model[0][0]>0:
                best_model[0] = best_model[0] * (-1)
            self.best_model = best_model    
        return self.best_model

def generate_sphere_points2():
    phi = np.random.uniform(0, 2*np.pi)
    costheta = np.random.uniform(-1, 1)
    u = np.random.uniform(0, 1)

    theta = np.arccos( costheta )
    r = ( u )**(1./3.)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return [x, y, z]

def generate_sphere(npoints=1500):
    """
    Generate a set of random points uniformly distributed on the surface of a sphere.

    Parameters:
    npoints (int): The number of points to generate. Default is 1500.

    Returns:
    np.ndarray: An array of shape (npoints, 3) containing the generated points.
    """
    empty_array = np.empty((npoints, 3))
    for i in range(npoints):
        # Generate a single point on the surface of the sphere
        empty_array[i] = np.array(generate_sphere_points2())
    return empty_array

def volume(elle):
    a, b, c = elle
    return 4.0 / 3 * np.pi*a*b*c

def plot(ellipsoid, npoints = 20):
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