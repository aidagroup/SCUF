import logging
import numpy as np

from scuf.base import AbstractRSFunctions

from scuf.polynomial_figures import Ellipsoid, Ellipse, Line2d, Plane

FIGURES_DICT ={
    "ellipsoid": Ellipsoid,
    "ellipse": Ellipse,
    "line2d": Line2d,
    "plane": Plane
}

logger = logging.getLogger(__name__)

class PolynomialRS(AbstractRSFunctions):
    def __init__(self, figure="ellipsoid"):
        """
        Initialize the PolynomialRS class.

        Parameters:
        - figure (str): The type of figure to initialize. Default is "ellipsoid".
        """
        # Initialize the figure from the FIGURES_DICT
        self.figure = FIGURES_DICT[figure]()
        
        # Set the size of the figure
        self.size = self.figure.size
        
        # Set the dimension of the figure
        self.dimension = self.figure.dimension
        
        # Set the equation function of the figure
        self.function = self.figure.equation
        
        # Set the model building function of the figure
        self.model = self.figure.build_model
        
        # Set the model checking function of the figure
        self.check = self.figure.check_model
        
        # Set the parameters of the figure
        self.params = self.figure.params
        
        # Initialize the default vector with ones
        self.vector_default = np.ones(self.size)

    def set_params(self, params):
        """
        Set the parameters of the figure.

        Parameters:
        params (list): List of parameters to set.
        """
        if not params == None:
            # Set the parameters of the figure
            self.figure.params = params

    def check_inliers(self, models, points, threshold, method="median"):
        """
        Check which model has the best fit to the given points based on the specified method.

        Parameters:
        models (list): List of models to evaluate.
        points (ndarray): Array of points to be checked against the models.
        threshold (float): Threshold value for inlier determination.
        method (str): Method to evaluate the models ('median', 'count', 'average', 'sum').

        Returns:
        list: The model that best fits according to the chosen method.
        """
        
        # Calculate the polynomial representation of points
        pt = np.empty((0, self.size))
        pt = np.vstack([self.equation(d, self.vector_default) for d in points])
        
        # Extract model parameters
        px = np.vstack([d[0] for d in models])

        # Compute residuals
        res = np.dot(pt, px.T)

        # Determine the best model index based on the chosen method
        if method == "median":
            internal = np.absolute(np.median(res, axis=0))
            bset_index = np.argmin(internal)
        elif method == "count":
            internal = np.count_nonzero(np.absolute(res) < threshold, axis=0)
            bset_index = np.argmax(internal)
        elif method == "average":
            internal = np.absolute(np.average(res, axis=0))
            bset_index = np.argmin(internal)
        elif method == "sum":
            internal = np.absolute(np.sum(res, axis=0))
            bset_index = np.argmin(internal)
        else:
            logger.warning(f"There is no method {method}. Returning default first model")
            bset_index = 0
        
        # Log internal computation values for debugging
        logger.debug(internal)
        logger.debug(internal.max())
        
        return models[bset_index]

    def get_samples(self, points, nsamples=1000):
        """
        Generate random samples from given points.

        Parameters
        ----------
        points : ndarray
            Array of points to sample from.
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        ndarray
            Array of sampled points.

        Notes
        -----
        This function generates random samples from the given points by
        randomly selecting a subset of points equal to the size of the
        polynomial minus one.
        """
        if len(points) < self.size - 1:
            logger.warning(f"The points are less than the required minimum for compiling the system")
            raise ValueError
        samples_points = np.empty((nsamples, self.size - 1, self.dimension))
        for i in range(nsamples):
            # Randomly select a subset of points equal to the size of the polynomial minus one
            row_i = np.random.choice(points.shape[0], size=self.size - 1, replace=False)
            samples_points[i] = np.array(points[row_i, :])
        logger.info(f"There are {len(samples_points)} samples")
        return samples_points

    def get_models(self, samples_points):
        models = []
        b = np.full((1, self.size - 1), 1)

        for another_points in samples_points:
            a = np.empty((0, self.size))            
            for d in another_points:
                a = np.vstack([a, self.equation(d, self.vector_default)])
            try:
                a = np.delete(a, -1, axis = 1)
                mat = np.linalg.solve(a,b.T).flatten()
                #D = -np.sum(self.equation(d, np.concatenate((mat,[0.0]))))
                #models.append(np.concatenate((mat,[D])))
                models.append(np.concatenate((mat,[-1])))
            except Exception as inst:
                logger.info(f"a certain set of points is assembled into a singular matrix")
                #logger.info(inst)
        return models

    def check_model(self, model)->bool:
        return self.check(model)

    def equation(self, point, quadric) -> np.ndarray:
        return self.function(point, quadric)

    def build_model(self, quadric):
        return self.model(quadric)
