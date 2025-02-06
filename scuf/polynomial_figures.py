import numpy as np

from scuf.base import Polynomials

'''
3d figures, abstract class

'''

class Quadric3d(Polynomials):
    def __init__(self):
        super().__init__()
        self.figure = "Quadric3d"
        self.dimension = 3 
        self.size = 10
        
    def equation(self, point=np.ones(3), coefficients=np.ones(10)) -> np.ndarray:
        """
        Compute the equation of a 3D quadric given by the coefficients.

        Parameters:
        - point (np.ndarray): The point to evaluate the equation at. Default is [1,1,1].
        - coefficients (np.ndarray): The coefficients of the equation. Default is [1,1,1,0,0,0,0,0,0,1].

        Returns:
        - np.ndarray: An array representing the evaluated equation components.
        """
        x, y, z = point
        a, b, c, f, g, h, p, q, r, d = coefficients
        return np.array([
            a * x**2, b * y**2, c * z**2,
            2 * f * x * y, 2 * g * x * z, 2 * h * y * z,
            2 * p * x, 2 * q * y, 2 * r * z, d
        ])
    
    def build_matrices(self, quadric=np.zeros(10)):
        """Build matrices for the quadric equation of an ellipsoid.

        Parameters:
        quadric (ndarray): The coefficients of the equation. Default is [0,0,0,0,0,0,0,0,0,0].

        Returns:
        dict: A dictionary containing the following matrices: Am, m1, m2, m3, mk, k1, k2, k3, d1, d2, d3, Fa, Fb.
        """
        # The matrix of the quadratic part of the equation
        Am = np.array([[quadric[0], quadric[3], quadric[5]],
                       [quadric[3], quadric[1], quadric[4]],
                       [quadric[5], quadric[4], quadric[2]]])

        # The matrices of the linear parts of the equation
        m1 = np.array([[Am[0, 0], Am[0, 1]], [Am[1, 0], Am[1, 1]]])
        m2 = np.array([[Am[0, 0], Am[0, 2]], [Am[2, 0], Am[2, 2]]])
        m3 = np.array([[Am[1, 1], Am[1, 2]], [Am[2, 1], Am[2, 2]]])

        # The matrix of the equation
        mk = np.array([[Am[0, 0], Am[0, 1], Am[0, 2], quadric[6]],
                       [Am[1, 0], Am[1, 1], Am[1, 2], quadric[7]],
                       [Am[2, 0], Am[2, 1], Am[2, 2], quadric[8]],
                       [quadric[6], quadric[7], quadric[8], quadric[9]]])

        # The matrices of the inhomogeneous part of the equation
        k1 = np.array([[Am[0, 0], Am[0, 1], quadric[6]],
                       [Am[1, 0], Am[1, 1], quadric[7]],
                       [quadric[6], quadric[7], quadric[9]]])
        k2 = np.array([[Am[0, 0], Am[0, 2], quadric[6]],
                       [Am[2, 0], Am[2, 2], quadric[8]],
                       [quadric[6], quadric[8], quadric[9]]])
        k3 = np.array([[Am[1, 1], Am[1, 2], quadric[7]],
                       [Am[2, 1], Am[2, 2], quadric[8]],
                       [quadric[7], quadric[8], quadric[9]]])

        # The matrices of the constant part of the equation
        d1 = np.array([[Am[0, 0], quadric[6]], [quadric[6], quadric[9]]])
        d2 = np.array([[Am[1, 1], quadric[7]], [quadric[7], quadric[9]]])
        d3 = np.array([[Am[2, 2], quadric[8]], [quadric[8], quadric[9]]])

        # The matrices of the equation in the Fa*x + Fb = 0 form
        Fa = np.array([[Am[0, 0], Am[0, 1], Am[0, 2]],
                       [Am[1, 0], Am[1, 1], Am[1, 2]],
                       [Am[2, 0], Am[2, 1], Am[2, 2]]])
        Fb = np.array([-quadric[6], -quadric[7], -quadric[8]])

        return {'Am': Am, 'm1': m1, 'm2': m2, 'm3': m3, 'mk': mk, 'k1': k1, 'k2': k2,
                'k3': k3, 'd1': d1, 'd2': d2, 'd3': d3, 'Fa': Fa, 'Fb': Fb}

""" 
figures 
"""
class Ellipsoid(Quadric3d):
    def __init__(self, params = []):
        super().__init__()
        self.figure = "Ellipsoid"
        self.params = params

    def build_model(self, quadric):
        quadric = np.asarray(quadric)

            # Check if quadric has the correct shape
        if quadric.shape != (10,):
            raise ValueError("Input quadric must be a 1D array of shape (10,)")

        # Step 1: Modify the last element based on the condition
        if quadric[0] * quadric[9] > 0:
            quadric[9] *= -1
        
        # Step 2: Build the Q and L matrices (3x3 and 3x1)
        Q = np.array([[quadric[0], quadric[3], quadric[4]],
                    [quadric[3], quadric[1], quadric[5]],
                    [quadric[4], quadric[5], quadric[2]]])

        L = 2 * np.array([[quadric[6]],
                        [quadric[7]],
                        [quadric[8]]])


        # Step 3: Calculate the center
        try:
            Q_inv = np.linalg.inv(Q)
            center = - (L.T @ Q_inv) / 2
        except np.linalg.LinAlgError:
            print("Warning: Matrix Q is singular or close to singular. Using pseudoinverse.")
            Q_inv = np.linalg.pinv(Q)
            center = - (L.T @ Q_inv) / 2


        # Step 4: Calculate TP
        TP = 0.25 * (L.T @ Q_inv @ L) + 1


        # Step 5: Calculate eigenvalues and eigenvectors
        try:
            vals, evecs = np.linalg.eigh(Q)
        except np.linalg.LinAlgError:
            print("Warning: Eigenvalue computation failed. Using SVD as fallback.")
            U, S, Vt = np.linalg.svd(Q)
            vals = S
            evecs = U
        # Step 6: Calculate radii
        radii = np.sqrt(TP / vals[:, np.newaxis])

        # Step 7: Combine all results into a single array
        return [quadric, [center.flatten(), radii.flatten(), evecs]]

    def check_model(self, model)->bool:
    #model looks like [quadric, [center, radii, evecs, quadric]]
    #semiaxis
        
        a,b,c = model[1][1]
        if np.isnan(a) or np.isnan(b) or np.isnan(c):
            return False
        
        if a<0 or b<0 or c<0:
            return False
        if self.params!=None:
            k, h = self.params
            if a>k or b>k or c>k:
                return False
            if max((a, b))/min((a,b)) > h  or max((b,c))/min((b,c)) > h or max((c,a))/min((c,a)) > h:
                return False
        return True

class Sphere(Quadric3d):
    def __init__(self, params = None):
        super().__init__()
        self.figure = "Sphere"
        self.params = params

    def build_model(self, quadric = np.zeros(10)):
        des = True
        matrix = self.build_matrices(quadric)  
        center = np.linalg.solve(matrix["Fa"], matrix["Fb"].T)
        translation_matrix = np.eye(4)
        translation_matrix[3, :3] = center.T
        R1 = translation_matrix.dot(matrix["mk"]).dot(translation_matrix.T)
        evals, evecs = np.linalg.eig(R1[:3, :3] / -R1[3, 3])
        evecs = evecs.T
        radii = np.sqrt(1. / np.abs(evals))
        radii *= np.sign(evals)
        r = radii
        return [quadric, [center, r], des]
    
class Spheroid(Quadric3d):
    def __init__(self, params = None):
        super().__init__()
        self.figure = "Sphere"
        self.params = params

    def build_model(self, quadric = np.zeros(10)):
        des = True
        matrix = self.build_matrices(quadric)  
        center = np.linalg.solve(matrix["Fa"], matrix["Fb"].T)
        translation_matrix = np.eye(4)
        translation_matrix[3, :3] = center.T
        R1 = translation_matrix.dot(matrix["mk"]).dot(translation_matrix.T)
        evals, evecs = np.linalg.eig(R1[:3, :3] / -R1[3, 3])
        evecs = evecs.T
        radii = np.sqrt(1. / np.abs(evals))
        radii *= np.sign(evals)
        r = radii
        return [quadric, [center, r], des]

'''
2d figures, abstract class

'''

class Quadric2d(Polynomials):
    def __init__(self):
        super().__init__()
        self.figure = "Quadric2d"
        self.dimension = 2 
        self.size = 6

    def equation(self, point = np.ones(2), coefficients = np.ones(6)):
        x, y = point
        A, B, C, D, E, F = coefficients
        return np.array([A*x**2, 2*B*y*x, C*y**2, 2*D*x, 2*E*y, F])
    
    def build_matrix(self, quadric = np.zeros(10)):
        return quadric


class Ellipse(Quadric2d):
    def __init__(self, params = []):
        super().__init__()
        self.figure = "Ellipse"
        self.params = params

    def build_model(self, quadric = np.zeros(6)):
        A, B, C, D, E, F = quadric
        Q = np.array([[  A, B/2, D/2],
                    [B/2,   C, E/2],
                    [D/2, E/2,   F]])
        Q = Q / Q[2,2]
        B *=2
        D *=2
        E *=2
        
        if np.linalg.det(Q) == 0:
            des = False
            return [quadric, [], des]
            raise ValueError("Degenerate conic found!")

        if np.linalg.det(Q[:2,:2]) <= 0: # According to Wikipedia
            des = False
            return [quadric, [], des]
            raise ValueError("These parameters do not define an ellipse!")

        # Get centre
        denominator = B**2 - 4*A*C
        centre_x = (2*C*D - B*E) / denominator
        centre_y = (2*A*E - B*D) / denominator
        center = np.array([centre_x, centre_y])
        #print("Centre x:{} y:{}".format(centre_x, centre_y))

        # Get major and minor axes
        K = - np.linalg.det(Q[:3,:3]) / np.linalg.det(Q[:2,:2])
        root = np.sqrt(((A - C)**2 + B**2))
        a = np.sqrt(2*K / (A + C - root))
        b = np.sqrt(2*K / (A + C + root))

        vsp = (B ** 2 - 4 * A * C)
        ab1 = (A * E ** 2 + C * D ** 2 - B * D * E + vsp * F)
        ab2 = np.sqrt((A - C) ** 2 + B ** 2)
        a = - np.sqrt(2 * ab1 * ((A + C) + ab2)) / vsp
        b = - np.sqrt(2 * ab1 * ((A + C) - ab2)) / vsp
        radii = np.array([a, b])
        #print("Major:{} minor:{}".format(a, b))

        # Get angle

        angle = np.arctan2(C - A + root, B)

        angle = - np.arctan2((C - A - ab2), B)
        #angle *= 180.0/np.pi # Convert angle to degrees
        #print("Angle in degrees: {}".format(angle))
        return [quadric, [center, radii, angle]]
    
    def check_model(self, model)->bool:
    #model looks like [quadric, [center, radii, evecs, quadric]]
    #semiaxis
        des = model[2]
        if not des:
            return False
        k, h = self.params
        a,b = model[1][1]
        if a<0 or b<0:
            return False
        
        if a>k or b>k:
            return False
        #if max((a, b))/min((a,b)) > h  or max((b,c))/min((b,c)) > h or max((c,a))/min((c,a)) > h:
        #    return False
        return True

class Plane(Polynomials):
    def __init__(self, params):
        super().__init__()
        self.figure = "Plane"
        self.dimension = 3 
        self.size = 4
        self.params = params
        
    def equation(self, point = np.ones(3), coefficients = np.ones(4)):
        x, y, z = point
        A, B, C, D = coefficients
        return np.array([A*x, B*y, C*z, D])
    
    def build_model(self, quadric = np.zeros(10)):
        return [quadric]

class Line2d(Polynomials):
    def __init__(self):
        super().__init__()
        self.figure = "Line2d"
        self.dimension = 2 
        self.size = 4
        
    def equation(self, point = np.ones(2), coefficients = np.ones(3)):
        x, y = point
        A, B, C = coefficients
        return np.array([A*x, B*y, C])
    
    def build_matrix(self, quadric = np.zeros(4)):
        return [quadric]