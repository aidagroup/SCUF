from abc import ABC, abstractmethod

#abstract figure class
class Polynomials(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def build_model():
        pass
        
    @abstractmethod
    def equation(self):
        pass

    @abstractmethod
    def check_model(self):
        pass
    
#abstract ransac class
class AbstractRSFunctions(ABC):
    @abstractmethod
    def __init__(self):
        self.model_class = ABC
    
    @abstractmethod
    def check_inliers():
        pass
    
    @abstractmethod
    def get_samples():
        pass
    
    @abstractmethod
    def get_models():
        pass
    
    @abstractmethod
    def check_model():
        pass