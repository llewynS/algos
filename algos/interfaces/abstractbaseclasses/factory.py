from abc import ABC, abstractmethod

from .experiment import AbstractExperiment

class AbstractComponentFactory(ABC):
    """Constructs components
    
    Register for each factory records which classes are constructed by
    the factory interface. The 
    """
    _register = []

    def __init__(self, experiment: AbstractExperiment):
        self._components = experiment.components
        self._exp_args = experiment._exp_args
        self._experiment = experiment
        self.build()

    @abstractmethod
    def build(self):
        pass
