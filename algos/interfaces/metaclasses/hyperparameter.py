from .register import RegisterMeta 
from .interfacegenerator import InterfaceGeneratorMeta


class HyperParser(RegisterMeta):
    """
    Creates a register for hyper-parametrised classes that could be 
    optimised. This is the metaclass for the AbstractParametered abstract
    base class. 
    Additionally to the register it attaches the hyper-parameters to cls
    """
    # _register = {}

    def __new__(mcl, name, bases, class_dict):
        cls = super().__new__(mcl, name, bases, class_dict)
        cls.harvest_parameters()
        return cls


class AbstractHyperParser(InterfaceGeneratorMeta, HyperParser):
    """
    The AbstractHyperParser allows the class using this metaclass to have
    both HyperParser and ABC functionality.
    This is so abstractclassmethod and abstractmethod decorators can be used
    on the classes.
    """
    pass