import argparse
import inspect
import json 

import numpy as np

from abc import ABC, abstractclassmethod
from typing import Any, Union, Callable

from ..metaclasses import AbstractHyperParser



def inspect_signature(func: Callable) -> dict:
    """
    Inspects the signature of a function and returns the 
    parameters of a function that have default values (aka kwargs)

    :param func: The function who's kwargs we want
    :type func: Callable
    :return: a dictionary with the default values of kwargs as values and their
             names as keys
    :rtype: dict
    """
    signature = inspect.signature(func)
    hp_dict = {
        k: v.default
        for k, v in signature.parameters.items()
        if not v.default is inspect.Parameter.empty
    }
    hp_dict.pop('args', None)
    hp_dict.pop('self', None)
    hp_dict.pop('kwargs', None)
    return hp_dict


class HyperParameter(object):
    """
    A default HyperParameter interface that allows bounded optimisation on a 
    classes input variables. The interface is designed specifically for optuna
    but should contain the information required for any generic bounded 
    optimisation implementation. 
    """
    __slots__ = ('default', 'bounds', 'type', 'length')

    def __init__(self,
                 default: Any,
                 bounds: tuple = None,
                 htype: type = None,
                 length: Union[list, int] = None):
        """
        Constructor for the HyperParameter object.

        :param default: The default value for the hyperparameter
        :type default: Any
        :param bounds: The bounds of the hyperparameter for optimisation will 
                       not be optimised if None, defaults to None
        :type bounds: tuple, optional
        :param htype: type of the hyperparameter, if None will be type(default)
                      , defaults to None
        :type htype: type, optional
        :param length: The length of the hyperparameter, defaults to None
        :type length: Union[list, int], optional
        """
        self.default = default
        self.bounds = bounds
        #Might need to consider if type(default) in list/tuple/np.ndarray
        self.type = htype if htype is not None else type(default)
        self._set_length(length)

    def _set_length(self, length: Union[list, int, None]) -> None:
        """
        Determines the length of the hyperparameter if it is not specified

        :param length: the specified length of the hyperparameter
        :type length: Union[list, int, None]
        """
        if length is not None:
            self.length = length
        elif self.type in [list, tuple, np.ndarray]:
            self.length = len(self.default)
        # elif self.type == np.ndarray:
        #     #I may need to make a special type of HP for numpy arrays
        #     self.length = self.default.shape
        else:
            assert length == None
            self.length = 1

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.default}, {self.bounds}, {self.type}, {self.length})'

    __repr__ = __str__


def dict2hpdict(hp_dict: dict) -> dict:
    """
    Converts all values in hp_dict to HyperParameter types.

    :param hp_dict: The dictionary who's values need to be transformed to 
                    HyperParameter.
    :type hp_dict: dict
    :return: A new dictionary where all values have been transformed to 
             HyperParameter
    :rtype: dict
    """
    return {k: HyperParameter(v) for k, v in hp_dict.items()}


class AbstractParametered(ABC, metaclass=AbstractHyperParser):
    """
    The interface required for any class that the user wishes to optimise. This
    is intended for use with the Component interface. As components within
    the RLOS framework are generally expected to require some level of tuning.

    Generates an argparse.ArgumentParser for any class by parsing the parameters
    of the class and any superclass it has. Doing this as part of the metaclass
    of this interface means that generating an Experiment's cli is automated. 
    """
    @classmethod
    def harvest_parameters(cls, hp_dict: dict = {}):
        """
        Generates the dictionary of hyperparameters for the class. 

        :param hp_dict: Any additional hyperparameters one wishes to add to the 
                        class has to be included in the class definition it is 
                        called by AbstractHyperParser at import, defaults to {}
        :type hp_dict: dict, optional
        """
        hp_dict = {**inspect_signature(cls.__init__), **hp_dict}
        cls.hyperparameters = dict2hpdict(hp_dict)
        cls.merge_hypers()
        cls.generate_argparser()

    @classmethod
    def generate_argparser(cls):
        """
        Creates the argparse.ArgumentParser for the class. Automates defining the
        cli input arguments for an Experiment. All cli arguments are in the format
        --{cls.__name__}-{parameter_name}.
        """
        #Will be a parent parser for the experiment so no help
        cls._parser = argparse.ArgumentParser(add_help=None)
        cls.set_up_hyperparameters()
        # unwanted = [None, [], {}, (), '']
        unwanted = [None]
        for k, v in cls.hyperparameters.items():
            #Handle multiple input arguments
            if isinstance(v.default, list) or isinstance(
                    v.default, tuple) or isinstance(v.default, np.ndarray):
                if isinstance(v.default, np.ndarray):
                    #numpy args will defualt to float32
                    typ = np.float32
                elif (v.default == []) or (v.default == ()) or (v.length == 0):
                    typ = None
                else:
                    typ = type(v.default[0])
                    
                cls._parser.add_argument(f'--{cls.__name__}-{k}'.replace(
                    '_', '-'),
                                         nargs='+',
                                         default=v.default,
                                         type=typ,
                                         help='A multi input argument')
            elif isinstance(v.default, dict):
                cls._parser.add_argument(f'--{cls.__name__}-{k}'.replace(
                    '_', '-'),
                                         default=str(v.default).replace("'",'"'),
                                         type=json.loads,
                                         help='Dictionary as string input')
            #Handle single input arguments
            #Gets rid of Nones <- not necessarily good behaviour
            elif v.default not in unwanted:
                typ = v.type
                if typ == bool:
                    typ = int
                cls._parser.add_argument(f'--{cls.__name__}-{k}'.replace(
                    '_', '-'),
                                         default=v.default,
                                         type=typ,
                                         help='A single input argument')

    @classmethod
    def merge_hypers(cls):
        """
        If the class inherits from a superclass that also has hyperparameters
        then those will be added to the hyperparameters of this class. 
        """
        if len(cls.mro()) > 1:
            super_class = cls.mro()[1]
            if hasattr(super_class, 'hyperparameters'):
                super_parameters = super_class.hyperparameters
                for key in super_parameters.keys():
                    if key not in cls.hyperparameters:
                        cls.hyperparameters[key] = super_parameters[key]

    @abstractclassmethod
    def set_up_hyperparameters(cls):
        """
        This function is used to add the bounds to any Component's hyperparameters.
        The optimisation library will parse a Component's hyperparameters and any 
        HyperParameter with a non NoneType bounds will be included in the optimisation
        process.
        """
        pass


def get_class_args(klass: AbstractParametered, args: dict) -> dict:
    """
    Helper function to generate the input arguments for an AbstractParametered
    klass given a dictionary args. args is generated by parsing the
    argparse.ArgumentParser given by an experiment after a program is invoked 
    that contains AbstractParametered components. 

    :param klass: The class who's arguments we require
    :type klass: AbstractParametered
    :param args: A dictionary that contains multiple component's input variables
    :type args: dict
    :return: A dictionary that contains the the input arguments required to create
             an instance of klass
    :rtype: dict
    """
    class_args = {}
    for k, v in args.items():
        if not k.startswith(f'{klass.__name__}_'):
            continue
        hp_k = k.replace(f'{klass.__name__}_', '')
        if hasattr(klass, 'hyperparameters') and hp_k in klass.hyperparameters:
            class_args[hp_k] = v
    return class_args