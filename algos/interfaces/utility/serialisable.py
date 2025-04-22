import json
from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
import pathlib
import inspect
import importlib

from ..metaclasses.register import RegisterMeta


class SerialRegister(RegisterMeta):
    """A metaclass for registering classes. Intended for use with serialisable.
    Automates decorating the init method with serialise_init
    """
    _register = {}
    serial_register = {'Serialisable'}
    experiment_register = {'AbstractExperiment'}

    def __new__(mcl, name, bases, class_dict):
        set_bases = {cls.__name__ for cls in bases}
        #intercept initialisers
        if (set_bases & SerialRegister.serial_register):
            SerialRegister.serial_register.add(name)
            if '__init__' in class_dict:
                class_dict['__init__'] = serialise_init(class_dict['__init__'])
        cls = super().__new__(mcl, name, bases, class_dict)
        #Make a set of experiments.
        if (set_bases & SerialRegister.experiment_register):
            cls._arg_parser()
            SerialRegister.experiment_register.add(name)
        return cls


class Serialisable(metaclass=SerialRegister):
    """A serialisable mixin that uses the SerialRegister metaclass
    
    Redefines str and repr and also provides a JSON serialise method.
    """
    def serialise(self, file_path: pathlib.Path = None) -> str:
        """Serialises self
        
        :return: A JSON string that contains all information necessary to reinitialise self
        :rtype: str
        """
        args, kwargs = self.args[:], self.kwargs.copy()
        for i, arg in enumerate(args):
            if hasattr(arg, 'serialise'):
                args[i] = arg.serialise()
        for key, value in kwargs.items():
            if hasattr(value, 'serialise'):
                kwargs[key] = value.serialise()
        j_dict = {
            'class': self.__class__.__name__,
            'args': args,
            'kwargs': kwargs
        }
        if file_path is not None:
            with open(file_path, 'w') as f:
                json.dump(j_dict, f, cls=NumpyEncoder, indent=4)
        return json.dumps(j_dict, cls=NumpyEncoder, indent=4)

    def __str__(self):
        items = [
            str(x) if not isinstance(x, str) else f"'{x}'" for x in self.args
        ] + [
            f'{k}={v}' if not isinstance(v, str) else f"{k}='{v}'"
            for k, v in self.kwargs.items()
        ]
        return f"{self.__class__.__name__}({','.join(items)})"

    __repr__ = __str__


class RegisteredAbstract(type(ABC), SerialRegister):
    """A metaclass for those classes that use both ABC and SerialRegister
    """
    pass


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int_, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return {'data': obj.tolist(), 'dtype': str(obj.dtype)}
        elif isinstance(obj, pathlib.Path):
            return {'path': str(obj)}
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif callable(obj): 
            return str(obj) 
        return json.JSONEncoder.default(self, obj)


def inspect_signature(func):
    signature = inspect.signature(func)
    kwonlys = {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    return kwonlys


def init_error_checker(args: tuple):
    if hasattr(args[0], 'args') != hasattr(args[0], 'kwargs'):
        raise ValueError(
            f'{args[0]} should either have args AND kwargs or neither')


def combine_kwargs_kwonlys(kwargs: dict, kwonlys: dict):
    if kwonlys:
        for key, val in kwonlys.items():
            if key not in kwargs:
                kwargs[key] = val


def create_args_kwargs_attr(args: tuple, kwargs: dict, kwonlys: dict):
    init_error_checker(args)
    combine_kwargs_kwonlys(kwargs, kwonlys)
    # object.__setattr__ is necessary for classes that have
    # __setattr__ overrides.
    object.__setattr__(args[0], 'args', list(args[1:]))
    object.__setattr__(args[0], 'kwargs', kwargs)
    return args, kwargs


def serialise_init(init):
    """The wrapper of the init function for any subclass of Serialisable

    :param init: The init function of a class that inherits from Serialisable
    :type init: Callable
    """
    def wrapper_serialise_init(*args, **kwargs):
        """Grabs the input arguments of the initialiser
        and makes them attributes of that instance"""
        kwonlys = inspect_signature(init)
        create_args_kwargs_attr(args, kwargs, kwonlys)
        init(*args, **kwargs)

    return wrapper_serialise_init


def check_if_serial(serialised):
    return 'class' in serialised and 'args' in serialised and 'kwargs' in serialised


def numpy_hook(dct):
    if 'dtype' in dct:
        return np.array(dct['data'], dtype=dct['dtype'])
    elif 'path' in dct:
        return pathlib.Path(dct['path'])
    return dct


def deserialise(serialised: str):
    """A function for deserialising Serialisable classes
    
    :param serialised: The output of Serialisable.serialise.
    :type serialised: str
    """
    if not check_if_serial(serialised):
        raise ValueError('Invalid input to deserialise')
    params = json.loads(serialised, object_hook=numpy_hook)
    cls = SerialRegister._register[params['class']]
    args, kwargs = params['args'], params['kwargs']
    for i, arg in enumerate(args):
        if type(arg) == str and check_if_serial(arg):
            args[i] = deserialise(arg)
    for key, value in kwargs.items():
        if type(value) == str and check_if_serial(value):
            kwargs[key] = deserialise(value)
    return cls(*args, **kwargs)