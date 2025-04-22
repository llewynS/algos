import time
import inspect
from functools import wraps
from typing import List, Union, Dict, Callable, Tuple

def find_para_pos_in_args(func: Callable, parameter: str) -> Union[Tuple[int, dict], None]:
    """Find the position of parameter in func's args

    :param func: The function who's parameter position we are finding
    :type func: Callable
    :param parameter: The name of the parameter of interest
    :type parameter: str
    :return: The position of said parameter
    :rtype: int
    """
    signature = inspect.signature(func)
    for i, para in enumerate(signature.parameters.values()):
        if para.name == parameter:
            return i, para.default if para.default != inspect._empty else None

def listify_inputs(check_param: Union[List[str], Dict[str,str]]) -> List[str]:
    """Returns the list of input names 

    :param check_param: Dict or list of inputs 
    :type check_param: Union[List[str], Dict[str,str]]
    :return: input names
    :rtype: List[str]
    """
    if type(check_param) == list:
        return check_param
    if type(check_param) == dict:
        if type(check_param[list(check_param.keys())[0]]) == str:
            return list(check_param.keys())
    out = []
    for key, val in check_param.items():
        if type(val) == dict:
            out = [*out, *list(val.keys())]
        elif type(val) == list:
            out = [*out, *val]
    return out

def check_inputs(check_param: Union[List[str], Dict[str,str]], inputs:List[str], parameter:str):
    """Helper function to verify the check_param contains all of inputs

    :param check_param: The list of values to check against inputs
    :type check_param: List[str]
    :param inputs: inputs check_param must contain
    :type inputs: List[str]
    :param parameter: the parameter of interest
    :type parameter: str
    :raises KeyError: all of inputs must be in check_param
    """
    check_param = listify_inputs(check_param)
    for input in inputs:
        if input not in check_param:
            raise KeyError(f"{parameter} must contain the string: {input}")

def check_parameter_contains(*inputs: List[str], parameter:str="inputs"):
    """Check that the function's parameter contains all of the strings in inputs

    This function is agnostic to whether the parameter is an arg or kwarg.

    :param parameter: expected to be inputs or outputs for components, defaults to "inputs"
    :type parameter: str, optional
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if parameter in kwargs:
                check_inputs(kwargs[parameter], inputs, parameter)
            else:
                check = find_para_pos_in_args(func, parameter)
                if check is None:
                    raise AttributeError(f"{parameter} is not in {func}")
                elif check[1] is None:
                    check_inputs(args[check[0]-1],inputs,parameter)
                else:
                    check_inputs(check[1], inputs, parameter)                    
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

def check_inputs_contains(*inputs):
    """Specifies the parameter of check_parameter_contains to be inputs
    """
    return check_parameter_contains(*inputs, parameter="inputs")

def check_outputs_contains(*inputs):
    """Specifies the parameter of check_parameter_contains to be outputs
    """
    return check_parameter_contains(*inputs, parameter="outputs")

def check_io_map_contains(*inputs):
    """Specifies the parameter of check_parameter_contains to be outputs
    """
    return check_parameter_contains(*inputs, parameter="io_map")