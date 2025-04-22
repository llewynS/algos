#################################################################################################################
# This is a fresh class to formalise the IO a little better so that it is easy to keep track of which           #
# subfunction each IO belongs to as the entire point of this OS is to control the flow of messages so that      #
# we can interrupt implemented algorithms and extract the necessary information to have our own segment work    #
# and alter the original algorithm i.e a truly modular solution.                                                #
# Important features:                                                                                           #
# 1) Automatic graph generation to ensure that we have completed loops                                          #
# 2) extensible to any number of IO segmentation                                                                #
# This class should implement the gating                                                                        #
#################################################################################################################
import inspect
import ast
import re
import random
import string

from typing import Dict, List, Tuple, Union, Set, Callable, Any
from ..patterns import Subject, Observer
from ..utility import listify_inputs

def get_operation(op: ast.operator)->str:
    """Get the operation of an ast.operator

    :param op: the operator
    :type op: ast.operator
    :return: the string representaion of the operator
    :rtype: str
    """
    if isinstance(op, ast.Add):
        return "+"
    elif isinstance(op, ast.Sub):
        return "-"
    elif isinstance(op, ast.Mult):
        return "*"
    elif isinstance(op, ast.Div):
        return "/"
    elif isinstance(op, ast.Pow):
        return "**"
    elif isinstance(op, ast.Mod):
        return "%"
    elif isinstance(op, ast.FloorDiv):
        return "//"
    elif isinstance(op, ast.BitAnd):
        return "&"
    elif isinstance(op, ast.BitOr):
        return "|"
    elif isinstance(op, ast.BitXor):
        return "^"
    elif isinstance(op, ast.LShift):
        return "<<"
    elif isinstance(op, ast.RShift):
        return ">>"
    elif isinstance(op, ast.MatMult):
        return "@"
    elif isinstance(op, ast.Lt):
        return "<"
    elif isinstance(op, ast.Gt):
        return ">"
    elif isinstance(op, ast.LtE):
        return "<="
    elif isinstance(op, ast.GtE):
        return ">="
    elif isinstance(op, ast.Not):
        return "not"

def extract_attribute(value: ast.Attribute)->str:
    """Extract the attribute from an ast.Attribute

    :param value: the value
    :type value: ast.Attribute
    :return: the string representation of the attribute
    :rtype: str
    """    
    values = []
    while isinstance(value, ast.Attribute):
        values.append(value.attr)
        value = value.value
    return ".".join(reversed(values))

def get_innermost_arg(call, args=None):
    if args is None:
        args = []
    if isinstance(call, ast.Call):
        # If the function being called is itself a call, recurse on that
        # Also, add the arguments of the call to the args list
        if isinstance(call.func, ast.Attribute):
            if isinstance(call.func.value, ast.Attribute):
                #print(f'appending {call.func.value.attr} in from call.func.value.attr')
                args.append(call.func.value.attr)
        for arg in call.args:
            if isinstance(arg, ast.Attribute):
                #print(f'appending {arg.attr} in args atteibute')
                args.append(arg.attr)
            elif isinstance(arg, ast.Call):
                get_innermost_arg(arg, args)
            elif isinstance(arg, ast.Starred):
                if isinstance(arg.value, ast.Attribute):
                    #print(f'appending {arg.value.attr} in args starred')
                    args.append(arg.value.attr)
        for keyword in call.keywords:
            if isinstance(keyword.value, ast.Attribute):
                #print(f'appending {keyword.value.attr} in keyword attr')
                args.append(keyword.value.attr)
            elif isinstance(keyword.value, ast.Call):
                get_innermost_arg(keyword.value, args)
        return get_innermost_arg(call.func, args)
    elif isinstance(call, ast.Attribute):
        # If the function being called is an attribute access, recurse on the value the attribute is being accessed on
        return get_innermost_arg(call.value, args)
    else:        
        # Otherwise, return the argument to the innermost call
        return [f'self.{arg}' for arg in args]
    
def get_value(value: ast.AST, cls:object) -> str:
    """Get the value of an ast.AST

    :param value: the value
    :type value: ast.AST
    :return: the string representation of the value
    :rtype: str
    """
    if isinstance(value, ast.Constant):
        return value.value
    elif isinstance(value, ast.Attribute):
        #print(value.attr)
        base = get_value(value.value, cls)
        return f"{base}.{value.attr}"
        
    elif isinstance(value, ast.Name):
        return value.id
    elif isinstance(value, ast.Call):
        # Assume that the function is not a method/function 
        # that has additional IO..., for now
        ret_value = ", ".join(get_innermost_arg(value))
        return ret_value
    elif isinstance(value, ast.Tuple):
        ret_value = ", ".join([get_value(arg,cls) for arg in value.elts])
        return ret_value
    elif isinstance(value, ast.DictComp):
        """The function needs to take in the instance. We then need to get all the values being pulled by the generator.
        We then get all of the values in the generator and generate a string that is all of those values separated by commas or something...
        """
        print("Warning this is parsing a DictComp and the implementation is not robust...")
        # key = value.key
        generators = value.generators
        # value = value.value
        nesting = get_value(generators[0],cls).split('.')[1:]
        for i,nest in enumerate(nesting):
            if i == 0:
                values = getattr(cls, nest)
            else:
                values = getattr(values, nest)
        return ", ".join([f"self.{value}" for value in values])
    
    elif isinstance(value, ast.comprehension):
        """This is a bandaid and needs to be streamlined way more..."""
        print("Warning this is parsing a comprehension and the implementation is not robust...")

        target = value.target
        iter = value.iter
        ifs = value.ifs
        # temp = f"{get_value(target,cls)} in {get_value(iter,cls)}"
        # if ifs:
        #     temp += f" if {get_value(ifs[0],cls)}"
        return f"{get_value(iter,cls)}"
    
    elif isinstance(value, ast.BinOp):
        left = value.left
        right = value.right
        return f"{get_value(left,cls)} {get_operation(value.op)} {get_value(right,cls)}"
    elif isinstance(value, ast.UnaryOp):
        operand = value.operand
        return f"{get_operation(value.op)} {get_value(operand,cls)}"
    elif isinstance(value, ast.Compare):
        left = value.left
        right = value.comparators[0]
        return f"{get_value(left,cls)} {get_operation(value.ops[0])} {get_value(right,cls)}"
    elif isinstance(value, ast.BoolOp):
        left= value.values[0]
        right = value.values[1]
        return f"{get_value(left,cls)} {get_operation(value.op)} {get_value(right,cls)}"
    elif isinstance(value, ast.Dict):
        keys = value.keys
        values = value.values
        return ', '.join([f'{get_value(key,cls)}' for key in keys] + [f'{get_value(value,cls)}' for value in values])
    elif isinstance(value, ast.Subscript):
        container = get_value(value.value, cls)
        subs = value.slice
        return f"{get_value(container,cls)}[{get_value(subs,cls)}]"
    elif not isinstance(value, ast.AST):
        return str(value)
    elif isinstance(value, ast.Slice):
        lower = value.lower
        upper = value.upper
        step = value.step
        if lower:
            lower = get_value(lower,cls)
        if upper:
            upper = get_value(upper,cls)
        if step:
            step = get_value(step,cls)
        return f"{lower}:{upper}:{step}"
    elif isinstance(value, ast.List):
        elts = value.elts
        return f"[{', '.join([str(get_value(elt,cls)) for elt in elts])}]"
    else:
        raise NotImplementedError(f"Value {value} not implemented")
    
# Instead of converting to a string we return a list of ast.AST
def get_value_v2(value: ast.AST)->List[ast.AST]:
    """Get the value of an ast.AST

    :param value: the value
    :type value: ast.AST
    :return: List of ASTs
    :rtype: List[ast.AST]
    """
    if isinstance(value, ast.Constant):
        return [value]
    elif isinstance(value, ast.Attribute):
        return [value]
    elif isinstance(value, ast.Name):
        return [value]
    elif isinstance(value, ast.Call):
        return [value]
    elif isinstance(value, ast.BinOp):
        left = value.left
        right = value.right
        return get_value_v2(left) + get_value_v2(right)
    elif isinstance(value, ast.UnaryOp):
        operand = value.operand
        return get_value_v2(operand)
    elif isinstance(value, ast.Compare):
        left = value.left
        right = value.comparators[0]
        return get_value_v2(left) + get_value_v2(right)
    elif isinstance(value, ast.BoolOp):
        left= value.values[0]
        right = value.values[1]
        return get_value_v2(left) + get_value_v2(right)

# Parses the list output by get_value_v2 and returns a list of strings if the value is a Name or Attribute
def process_values(values: List[ast.AST])->List[str]:
    """Process the values

    :param values: the values
    :type values: List[ast.AST]
    :return: the string representation of the values
    :rtype: List[str]
    """
    processed_values = []
    for value in values:
        if isinstance(value, ast.Name):
            processed_values.append(value.id)
        elif isinstance(value, ast.Attribute):
            processed_values.append(f"self.{extract_attribute(value)}")
    return processed_values

def extract_method_source_and_remove_tab(method: Callable)->str:
    """Extract the source code of a method and remove the tab

    :param method: The method
    :type method: Callable
    :return: The source code of the method
    :rtype: str
    """
    source = inspect.getsource(method)
    lines = source.split("\n")
    non_empty_lines = [line for line in lines if line.strip()]
    if non_empty_lines:
        first_line = non_empty_lines[0]
        indent = len(first_line) - len(first_line.lstrip())
        source_code = "\n".join([line[indent:] for line in lines])
    else:
        source_code = ""
    return source_code

def get_var_num(var):
    """Get the max number of a variable

    :param var: the variable
    :type var: str
    :return: the max number of the variable
    :rtype: int
    """
    #Check if var contains a number
    if not var.split("_")[-1].isdigit():
        return 0
    return int(var.split("_")[-1])

def get_max_var_num(vars):
    """Get the max number of a list of variables

    :param vars: the list of variables
    :type vars: list
    :return: the max number of the variables
    :rtype: int
    """
    if not len(vars):
        return 0
    return max([get_var_num(var) for var in vars])

def parse_assignment(node, assignments, cls):
    targets = node.targets
    for target in targets:
        process_targets(target, node, assignments, cls)

def process_targets(target, node, assignments, cls):
    # if isinstance(target, ast.Name):
    #         var_name = target.id
    #         process_var_name(var_name, node, assignments)
    # elif isinstance(target, ast.Attribute):
    #     var_name = f"self.{extract_attribute(target)}"
    #     process_var_name(var_name, node, assignments)
    # elif isinstance(target, ast.Tuple):
    #     for elt in target.elts:
    #         process_targets(elt, node, assignments)
    # else:
    #     raise Exception(f"Unsupported target type: {type(target)}")
    var_name = get_value(target, cls)
    process_var_name(var_name, node, assignments, cls)

def process_value(value:str, var_name:str, assignments:Dict[str,str]):
    vals = str(value).split(" ")
    for val in vals:
        max_num = get_max_var_num(assignments.keys())
        if max_num:
            val = f'{val}_{max_num}'
        if val in assignments.keys() and 'self.' not in val:
            try:
                value = value.replace(val, assignments[val])
            except TypeError as e:
                value = value.replace(val, str(assignments[val]))
            del assignments[val]
    # here is where I distinguish left from right
    assignments[var_name] = value

def process_var_name(var_name:str, node:ast.AST, assignments:Dict[str,str], cls:object):
    """Process the variable name

    :param var_name: the variable name
    :type var_name: str
    :param node: the node
    :type node: ast.AST
    """
    #Check if var_name in assignments
    #Need to handle repeated var_names that are attributes.
    if var_name in assignments.keys():
        #Add _i to var name
        var_name = f"{var_name}_{get_max_var_num(assignments.keys())+1}"
    value = get_value(node.value, cls)
    process_value(value, var_name, assignments)

def get_variable_assignments(cls: object, method_name: str)->Dict[str,str]:
    """Get the variable assignments of a method

    :param cls: the class with the method
    :type cls: object
    :param method_name: the name of the method
    :type method_name: str
    :return: the variable assignments in the method
    :rtype: dict
    """
    method = getattr(cls, method_name)
    source_code = extract_method_source_and_remove_tab(method)
    tree = ast.parse(source_code)
    assignments = {}
    #Iterate through all nodes in ast and get assignments
    for node in ast.walk(tree):
        #check if node is an assignment and get the left and righthand side values
        if isinstance(node, ast.Assign):
           parse_assignment(node, assignments, cls)
        #check if if statement
        elif isinstance(node, ast.If):
            #pull value out of if statement e.g if self.terminal: retrieve self.terminal
            value = get_value(node.test, cls)
            #generate 16 character random string
            random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))
            process_value(value, random_string, assignments)
            #check conditional of if statement
            # if isinstance(node.test, ast.Compare):
            #     pass
            #check if node has an orelse
            # if node.orelse:
            #     #check if orelse is an assignment
            #     if isinstance(node.orelse[0], ast.Assign):
            #         parse_assignment(node.orelse[0], assignments, cls)
        elif isinstance(node, ast.While):
            #pull value out of if statement e.g if self.terminal: retrieve self.terminal
            value = get_value(node.test, cls)
            #generate 16 character random string
            random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))
            process_value(value, random_string, assignments)
    return assignments  

def replace_intermediate_variables(attributes: Dict[str,str])->Dict[str,str]:
    """Replace intermediate variables in attributes

    :param attributes: the attributes
    :type attributes: Dict[str,str]
    :return: the attributes with intermediate variables replaced
    :rtype: Dict[str,str]
    """    
    intermediate_keys = []
    for key, value in attributes.items():
        if value in attributes.keys():
            attributes[key] = attributes[value]
            intermediate_keys.append(value)
    for key in intermediate_keys:
        del attributes[key]
    return attributes

def remove_self_and_underscore(word: str)->str:
    """Remove self and underscore after self dot

    :param word: the word
    :type word: str
    :return: the word without self and underscore after self dot
    :rtype: str
    """
    if "self." in word:
        #replace first underscore
        return word.replace("self.", "").split(".")[0]
        # if word[0] == "_":
        #     return word[1:]
        # else:
        #     return word
    else:
        return word

#extract all attributes from values in assignments
def extract_all_attributes_from_values_in_assignments(assignments: Dict[str,str])->Dict[str,Set[str]]:
    """Extract all attributes from values in assignments

    :param assignments: the assignments dictionary
    :type assignments: Dict[str,str]
    :return: assignments divided into inputs and outputs
    :rtype: Dict[str,Set[str]]
    """
    attributes = {'inputs':[], 'outputs':[]}
    for key, value in assignments.items():
        if isinstance(value, str):
            if "self." in value:
                d_val = []
                for word in value.split(" "):
                    if "self." in word:
                        d_val.append(remove_self_and_underscore(word))
                attributes['inputs'] += d_val
                attributes['outputs'].append(remove_self_and_underscore(key))
    attributes['inputs'] = list(set(attributes['inputs']))
    attributes['outputs'] = list(set(attributes['outputs']))
    return attributes

def process_words(word_str: str, split_val: str)->List[str]:
    """Process words

    :param word_str: the word string
    :type word_str: str
    :param split_val: the split value
    :type split_val: str
    :return: the processed words
    :rtype: List[str]
    """
    words = []
    for word in word_str.split(split_val):
        if "self." in word:
            words.append(remove_self_and_underscore(word))
    return words

def strip_all_but_class_attribute_names(assignments: Dict[str,str])->Dict[str,str]:
    """Strip all but class attribute names

    :param assignments: the assignments dictionary
    :type assignments: Dict[str,str]
    :return: assignments with only class attribute names
    :rtype: Dict[str,str]
    """
    stripped_assignments = {}
    for key, value in assignments.items():
        if 'self.' not in key:
            key = ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))
        if ", " in key:
            key = tuple(process_words(key, ", "))
        else:
            key = (remove_self_and_underscore(key),)
        if isinstance(value, str):
            if ", " in value:
                words = tuple(process_words(value, ", "))
            else:
                words = process_words(value, " ")
            stripped_assignments[key] = words
        else:
            stripped_assignments[key] = ''
    return stripped_assignments

# def remove_non_io_attrs(stripped_assignments: Dict[str,str], cls: object)->Dict[str,str]:
#     """Remove non io attributes

#     :param assignments: the assignments dictionary
#     :type assignments: Dict[str,str]
#     :param io_attrs: the io attributes
#     :type io_attrs: List[str]
#     :return: the assignments with non io attributes removed
#     :rtype: Dict[str,str]
#     """
#     io_assignments = {}
#     for key, value in stripped_assignments.items():
#         if key in io_attrs:
#             stripped_assignments[key] = value
#     return stripped_assignments

def get_io_groupings(cls: object, method_name: str)->Dict[str,str]:
    """Get the input and output groupings of a method

    :param cls: class with method
    :type cls: object
    :param method_name: name of method
    :type method_name: str
    :return: input and output groupings of method
    :rtype: Dict[str,str]
    """    
    # assignments = replace_intermediate_variables(get_variable_assignments(cls, method_name))
    assignments = get_variable_assignments(cls, method_name)
    return strip_all_but_class_attribute_names(assignments)


def get_inputs_and_outputs(cls: object, method_name: str)->Dict[str,Set[str]]:
    """Get the inputs and outputs of a method

    :param cls: the class with the method
    :type cls: object
    :param method_name: the name of the method
    :type method_name: str
    :return: inputs and outputs of the method
    :rtype: Dict[str,Set[str]]
    """
    assignments = get_variable_assignments(cls, method_name)
    return extract_all_attributes_from_values_in_assignments(replace_intermediate_variables(assignments))

def dictify_list(container: Union[List[str], Dict[str, str], None])-> Dict[str, str]:
    """Convert a list to a dictionary with the same keys and values or return the dictionary

    :param container: The container to convert
    :type container: Union[List[str], Dict[str, str], None]
    :raises TypeError: if the container is not a list or dict
    :return: the converted container
    :rtype: Dict[str, str]
    """    
    if type(container) == list:
        return {val:val for val in container}
    elif type(container) == dict:
        return container
    elif container is None:
        return {}
    else:
        raise TypeError(f'{container} is not list or dict')


class IO:
    """A class to represent the inputs and outputs of a method
    """    
    def __init__(self, io:List[str], inst: 'AbstractComponent'):
        self._outputs = io[0]
        self._method = io[1]
        self._inputs = io[2]
        self._observer_names = []
        self._subject_names, out_rem = self.process_io(self._outputs, inst)
        self._outputs = [self.remove_underscore_number(x) for x in self._outputs if x not in out_rem]
        self._observer_names, inp_rem = self.process_io(self._inputs, inst)
        self._inputs = [self.remove_underscore_number(x) for x in self._inputs if x not in inp_rem]

    def process_io(self, inp_out:List[str], inst: 'AbstractComponent'):
        remove = []
        ret_list = []
        for val in inp_out:
            val = self.remove_underscore_number(val)
            if not hasattr(inst, val) and not hasattr(inst, f"_{val}") and val in inst._io_map:
                ret_list.append(inst._io_map[val])
            else: remove.append(val)
        return ret_list, remove
    
    def remove_underscore_number(self, val:str):
        # check if there is an underscore number at end of string using regex and remove it if there is
        return re.sub(r'_[0-9]+$', '', val)
    
    def __repr__(self) -> str:
        return f'IO({self._subject_names}, {self._observer_names})'

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, IO):
            return False
        return self._subject_names == __value._subject_names and self._observer_names == __value._observer_names and self._method == __value._method

class MasterIO:
    def __init__(self, inst):
        self.inst = inst
        self.io = []
        self.inputs = {}
        self.outputs = {}
        self.observers = {}
        self.subjects = {}
        self.getIO(inst)
        
    def getIO(self, inst):
        if not hasattr(inst, 'abstract_methods'):
            raise TypeError(f'{inst.__name__} does not have attribute abstract_methods')
        for method in inst.abstract_methods + inst._additional_io_methods:
            ios = get_io_groupings(inst, method)
            for key, value in ios.items():
                io = IO([key,method,value], inst)
                self.io.append(io)
                self.process_ios(io)
            self.remove_from_inputs_if_in_outputs()
        for io in self.io:
            self.create_obs_sub(io)

    def remove_from_inputs_if_in_outputs(self):
        for key in self.outputs.keys():
            if key in self.inputs.keys():
                del self.inputs[key]

    def process_ios(self, io:IO):
        for i, key in enumerate(io._outputs):
            if key not in self.outputs.keys():
                self.outputs[key] = io._subject_names[i]
            
        for i, value in enumerate(io._inputs):
            if value not in self.inputs.keys():
                self.inputs[value] = io._observer_names[i]

    def create_obs_sub(self, io:IO):
        for i, key in enumerate(io._outputs):
            if key not in self.subjects.keys():
                self.subjects[key] = Subject(io._subject_names[i], self.inst)
            
        for i, value in enumerate(io._inputs):
            if value not in self.outputs.keys():
                self.observers[value] = Observer(io._observer_names[i], self.inst)
    