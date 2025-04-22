
from typing import List, Union, Dict, Callable, Tuple
from abc import abstractmethod
from threading import Thread, Event
from ...logger import DatabaseLogger

import time
import sys

from .hyperparameter import AbstractParametered
# from ..patterns import Observer, Subject
from .io import MasterIO

class AbstractComponent(AbstractParametered, Thread):
    """
    The basic core building block of an RLOS experiment. Components are isolated
    algorithmic sections that can interact with one another via Observer, Subject
    singletons. 
    
    A Component uses Subjects to output information and Observers as inputs.
    The use of this paradigm allows components to be agnostic of each other. All
    that is required is that every Subject has an Observer in the Experiment space. 

    Note: The inputs and outputs can be either a list or a dictionary. If they are a list the 
    attributes of the class and the names of the observers/subjects will be each element of the list. If they 
    are a dictionary then the keys will represent the name of the attribute in the class and the item 
    will represent the name of the observer/subject.
    """
    _is_training = True
    _all_instances_started = Event()
    _all_instances_finished = Event()
    _num_live_instances = 0
    _error_event = Event()

    def __init__(self, io_map: Dict[str,str], additional_io_methods:List[str]=[], *args, **kwargs):
        """
        Constructor of the Component base class

        :param inputs: A dict of list or dict of names that contain the input namespace 
        :type inputs: Dict[Union[Dict[str,str],List[str]]]
        :param outputs: A dict of list or dict of names that contain the output namespace
        :type outputs: Dict[Union[Dict[str,str],List[str]]]
        """
        super().__init__(*args, **kwargs)
        if DatabaseLogger._instance is not None:
            self._logger_inst = DatabaseLogger._instance.bind(f'{self.__class__.__name__}')
        else:
            self._logger_inst = None
        self._io_map = io_map
        self._additional_io_methods = additional_io_methods
        self.io = MasterIO(self)
        self._inputs = self.io.inputs
        self._outputs = self.io.outputs
        
        
    def _generate_sub_obs(self):
        [self.__setattr__(f'_{key}', val) for key,val in self.io.subjects.items()]
        [self.__setattr__(f'_{key}', val) for key, val in self.io.observers.items()]


    def _process_io(self):
        """
        Sets inputs/outputs as attributes of the class

        :param inp: Either the input or output of the component
        :type inp: Union[Dict[str,str],List[str]]
        :raises TypeError: input must be a dictionary or list
        """
        self._generate_sub_obs()
        
        #Getter for property
        def observer_getter_factory(name):
            def getter(self):
                # while getattr(self, f'_{name}').accessed:
                #     time.sleep(0.0005)
                return self.__dict__[f'_{name}'].state
            return getter
        
        #getter for subject property
        def getter_factory(name):
            def getter(self):
                return self.__dict__[f'_{name}'].state
            return getter
        
        #Setter for property
        def setter_factory(name):
            def setter(self, value):
                self.__dict__[f'_{name}'].state = value
            return setter
        
        # Create the private attributes and properties
        for k, _ in self.io.inputs.items():
            #Use getter factory as a property of the class
            setattr(self.__class__, k, property(observer_getter_factory(f'{k}'), None))
        for k, _ in self.io.outputs.items():
            setattr(self.__class__, k, property(getter_factory(f'{k}'), setter_factory(f'{k}')))
        
    def run(self):
        self._all_instances_started.wait()
        AbstractComponent._num_live_instances += 1
        # print(self._num_live_instances)
        try:
            self.do_run()
            self.done()
        except Exception as e:
            AbstractComponent._error_event.set()
            self.done()
            raise e
        self._all_instances_finished.wait()
        print(f'{self.__class__.__name__} finished')
    
    @abstractmethod
    def do_run(self):
        """
        Threads need to define run
        """        
        pass           

    @abstractmethod
    def save(self):
        """
        Save the component state so that an experiment can be resumed.
        """
        pass

    @abstractmethod
    def load(self):
        """
        Load the a saved component to resume correct state. 
        """
        pass

    def done(self):
        AbstractComponent._num_live_instances -= 1
        # print(self._num_live_instances)
        if self._num_live_instances == 0:
            # print('All instances finished')
            self._all_instances_finished.set()