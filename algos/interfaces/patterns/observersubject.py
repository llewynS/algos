from typing import Any

import time
from functools import wraps
import numpy as np

from . import SingleEntity, HybridSingleEntity
from ..core_imports import Lock

class Observer:
    """An Observer from the observer pattern. Uses a name as an id.

    Define an updating interface for objects that should be notified of
    changes in a subject.

    To save space only one observer for each subject is created (quasi singleton)
    When observers are created if one already exists with the name space then 
    the existing observer is given 
    """
    _pending = set()
    _entities = {}

    def __init__(self, name: str, owner: object):
        """Constructor of Observer
        
        :param name: This Observer's name
        :type name: str
        """
        assert isinstance(name, str), f'name: {name} must be a string'
        self._state = None
        self._name = name
        self._owner = owner
        self._lock = Lock()
        self._num_access = 1 if not hasattr(self, '_num_access') else self._num_access
        self._accessed = self._num_access

    def __new__(cls, name:str, owner: object, *args, **kwargs):
        """
        On creation attempts to attach Observer to Subject

        If Subject does not already exist adds Observer to pending
        """
        self = cls.create_new(name, owner, *args, **kwargs)
        try:
            Subject._entities[name].attach(self)
        except KeyError:
            cls._pending.add(self)
        return self
    
    @classmethod
    def create_new(cls, name:str, owner: object, *args, **kwargs):
        """
        Creates a new Observer and attaches it to the Subject
        """
        key = (name, owner)
        if key in cls._entities:
            #print(f'Observer {name} got')
            #entity has been created
            self = cls._entities[key]
            #Increment number of times to access observer
            self._num_access += 1
            return self
        else:
            #print(f'Observer {name} created')
            #create the entity
            new = super().__new__(cls)
            #add entity to the entities dictionary
            cls._entities[key] = new
            return new

    @property
    def name(self) -> str:
        """Gets the Observers name (read only)
        
        :return: The name of this Observer
        :rtype: str
        """
        return self._name

    @property
    def state(self) -> Any:
        """Get Observer's state
        
        :return: Return the Observer state
        :rtype: Any
        """
        counter = 0
        while self.accessed:
            time.sleep(0.01)
            counter += 1
            if counter > 40000:
                raise ValueError(f'Observer {self} has been waiting for {counter*0.01} seconds')
            if self._owner._error_event.is_set():
                raise ValueError(f'Error event erroring out of Observer: {self}')
        self._accessed += 1
        ret_val = self._state
        return ret_val

    def _set_state(self, value: Any):
        """Set the Observer's state
        Should only be called by Subject       
        :param value: The new state for the Observer
        :type value: Any
        """
        # self._lock.acquire()
        self._accessed = 0
        self._state = value
        # self._lock.release()

    def __repr__(self) -> str:
        """Representation of Observer
        
        :return: string representation of Observer
        :rtype: str
        """
        return f"Observer('{self.name}', {self._owner})"
    @property
    def accessed(self):
        return self._accessed == self._num_access

class Subject(SingleEntity):
    """A ConcreteSubject from the Observer pattern. Uses a name as an id.
    """
    _entities = {}

    def __init__(self, name: str, owner: object):
        """Constructor of Subject
        
        :param name: Name of subject
        :type name: str
        """
        assert isinstance(name, str), f'name: {name} must be a string'
        self._observers = set()
        self._owner = owner
        self._state = None
        self._name = name
        self._lock = Lock()
        self.attach_pending_observers()
    
    def attach_pending_observers(self):
        temp = Observer._pending.copy()
        for observer in Observer._pending:
            if self.name == observer.name:
                self.attach(observer)
                temp.remove(observer)
        if temp != Observer._pending:
            Observer._pending = temp

    def attach(self, observer: Observer):
        """Attach an observer to the subject
                
        :param observer: the observer to be attached
        :type observer: Observer
        """
        observer._subject = self
        self._observers.add(observer)

    def detach(self, observer: Observer):
        """Detach an observer from the subject
        
        :param observer: the observer to be detached
        :type observer: Observer
        """
        self._observers.remove(observer)

    def _notify(self):
        """Notify the observers of a change in state
        """
        [observer._set_state(self.state) for observer in self._observers]

    @property
    def name(self) -> str:
        """Getter of the name of the Subject
        
        :return: Name of this agent
        :rtype: str
        """
        return self._name

    #Getter and setter (Concrete Subject?)
    @property
    def state(self) -> Any:
        """Getter of subject state
        
        :return: The subject state
        :rtype: Any
        """
        return self._state

    @state.setter
    def state(self, arg: Any):
        """Setter of the subject state, also notifies all observers
        
        :param arg: The subjects new state
        :type arg: Any
        """
        
        while self.check_locks():
            time.sleep(0.01)
        if any([not x._owner.is_alive() for x in self._observers]):
                self._state = arg
                return
        self._assign_state_notify(arg)    

    def _assign_state_notify(self, arg: Any):
        """Assigns the state of the subject and notifies all observers
        
        :param arg: The new state of the subject
        :type arg: Any
        """
        #print(f'Subject {self._name}: {arg}')
        self._state = arg
        self._notify()

    def initialise_state(self, arg: Any, increment: bool = False):
        """Initialise the state of the subject
        
        :param arg: The initial state of the subject
        :type arg: Any
        """
        # self._lock.acquire()
        self._assign_state_notify(arg)
        if increment:
            for observer in self._observers:
                observer._accessed += 1
        # self._lock.release()

    def check_locks(self)->bool:
        #note we could just detach observers as a matter of course when we end a thread lol.
        return any([(not x.accessed) and x._owner.is_alive() for x in self._observers])


    def release_all_locks(self):
        [x._lock.release() for x in self._observers if x._lock.locked()]

    def __repr__(self) -> str:
        """String representation of the Subject
        
        :return: The string representaion of this subject.
        :rtype: str
        """
        return f"Subject('{self.name}', {self._owner})"

    @classmethod
    def verify_obs_subs(cls):
        """
        All pending observers have been removed and every subject has at least one observer
        """
        if len(Observer._pending) > 0:
            raise ValueError(f"{Observer._pending} is not empty")

        for name in cls._entities:
            #Check no subject has no observer
            if not cls._entities[name]._observers:
                raise ValueError(
                    f'{cls._entities[name]} has no attached observer')