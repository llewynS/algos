import pytest
import time

from algos.interfaces import Subject, Observer, AbstractComponent

class RunnerDummyInterface(AbstractComponent):
    abstract_methods = ["initiate_sequence", "step"]
    def __init__(self):
        super().__init__(self._io_map)

    def do_run(self):
        self.initiate_sequence()
        for i in range(3):
            self.step()

class ComponentA(RunnerDummyInterface):
    """
    Simple Component
    """
    _io_map = {"x":"x", "y":"y", "z":"z"}

    def initiate_sequence(self):
        pass

    def step(self):
        temp = self.x * self.y
        self.z = temp

    def save(self):
        pass

    def load(self):
        pass

    @classmethod
    def set_up_hyperparameters(cls):
        pass

class ComponentB(RunnerDummyInterface):
    """
    Simple Component
    """
    _io_map = {"x":"x", "z":"z", "alpha":"alpha"}
    
    def initiate_sequence(self):
        pass

    def step(self):
        self.alpha = self.x + self.z

    def save(self):
        pass

    def load(self):
        pass

    @classmethod
    def set_up_hyperparameters(cls):
        pass

class ComponentC(RunnerDummyInterface):
    """
    Simple Component
    """
    _io_map = {"x":"x", "y":"y", "alpha":"alpha"}
    
    def initiate_sequence(self):
        self._x.initialise_state(1)
        self._y.initialise_state(1)

    # @gate_observers("alpha")
    def step(self):
        temp = self.alpha*2
        self.x = temp
        temp = self.alpha/2
        self.y = temp

    def save(self):
        pass

    def load(self):
        pass

    @classmethod
    def set_up_hyperparameters(cls):
        pass

class TestComponent:
    """
    Test the input output functionality of Components. 

    Test that the normal conventions of Observers and Subjects are obeyed. 
    """
    @classmethod
    def setup_class(cls):
        """ 
        Set-up prior to all tests.
        """
        Subject._entities = {}
        Observer._entities = {}

    @classmethod
    def teardown_class(cls):
        """ 
        Teardown after all tests
        """
        Subject._entities = {}
        Observer._entities = {}

    def setup_method(self):
        """
        Setup prior to each test
        """
        Subject._entities = {}
        Observer._entities = {}

    def teardown_method(self):
        """
        Teardown after each test
        """
        Subject._entities = {}
        Observer._entities = {}
        
    def test_components_will_thread(self):
        A = ComponentA()
        B = ComponentB()
        C = ComponentC()

        A._process_io()
        B._process_io()
        C._process_io()

        threads = [A,B,C]
        for t in threads:
            t.start()
        A._all_instances_started.set()
        for t in threads:
            t.join()

        assert(A.z == 64)
        assert(B.alpha == 80)
        assert(C.x == 160)
        assert(C.y == 40)
        
