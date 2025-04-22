import pytest
import time

from algos.interfaces import Subject, Observer

from .test_component_v2 import RunnerDummyInterface, ComponentA, ComponentB, ComponentC


class ComponentD(RunnerDummyInterface):
    """
    Simple Component
    """
    _io_map = {"y":"y", "yy":"yy"}

    def initiate_sequence(self):
        pass

    def step(self):
        self.yy = self.y*2

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
        ComponentA._io_map = {"x":"x", "y":"yy", "z":"z"}
        A = ComponentA()
        B = ComponentB()
        C = ComponentC()
        D = ComponentD()

        A._process_io()
        B._process_io()
        C._process_io()
        D._process_io()

        threads = [A,B,C,D]
        for t in threads:
            t.start()
        A._all_instances_started.set()
        for t in threads:
            t.join()

        assert(A.z == 1152)
        assert(B.alpha == 1200 )
        assert(C.x == 2400)
        assert(C.y == 600)
        
