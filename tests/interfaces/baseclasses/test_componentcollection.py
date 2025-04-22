import pytest
import gc

from algos.interfaces import ComponentCollection, DuplicateError, Subject, Observer
from ..abstractbaseclasses.test_component_v2 import ComponentA, ComponentB, ComponentC



class TestComponentCollection:
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
        self.A = ComponentA()
        self.B = ComponentB()
        self.C = ComponentC()
        self.component_collection = ComponentCollection(A = self.A, B = self.B)

    def teardown_method(self):
        """
        Teardown after each test
        """
        Subject._entities = {}
        Observer._entities = {}



    def test_add_component(self):
        """
        Check adding a compoonent correctly works. 
        Should not change the order of reordered keys.
        """
        assert(["A","B"] == self.component_collection.keys)
        self.component_collection["C"] = self.C
        assert(["A","B","C"] == self.component_collection.keys)


    def test_add_non_component(self):
        with pytest.raises(TypeError):
            self.component_collection['zeta'] = 1

    def test_add_duplicate_namespace(self):
        with pytest.raises(DuplicateError):
            self.component_collection['A'] = self.C
        #TypeError should only be called if the name is not duplicated
        with pytest.raises(DuplicateError):
            self.component_collection['A'] = 1

    def test_returns_correct_component(self):
        self.component_collection["C"] = self.C
        assert self.component_collection['A'] == self.A
        assert self.component_collection['B'] == self.B
        assert self.component_collection['C'] == self.C


    def test_run(self):
        self.component_collection["C"] = self.C
        self.component_collection.run()
        assert(self.component_collection["A"]._z.state == 64)
        assert(self.component_collection["B"]._alpha.state == 80)
        assert(self.component_collection["C"]._x.state == 160)
        assert(self.component_collection["C"]._y.state == 40)
        


