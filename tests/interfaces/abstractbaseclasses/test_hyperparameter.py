import pytest
from typing import List, Tuple

from .test_component_v2 import ComponentA
from algos.interfaces import get_class_args, Subject, Observer


class DummyComponentWithHPs(ComponentA):
    """
    Adds hyperparameters to Component.__init__ while having the abstractmethod
    definitions from DummyComponent
    """
    def __init__(self,
                 int_val: int = 0,
                 list_int_val: List[int] = [1],
                 str_val: str = 'hello',
                 str_list_val: List[str] = ['world'],
                 str_tuple_val: Tuple[str] = ('foo'),
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def set_up_hyperparameters(cls):
        cls.hyperparameters['int_val'].bounds = (-10, 10)


class InheritsHPs(DummyComponentWithHPs):
    """
    For checking HP inheritance and namespace adherence
    """
    def __init__(self, a_new_str_hp: str = 'peanuts', *args, **kwargs):
        super().__init__(*args, **kwargs)


class TestHPs:
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

    def test_hyperparameters(self):
        dummy_hp = DummyComponentWithHPs.hyperparameters.copy()
        inherited_dummy_hps = InheritsHPs.hyperparameters.copy()
        self.validate_dummy_hps(dummy_hp)
        self.validate_dummy_hps(inherited_dummy_hps, True)

    def test_argparser(self):
        dummy_parser = DummyComponentWithHPs._parser
        inherited_parser = InheritsHPs._parser
        dummy_args = vars(dummy_parser.parse_args([]))
        inherited_args = vars(inherited_parser.parse_args([]))
        dummy_class_args = get_class_args(DummyComponentWithHPs, dummy_args)
        dummy_inherited_args = get_class_args(InheritsHPs, inherited_args)
        self.validate_dummy_hps(dummy_class_args)
        self.validate_dummy_hps(dummy_inherited_args, inherited=True)

    def test_hp_bounds(self):
        dummy_hp = DummyComponentWithHPs.hyperparameters.copy()
        assert dummy_hp['int_val'].bounds == (-10, 10)
        assert dummy_hp['str_val'].bounds is None

    def validate_dummy_hps(self, dic, inherited=False):
        #Check all hps and remove from dict
        dic.pop('additional_io_methods')
        assert 'int_val' in dic.keys()
        dic.pop('int_val')
        assert 'list_int_val' in dic.keys()
        dic.pop('list_int_val')
        assert 'str_val' in dic.keys()
        dic.pop('str_val')
        assert 'str_list_val' in dic.keys()
        dic.pop('str_list_val')
        assert 'str_tuple_val' in dic.keys()
        dic.pop('str_tuple_val')
        if inherited:
            assert 'a_new_str_hp' in dic.keys()
            dic.pop('a_new_str_hp')
        #Ensure no unexpected HPs
        assert dic == {}
