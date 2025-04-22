import pytest


class TestGeneric:
    @classmethod
    def setup_class(cls):
        """ 
        Set-up prior to all tests.
        """
        pass

    @classmethod
    def teardown_class(cls):
        """ 
        Teardown after all tests
        """
        pass

    def setup_method(self):
        """
        Setup prior to each test
        """
        pass

    def teardown_method(self):
        """
        Teardown after each test
        """
        pass

    def test_a(self):
        pass

    def test_b(self):
        pass