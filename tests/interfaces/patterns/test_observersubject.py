import pytest
from algos.interfaces import Subject, Observer, DuplicateError
class event_mock:
    def is_set(self):
        return False
class thread_mock:
    _error_event = event_mock()
    def is_alive(self):
        return True
    

class TestObserverSubject:
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
        self.suba = Subject('abcd', thread_mock())
        self.oba = Observer('abcd', thread_mock())

    def teardown_method(self):
        """
        Teardown after each test
        """
        Subject._entities = {}
        Observer._entities = {}

    def test_connect_subject_observer(self):
        assert self.suba._observers == set([self.oba])

    def test_state_change(self):
        assert self.suba.state is None
        self.suba.state = 5
        assert self.suba.state == 5
        assert self.oba.state == self.suba.state
        # self.oba._lock.release()
        self.suba.state = 10
        assert self.suba.state == 10
        assert self.oba.state == self.suba.state
        with pytest.raises(AttributeError):
            self.oba.state = 10

    def test_create_same_subject(self):
        with pytest.raises(DuplicateError):
            Subject('abcd', self)

    def test_create_subject_without_observer(self):
        Subject('b', self)
        with pytest.raises(ValueError):
            Subject.verify_obs_subs()

    def test_create_multiple_subject_observer(self):
        subb = Subject('bfg', thread_mock())
        obsb = Observer('bfg', thread_mock())
        subc = Subject('cem', thread_mock())
        obsc = Observer('cem', thread_mock())
        #Assert correct observers assigned
        assert subb._observers == set([obsb])
        assert subc._observers == set([obsc])
        #Assert no observers are crossed or duplicated with different names
        assert obsb not in self.suba._observers
        assert obsb is not obsc
        assert obsb is not self.oba
        subb.state = 5
        #assert only obsb.state as changed
        assert obsb.state == subb.state
        assert obsc._state != obsb._state
        assert self.oba._state != obsb._state
        assert subc.state != obsb._state
