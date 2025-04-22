import pytest
from algos.interfaces import TypedDict


class Test():
    pass


class ExtTest(Test):
    pass


class SomethingElse():
    pass


class TestClass():
    def test(self):
        a = TypedDict(Test)
        assert a._value_klass == Test
        assert a == {}
        a["first"] = Test()
        a["second"] = ExtTest()
        assert type(a['first']) == Test
        assert type(a['second']) == ExtTest
        with pytest.raises(TypeError):
            a[1] = Test()
        with pytest.raises(TypeError):
            a["third"] = SomethingElse()
        del a["first"]
        a["FiRsT"] = Test()
        assert type(a['first']) == Test