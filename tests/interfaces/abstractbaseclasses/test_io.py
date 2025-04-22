from algos.interfaces.abstractbaseclasses.io import MasterIO, get_inputs_and_outputs

from .test_component_v2 import ComponentA

class TestMasterIO:
    def test_get_inputs_and_outputs(self):
        io = get_inputs_and_outputs(ComponentA, 'step')
        assert io['inputs'] == ['x', 'y']
        assert io['outputs'] == ['z']

    def test_init(self):
        inst = ComponentA()
        io = inst.io
        assert io.inst == inst
        assert len(io.io) == 1
        assert io.inputs == {'x': 'x', 'y': 'y'}
        assert io.outputs == {'z': 'z'}

