from ..interfaces.abstractbaseclasses.component import AbstractComponent

class ProxyLogger:
    def __init__(self, logger, component_name):
        self._logger = logger
        self._component_name = component_name
        self._component_id = self.get_component_id()

    def __getattr__(self, name):
        return getattr(self._logger, name)
    
    def record(self, tag_name, input_value, *args, **kwargs):
        tag_name = f"{'train' if AbstractComponent._is_training else 'eval'}/{tag_name}"
        self._logger.record(self._component_id, self._component_name, tag_name, input_value)

    def get_component_id(self):
        component_id = self._logger._workers[0].get_component_id(self._component_name)
        return component_id