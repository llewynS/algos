from abc import ABC, abstractmethod

from .experimentrunner import ExperimentRunner

class ScriptGenerator(ABC):
    """Generates the script to be run for HPOptimisation
    """
    @abstractmethod
    def generate_script(self, hp_cmd_str: str) -> str:
        """The function that generates the script. Will either
        return a file path or the string to be executed locally.

        :param hp_cmd_str: the string with arguments from HPGenerator
        :type hp_cmd_str: str
        :return: a file path or local execution string
        :rtype: str
        """
        pass


class ScriptRunner(ExperimentRunner):
    def __init__(self, script_gen: ScriptGenerator):
        self._generator = script_gen