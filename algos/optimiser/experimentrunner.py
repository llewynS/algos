from abc import ABC, abstractmethod

class ExperimentRunner(ABC):
    @abstractmethod
    def run(self, hp_cmd_str: str) -> float:
        """Runs the experiment and returns the result

        :param hp_cmd_str: the string with arguments from HPGenerator
        :type hp_cmd_str: str
        :return: the result of the experiment
        :rtype: float
        """
        pass
