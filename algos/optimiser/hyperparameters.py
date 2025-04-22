import argparse
from ..interfaces import AbstractExperiment, HyperParameter


class ExperimentHyperParameters:
    _hyperparameters = None

    @property
    def hyperparameters(self):
        return self._hyperparameters


class DefaultExperimentHyperParameters(ExperimentHyperParameters):
    def __init__(self, experiment: AbstractExperiment):
        self._experiment = experiment
        self.get_cmd_args()
        self.get_hyperparameters()

    def get_cmd_args(self):
        self._parser = argparse.ArgumentParser(
            parents=[self._experiment._parser])
        self._cmd_args = list(self._parser._option_string_actions.keys())

    def get_hyperparameters(self):
        self._hyperparameters = {}
        for builder in self._experiment._builders:
            for component in builder._register:
                for k, v in component.hyperparameters.items():
                    hp_name = f'--{component.__name__}-{k}'.replace('_', '-')
                    self._hyperparameters[hp_name] = v


class CustomExperimentHyperParameters(ExperimentHyperParameters):
    def __init__(self, custom_hp_dict: dict):
        """For custom HP optimisation

        Contains cmd-arg as key and tuple containing optimisation bounds and dist as value

        :param custom_hp_dict:  {--key-name - str:
                                (lower bound - Union(int,float),
                                 upper bound - Union(int,float))}
        :type custom_hp_dict: dict
        """
        self.get_hyperparameters(custom_hp_dict)

    def get_hyperparameters(self, custom_hp_dict: dict):
        self._hyperparameters = {}
        for k, v in custom_hp_dict.items():
            self.hyperparameters[k] = HyperParameter(None, bounds=v,
                                                     htype=type(v[0]))
