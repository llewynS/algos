import optuna
import numpy as np
from typing import Union

from .hyperparameters import DefaultExperimentHyperParameters
from .experimentrunner import ExperimentRunner
from ..interfaces import AbstractExperiment, HyperParameter
from ..logger import create_experiment_direct
from ..optimiser.sftp import PexpectConnectionError, PexpectTimeoutError

class OptunaObjective:
    def __init__(self, experiment: AbstractExperiment, exp_runner: ExperimentRunner, opt_kwargs:dict, storage:str=None):
        """An objective function class for Optuna to optimise. This class
        is used to generate the hyperparameters for the experiment. It does this
        by parsing the experiments for hyperparameters defined by set_up_hyperparameters.

        :param experiment: The experiment to be optimised
        :type experiment: AbstractExperiment
        :param exp_runner: a class that runs experiment, must have run method that returns a float, defaults to None
        :type exp_runner: ExperimentRunner, optional
        """
        opt_kwargs = self.convert_opt_kwargs(opt_kwargs)
        self._opt_args = {'--strategy': experiment.__name__, **opt_kwargs}
        self._name = self._opt_args.pop('--name', None)
        if self._name is None:
            self._name = experiment.__name__  
        self._experiment = experiment
        self._exp_runner = exp_runner
        self._hp_dict = {}
        self._exp_hps = DefaultExperimentHyperParameters(experiment)
        self._storage = storage

    def objective(self, trial: optuna.trial.Trial) -> float:
        """The objective function for Optuna to optimise

        :param trial: the trial object from Optuna
        :type trial: optuna.trial.Trial
        :return: the result of the experiment
        :rtype: float
        """ 
        self._hp_dict['--name'] = create_experiment_direct(self._name, study_id=trial.study._study_id, db_url=self._storage)[1]
        trial.set_user_attr("name", self._hp_dict['--name'])
        hp_cmd_str = self.parse_hyper_parameters(trial)
        max_retries = 3
        for i in range(max_retries):
            try:
                return self._exp_runner.run(hp_cmd_str,
                                            trial, self._storage)
            except (PexpectConnectionError, PexpectTimeoutError) as e:
                raise e
            except Exception as e:
                print(f'Error occurred: {e}')
                print(f'Retrying {i + 1}/{max_retries}')
                trial.set_user_attr("failed", True)
                raise e
        return None

    def convert_opt_kwargs(self, opt_kwargs: dict) -> str:
        """Convert the optuna kwargs into a string that can be parsed by the experiment

        :param opt_kwargs: the optuna kwargs
        :type opt_kwargs: dict
        :return: a string that can be parsed by the experiment
        :rtype: str
        """
        temp_dict = {}      
        for key, val in opt_kwargs.items():
            if not key.startswith('--'):
                key = f'--{key}'.replace('_', '-')
            temp_dict[key] = val
        return temp_dict

    def parse_hyper_parameters(self, trial:optuna.trial.Trial) -> str:
        """Parse the hyper parameters of the experiment and determine 
        if they should have a suggestion made.

        :param trial: the current optuna trial
        :type trial: optuna.trial.Trial
        :return: the string to be fed to the experiment runner
        :rtype: str
        """        
        remove = []
        for hp_name, hp in self._exp_hps.hyperparameters.items():
            length = self._parse_length(trial, hp_name, hp)
            suggestions = []
            for i in range(length):
                name = hp_name
                if length > 1: name += f'_{i}'
                suggestion = self._make_suggestion(trial, name, hp, i)
                if suggestion not in [None, (), {}, []]:
                    suggestions.append(str(suggestion))
                else:
                    remove.append(name)
            if suggestions not in [None, (), {}, []]:
                self._hp_dict[hp_name] = suggestions
        [self._hp_dict.pop(x, None) for x in remove]
        temp_dict = self._hp_dict.copy()
        [self._hp_dict.pop(x, None) for x in temp_dict if self._hp_dict[x] in [None, (), {}, []]]
        self._hp_dict = {**self._hp_dict, **self._opt_args}
        return self.convert_hp_dict_str()

    def convert_hp_dict_str(self) -> str:
        """Convert the hyperparameter dictionary into a string that can be used by the exprunner CLI

        :return: the experiment parameters as a string for exprunner
        :rtype: str
        """        
        hp_str = ''
        for key, val in self._hp_dict.items():
            if hasattr(val, '__len__') and not isinstance(val, str):
                val = [str(x) for x in val]
                hp_str += f"{key} {' '.join(val)} "
            else:
                hp_str += f"{key} {val} "
        return hp_str

    def _parse_length(self, trial: optuna.trial.Trial, hp_name: str,
                      hyperparameter: HyperParameter) -> int:
        """Determine if parsing should be done for the hyperparameter

        :param trial: the current optuna trial
        :type trial: optuna.trial.Trial
        :param hp_name: the name of the hyperparameter
        :type hp_name: str
        :param hyperparameter: The hyperparameter object
        :type hyperparameter: HyperParameter
        :raises NotImplementedError: errors if hyperparameter type is not accounted for
        :return: returns the length of the hyperparameter
        :rtype: int
        """        
        if type(hyperparameter.length) in (tuple, list):
            if hyperparameter.type == np.ndarray:
                raise NotImplementedError(
                    'Not actually sure what this type is' +
                    'for but you should be now so fix this')
            else:
                assert len(hyperparameter.length) == 2
                return trial.suggest_int(hp_name.strip('--')+"-length",
                                         hyperparameter.length[0],
                                         hyperparameter.length[1])
        else:
            return hyperparameter.length

    def _make_suggestion(self, trial: optuna.trial.Trial, hp_name: str,
                         hyperparameter: HyperParameter, index: int)->Union[int, float, None]:
        """Make a suggestion for the hyperparameter

        :param trial: the current trial
        :type trial: optuna.trial.Trial
        :param hp_name: the name of the hyperparameter
        :type hp_name: str
        :param hyperparameter: the hyperparameter object
        :type hyperparameter: HyperParameter
        :param index: _description_
        :type index: int
        :raises NotImplementedError: errors if type not considered (not numeric)
        :return: The suggestion
        :rtype: Union[int, float, None]
        """        
        hp_name = hp_name.strip('--')
        if hyperparameter.bounds is None:
            return None
        elif hyperparameter.type == int or type(
                hyperparameter.bounds[0]) == int:
            return trial.suggest_int(hp_name, hyperparameter.bounds[0],
                                     hyperparameter.bounds[1])
        elif hyperparameter.type == float or type(
                hyperparameter.bounds[0]) == float:
            return trial.suggest_float(hp_name, hyperparameter.bounds[0],
                                         hyperparameter.bounds[1])
        else:
            raise NotImplementedError(
                f'{hyperparameter} not yet accounted for')