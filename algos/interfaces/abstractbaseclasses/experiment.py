from abc import ABC, abstractmethod, abstractclassmethod
import pathlib
import argparse
import time

from ..baseclasses import ComponentCollection
from ..utility import check_create, save_json
from ..metaclasses.hyperparameter import HyperParser
from ...logger import DatabaseLogger


class ExperimentParser(HyperParser):
    _register = {}


class AbstractExperimentParser(type(ABC), ExperimentParser):
    pass


class AbstractExperiment(ABC, metaclass=AbstractExperimentParser):
    """Abstract experiment class for running core experiments

    Provides an interface as well as loading and saving methods.
    Will generate an experiment folder and add a json config for 
    all serialisable objects in the experiment.
    """
    _builders = []
    _tester = None
    _load = False

    def __init__(self, file_path: pathlib.Path,
                 exp_args: dict, *args, load = False, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)
        
        self._exp_args = exp_args
        self._file_path = file_path
        self._exp_args['file_path'] = str(file_path.parent)
        self._json = self._file_path / 'config.json'
        self._results = self._file_path / 'results.json'
        if not load:
            self._save()
        # self._component_path = self._file_path / 'components'
        # check_create(self._component_path)
        if DatabaseLogger._instance is None:
            self._logger = DatabaseLogger(str(self._file_path.stem), exp_metadata_dict=exp_args)
        else:
            self._logger = DatabaseLogger._instance
        self._components = ComponentCollection()
        self._trainer = None
        self.build()
        self._reorder_components()
        # self._tester = self._tester(self).build()
        if load:
            self._load()

    def run(self):
        self._components.run()
        logger_timeout = 0
        logger_qsize = 0
        while not self._logger.queue.empty():
            # for res in self._logger.results:
            #     res.get()
            new_logger_qsize = self._logger.queue.qsize()
            if new_logger_qsize != logger_qsize:
                logger_timeout = 0
                logger_qsize = new_logger_qsize
            else:
                logger_timeout += 1
            print(f"Waiting for logger to finish...queue size: {logger_qsize}")
            
            if logger_timeout > 10:
                break
            time.sleep(1)
        print("Closing Logger...")      
        self._logger.close()

    def test(self):
        if self._tester is None:
            return
        scalar, results = self._tester.test()

        results_dict = {
            'scalar': scalar,
            'results': list(results),
        }
        if self._tester._tester._initial_states is not None:
            initials = self._tester._tester._initial_states.tolist()
            results_dict['initial_states'] = initials
        if hasattr(self._tester, '_goals'):
            results_dict['goals'] = self._tester._goals._goal_schedule.tolist()
        save_json(self._results, results_dict)

    def build(self):
        temp_builders = self._builders[:]
        for i, builder in enumerate(temp_builders):
            self._builders[i] = builder(self)

    def _save(self):
        check_create(self._json)
        save_json(self._json, self._exp_args)

    def _reorder_components(self):
        pass

    @classmethod
    def harvest_parameters(cls):
        cls._parser = cls._arg_parser()
        cls.experiment_params()

    @classmethod
    def _arg_parser(cls):
        parents = [
            klass._parser for builder in cls._builders
            for klass in builder._register if hasattr(klass, "_parser")
        ]
        return argparse.ArgumentParser(parents=parents, add_help=None)

    @classmethod
    def experiment_params(cls):
        #Adds class experiment specific arguments to the experiment
        cls._parser.add_argument('--db-url', default=None, type=str, help='Database URL or Config file')

    def _load(self):
        for component in self._components:
            component.load()

    @property
    def components(self):
        return self._components
