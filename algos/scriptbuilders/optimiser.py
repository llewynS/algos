import sys
import os
import optuna
import argparse
import pathlib
import threading
import time
from typing import Dict, List, Tuple

from ..optimiser.hpoptimiser import OptunaObjective
from ..logger import get_default_storage
from ..interfaces.abstractbaseclasses import AbstractExperiment
from ..experimentrunners import LocalRunner, SBatchGenerator, FilePathGenerator, PexpectSFTPClient, RemoteSBatchSFTPRunner

def get_opt_inps(args:Dict[str, str], 
                 unknowns: List[str]) -> Tuple[str, str, Dict[str, str], str]:
    """Get the inputs for the optimisation from the command line arguments

    :param args: The known arguments
    :type args: Dict[str, str]
    :param unknowns: The arguments related to the experiment strategy
    :type unknowns: List[str]
    :return: The strategy, study name, optimisation kwargs and storage string
    :rtype: Tuple[str, str, Dict[str, str], str]
    """ 
    strategy = args.pop('strategy')
    study_name = args.pop('study_name')
    if study_name == None:
        study_name = strategy
    parser = argparse.ArgumentParser(parents=[AbstractExperiment._register[strategy]._parser])
    storage_string = get_default_storage() if args['storage'] is None else args['storage']
    args.pop('storage')
    if args['name'] == '':
        args.pop('name')
    parsed_unknowns = vars(parser.parse_args(unknowns))
    unknowns = [val.replace('--','').replace('-','_') for val in unknowns]
    non_default_args = {k: v for k, v in parsed_unknowns.items() if k in unknowns}
    opt_kwargs = {**args, **non_default_args}
    return strategy, study_name, opt_kwargs, storage_string

def get_storage(storage_string:str)->optuna.storages.RDBStorage:
    """Get the storage object for the experiment

    :param storage_string: the SQLAlchemy RDB String
    :type storage_string: str
    :return: The RDBStorage object
    :rtype: optuna.storages.RDBStorage
    """   
    storage = optuna.storages.RDBStorage(
                url=storage_string,
                engine_kwargs={'pool_recycle': 3600, 'pool_pre_ping': True}
            )
    return storage

def create_and_run_remote_study(strategy: str, 
                                study_name: str, 
                                opt_kwargs: Dict[str,str], 
                                storage: optuna.storages.RDBStorage,
                                remote_prompt: str,
                                remote_path: str,
                                sbatch_config: str,
                                config_kwargs: Dict[str,str] = None,
                                ssh_name: str = None,
                                file_path: str = None,
                                storage_string: str = None
                                ) -> None:
    """Create and run a remote study

    :param strategy: The experiment
    :type strategy: str
    :param study_name: The name of the optuna study
    :type study_name: str
    :param opt_kwargs: The static optimiser kwargs
    :type opt_kwargs: Dict[str,str]
    :param storage: The storage object
    :type storage: optuna.storages.RDBStorage
    :param remote: the name of the remote host
    :type remote: str
    :param remote_prompt: the prompt on your remote
    :type remote_prompt: str
    :param remote_path: the location to store experiment files on the remote host
    :type remote_path: str
    """   
    sampler = opt_kwargs.pop('sampler', 'tpe')
    study = optuna.create_study(storage = storage,
                                    study_name = study_name, 
                                    direction='maximize', 
                                    load_if_exists=True,
                                    sampler = create_sampler(sampler))
    file_path = file_path / 'SB3ExperimentsBatchFiles' / study.study_name
    generator = SBatchGenerator(study.study_name, sbatch_config, config_kwargs)
    sftp = PexpectSFTPClient(ssh_name, remote_prompt)
    file_path_generator = FilePathGenerator(file_path, remote_path)
    runner = RemoteSBatchSFTPRunner(file_path, sftp, generator, file_path_generator)
    objective = OptunaObjective(AbstractExperiment._register[strategy], runner, opt_kwargs=opt_kwargs, storage=storage_string)
    study.optimize(objective.objective, n_trials=1, n_jobs=1)

def create_sampler(sampler:str):
    if sampler == 'tpe':
        print("Using TPE Sampler")
        return optuna.samplers.TPESampler()
    elif sampler == 'gp':
        print("Using GP Sampler")
        return optuna.samplers.GPSampler(deterministic_objective=True)
    elif sampler == 'cmaes':
        print("Using CMAES Sampler")
        return optuna.samplers.CmaEsSampler()
    else:
        raise ValueError("Invalid sampler")

def run_local(args: Dict[str, str], unknowns: List[str], runner_script: str):
    """Run the experiments locally

    :param args: the known arguments
    :type args: Dict[str, str]
    :param unknowns: the experiment arguments
    :type unknowns: List[str]
    """
    N_JOBS = args.pop('num_jobs')
    sampler = args.pop('sampler', 'tpe') 
    assert runner_script is not None  
    strategy , study_name, opt_kwargs, storage_string = get_opt_inps(args, unknowns)
    
    for _ in range(N_JOBS):
        storage = get_storage(storage_string)
        study = optuna.create_study(storage = storage,
                                    study_name = study_name, 
                                    direction='maximize', 
                                    load_if_exists=True,
                                    sampler = create_sampler(sampler))
        runner = LocalRunner(runner_script, storage_string)
        objective = OptunaObjective(AbstractExperiment._register[strategy], runner, opt_kwargs=opt_kwargs, storage=storage_string)
        study.optimize(objective.objective, n_trials=1, n_jobs=1)

def run_remote(args, unknowns, remote_prompt, remote_path, sbatch_config, config_kwargs, ssh_name):
    N_JOBS = args.pop('num_jobs')
    path = args.pop('path', None)
    assert path is not None
    strategy, study_name, opt_kwargs, storage_string = get_opt_inps(args, unknowns)
    for _ in range(N_JOBS):
        storage=get_storage(storage_string)
        create_and_run_remote_study(strategy, study_name, opt_kwargs, storage, remote_prompt, remote_path, sbatch_config, config_kwargs, ssh_name, file_path=pathlib.Path(path), storage_string=storage_string)

def runopt(args, unknowns, remote_prompt:str=None, remote_path:str=None, sbatch_config:str=None, config_kwargs:Dict[str,str]=None, ssh_name:str=None):
    computer = args.pop('computer', None)
    assert computer is not None
    if computer == "remote":
        assert remote_prompt is not None
        assert remote_path is not None
        assert ssh_name is not None
    if computer == "local":
        runner_script = args.pop('runner_script', None)
        assert runner_script is not None
    N_RUNS = args.pop('num_runs', 5)
    threads = []    
    for _ in range(N_RUNS):
        if computer == 'remote':
            thread = threading.Thread(target=run_remote, args=(args.copy(), unknowns.copy(), remote_prompt, remote_path, sbatch_config, config_kwargs, ssh_name))
            thread.start()
            threads.append(thread)
        if computer == 'local':
            thread = threading.Thread(target=run_local, args=(args.copy(), unknowns.copy(), runner_script))
            thread.start()
            threads.append(thread)
        time.sleep(30)
    for thread in threads:
        thread.join()