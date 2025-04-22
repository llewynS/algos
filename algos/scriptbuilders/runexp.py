import pathlib
import sys
import os
import argparse
import shutil

from typing import Dict, List

from .. import AbstractExperiment, DatabaseLogger

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def check_exps(strat: str) -> bool:
    """Check if an experiment has been defined

    :param strat: the desired experiment strategy
    :type strat: str
    :return: if the experiment has been defined
    :rtype: bool
    """    
    exps = [x for x in AbstractExperiment._register if 'Abstract' not in x]
    return strat in exps


def runexp(experiment: AbstractExperiment, args: Dict[str,str], unknowns:List[str]):
    """Run the experiment with the given arguments

    :param experiment: The experiment to run
    :type experiment: AbstractExperiment
    :param args: The known arguments as defined in if name main block
    :type args: Dict[str,str]
    :param unknowns: The experiment arguments
    :type unknowns: List[str]
    """    
    
    parser = argparse.ArgumentParser(parents=[experiment._parser])
    args = {**args, **vars(parser.parse_args(unknowns))}
    name = args.pop('name', None)
    name = name if name else experiment.__name__
    file_path = pathlib.Path(__file__).parent.parent.parent.parent / 'SB3Experiments' 
    args['file_path'] = file_path if args['path'] == '' else args['path']
    logger = DatabaseLogger(name, exp_metadata_dict=args)
    should_load = False
    if args['path'] == '':
        args.pop('path')
        file_path = pathlib.Path(__file__).parent.parent.parent.parent / 'SB3Experiments' / logger.experiment_name
        try:
            file_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            shutil.rmtree(file_path)
            file_path.mkdir(parents=True, exist_ok=False)
    else:
        file_path = pathlib.Path(args.pop('path')) / "Experiments" / logger.experiment_name
        should_load = file_path.exists()
    experiment = experiment(file_path, args)
    if should_load:
        pass
    else:
        experiment.run()
    experiment.test()