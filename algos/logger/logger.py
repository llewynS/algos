from sqlalchemy import create_engine, func, case
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import sessionmaker
import multiprocessing
from datetime import datetime

from typing import Dict,Any
import pathlib
import re
import numpy as np
import psycopg2
import os
from optuna.storages._rdb.models import StudyModel

from .databaseworker import DBWorker
from .tables import Base, Experiment, Evaluation, ExpMetadata, Tag, Component
from .util import create_connection_string_from_yaml, module_dir, DataPointInfo, EvaluationPoint

from ..interfaces.patterns.singleton import SingletonMeta




class DatabaseLogger(metaclass=SingletonMeta):
    _instance = None
    _is_training = True

    def __init__(self, base_name, 
                 db_url = None, experiment_start_time=None, 
                 num_workers=2, exp_metadata_dict:Dict[str,Any]={}, frequency=10):
        experiment_dir = exp_metadata_dict.get('file_path', None)
        if db_url is None:
            db_url = get_default_storage()
        if db_url is None:
            db_file = f'{experiment_dir}/primary.db' if experiment_dir else 'primary.db'
            if not isinstance(db_file, pathlib.Path):
                db_file = pathlib.Path(db_file)
            if not db_file.exists():
                db_file.parent.mkdir(parents=True, exist_ok=True)
                db_file.touch(exist_ok=True)
            db_url = f'sqlite:///{db_file}'
        elif ".yaml" in db_url:
            db_url = create_connection_string_from_yaml(db_url)
        backup_db_file = f'{experiment_dir}/backup.db' if experiment_dir else 'backup.db'
        if not isinstance(backup_db_file, pathlib.Path):
            backup_db_file = pathlib.Path(backup_db_file)
        if not backup_db_file.exists():
            backup_db_file.parent.mkdir(parents=True, exist_ok=True)
            backup_db_file.touch(exist_ok=True)
        self._backup_db_url = f'sqlite:///{backup_db_file}'
        if not db_url:
            db_url = self._backup_db_url
        manager = multiprocessing.Manager()
        self.queue = manager.Queue()
        self.stop_signal = manager.Event()
        self.db_url = db_url
        self._step = 0
        self._component_map = {}
        dbworker = DBWorker(self.queue, self.stop_signal)
        experiment_id, backup_experiment_id, self.experiment_name = dbworker.create_experiment(self.db_url, self._backup_db_url, base_name, experiment_start_time)
        dbworker.add_exp_metadata(exp_metadata_dict)
        self.db_url = dbworker.db_url
        self._backup_db_url = dbworker.backup_db_url
        self._workers = [dbworker]
        self.frequency = frequency

        self.pool = multiprocessing.Pool(processes=num_workers)
        self.results = []
        for i in range(num_workers):
            if i>0:
                dbworker = DBWorker(self.queue, self.stop_signal)
                dbworker.experiment_id = experiment_id
                dbworker.backup_experiment_id = backup_experiment_id
                dbworker.db_url = self.db_url
                dbworker.backup_db_url = self._backup_db_url
                self._workers.append(dbworker)
            self.results.append(self.pool.apply_async(self._workers[i].run))

    def create_experiment(self, name, description, start_time, end_time=None):
        self.queue.put((Experiment(name=name, description=description, start_time=start_time, end_time=end_time), 'add'))
        
    def create_tag(self, name, description=None):
        self.queue.put((Tag(name=name, description=description), 'add'))
        
    def record(self, component_id, component_name, tag_name, input_value, *args, ignore_frequency=False, **kwargs):
        if (self._step % self.frequency and not ignore_frequency):
            return
        int_value, float_value, bool_value, string_value, timestamp = None, None, None, None, None
        if isinstance(input_value, (int, np.int32, np.int64)):
            int_value = int(input_value)
        elif isinstance(input_value, (float, np.float32, np.float64)):
            float_value = float(input_value)
        elif isinstance(input_value, (bool, np.bool_)):
            bool_value = bool(input_value)
        elif isinstance(input_value, (str, np.str_)):
            string_value = str(input_value)
        else:
            raise TypeError(f'Invalid data type {type(input_value)}')
        timestamp = timestamp or datetime.now()
        self.queue.put((DataPointInfo(tag_name=tag_name, 
                                      int_value=int_value,
                                      component_id = component_id,
                                      component_name=component_name,
                                      float_value=float_value,
                                      bool_value=bool_value, 
                                      string_value=string_value,
                                      step = self._step,
                                      timestamp=timestamp), 
                                      'add_data_point'))
        
    def record_evaluation(self, init_position, goal_value, reward, success, step):
        self.queue.put((EvaluationPoint(init_position=str(init_position), 
                                         goal_value=str(goal_value),
                                         reward=float(reward),
                                         success=bool(success),
                                         step=int(step)), 
                                         'add_experiment_result'))
    
    def set_step(self, value):
        self._step = value

    def close(self):
        print("Closing Logger...")
        self.stop_signal.set()
        print("stop signal set")
        self.pool.close()
        self.pool.join()
        self._workers[0].experiment_end()
        self._instance = None

    def bind(self, component_name):
        return ProxyLogger(self, component_name)
    
    

class ProxyLogger:
    def __init__(self, logger, component_name):
        self._logger = logger
        self._component_name = component_name
        self._component_id = self.get_component_id()

    def __getattr__(self, name):
        return getattr(self._logger, name)
    
    def record(self, tag_name, input_value, *args, **kwargs):
        tag_name = f"{'train' if DatabaseLogger._is_training else 'eval'}/{tag_name}"
        self._logger.record(self._component_id, self._component_name, tag_name, input_value)

    def get_component_id(self):
        component_id = self._logger._workers[0].get_component_id(self._component_name)
        return component_id


def get_coverage(experiment_name: str, db_url:str = None):
    # Create a SQLAlchemy engine and session
    if db_url is None:
        db_url = get_default_storage()
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    # Query for the experiment ID using the experiment name
    experiment = session.query(Experiment).filter(Experiment.name == experiment_name).first()
    success_rate_values = []
    if experiment:
        success_counts = (
            session.query(
                    Evaluation.step,
                    func.count().filter(Evaluation.success.is_(True)).label('success_count')
                )
                .filter(Evaluation.experiment_id == experiment.experiment_id)
                .group_by(Evaluation.step)
                .order_by(Evaluation.step)
                .all()
        )
        # Extract step and success rate values
        step_values = [row.step for row in success_counts]
        success_count_values = [row.success_count for row in success_counts]
        total_test_counts = [session.query(Evaluation).filter(Evaluation.experiment_id == experiment.experiment_id, Evaluation.step == step).count() for step in step_values]
        success_rate_values = [float(success_count) / float(total_tests) for success_count, total_tests in zip(success_count_values, total_test_counts)]
    session.close()
    engine.dispose()
    if len(success_rate_values)==0:
        return None
    return success_rate_values[-1], step_values[-1]

def get_coverages(experiment_name: str, db_url:str = None):
    # Create a SQLAlchemy engine and session
    if db_url is None:
        db_url = get_default_storage()
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    # Query for the experiment ID using the experiment name
    experiment = session.query(Experiment).filter(Experiment.name == experiment_name).first()
    success_rate_values = []
    if experiment:
        success_counts = (
            session.query(
                    Evaluation.step,
                    func.count().filter(Evaluation.success.is_(True)).label('success_count')
                )
                .filter(Evaluation.experiment_id == experiment.experiment_id)
                .group_by(Evaluation.step)
                .order_by(Evaluation.step)
                .all()
        )
        # Extract step and success rate values
        step_values = [row.step for row in success_counts]
        success_count_values = [row.success_count for row in success_counts]
        total_test_counts = [session.query(Evaluation).filter(Evaluation.experiment_id == experiment.experiment_id, Evaluation.step == step).count() for step in step_values]
        success_rate_values = [float(success_count) / float(total_tests) for success_count, total_tests in zip(success_count_values, total_test_counts)]
    session.close()
    engine.dispose()
    if len(success_rate_values)==0:
        return None
    return success_rate_values, step_values
    

def get_unique_experiment_name(base_name, session):
    """
    This function checks if an experiment name exists in the database and, if it does,
    increments a number appended to the name by 1.
    """
    # Regular expression to find a number at the end of the name
    regex = re.compile(r'(.*)_(\d+)$')
    
    # Query the database for names like the base name
    similar_names = session.query(Experiment.name).filter(
        Experiment.name.ilike(f"{base_name}%")
    ).all()
    
    max_number = -1
    for (name,) in similar_names:
        match = regex.match(name)
        if match:
            # If the name matches the pattern, update max_number
            prefix, number = match.groups()
            if prefix.lower() == base_name.lower():
                max_number = max(max_number, int(number))
        elif name.lower() == base_name.lower():
            # If the name is exactly base_name, ensure we at least get _1
            max_number = max(max_number, 0)
    
    # If max_number is 0, the base name is unique and can be used as is
    # Otherwise, append _{max_number+1} to make it unique
    return f'{base_name}_0' if max_number < 0 else f"{base_name}_{max_number + 1}"

def create_experiment_direct(name, db_url=None, start_time=None, study_id=None):
    if db_url is None:
        db_url = create_connection_string_from_yaml(f'{module_dir}/config.yaml')
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    Base.metadata.create_all(engine)
    successfully_created = False   
    while not successfully_created:
        try:
            if start_time is None:
                    start_time = datetime.now()
            name = get_unique_experiment_name(name, session)
            new_experiment = Experiment(name=name, start_time=start_time, study_id=study_id)
            session.add(new_experiment)
            session.commit()
            experiment_id = new_experiment.experiment_id
            successfully_created = True
        except (psycopg2.errors.UniqueViolation, IntegrityError):
            session.rollback()
    session.close()
    engine.dispose()
    return experiment_id, name

def get_default_storage(db_name:str = None) -> str:
    db_url = os.environ.get("ALGOS_DB_URL")
    if db_url:
        return db_url
    # check if the file exists
    file = os.environ.get("ALGOS_DB_FILE")
    if file and not os.path.exists(file):
        raise FileNotFoundError(f"Configuration file {file} does not exist.")
    elif not file:
        return
    return create_connection_string_from_yaml(file, db_name)