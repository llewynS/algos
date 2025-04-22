from multiprocessing import Queue, Event

from queue import Empty
import re
import time
import logging
from datetime import datetime
import psycopg2

from sqlalchemy import create_engine, and_, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError, OperationalError

from .tables import DataPoint, Evaluation, Tag, QUERY_BATCH, ExpMetadata, Experiment, Base, Component

# logging.basicConfig(level=logging.DEBUG)

class DBWorker:
    

    def __init__(self, queue: Queue, stop_signal: Event): # type: ignore
        self.db_url = None
        self.main_experiment_id = None
        self.backup_db_url = None
        self.backup_experiment_id = None

        self.queue = queue
        self.stop_signal = stop_signal
        self.timeout = False
        self.timeout_time = 0
        self.experiment_id = None
        #set the current experiment id
        self.main_tag_cache = {}
        self.backup_tag_cache = {}
        self.main_component_cache = {}
        self.backup_component_cache = {}

        self.main_datapoint_queries = []
        self.backup_datapoint_queries = []
        self.main_evaluation_queries = []
        self.backup_evaluation_queries = []
        self.main_component_names = []
        self.backup_component_names = []

    
    def _is_backup(self):
        return self.experiment_id == self.backup_experiment_id

    def load_tags(self):
        session = self.Session_main()
        tags = []
        try:
            tags = session.query(Tag).all()
            self.main_tag_cache = {'ids': {tag.tag_id: tag.name for tag in tags}, 'names': {tag.name: tag.tag_id for tag in tags}}
            session.close()
        except OperationalError:
            session.close()
        session = self.Session_backup()
        tags_bu = session.query(Tag).all()            
        self.backup_tag_cache = {'ids': {tag.tag_id: tag.name for tag in tags_bu}, 'names': {tag.name: tag.tag_id for tag in tags_bu}}
        for tag in tags:
            if tag.name not in self.backup_tag_cache['names']:
                t = Tag(name=tag.name)
                session.add(t)
        session.commit()
        tags_bu = session.query(Tag).all()
        self.backup_tag_cache = {'ids': {tag.tag_id: tag.name for tag in tags_bu}, 'names': {tag.name: tag.tag_id for tag in tags_bu}}      
        session.close()

    def load_components(self):
        session = self.Session_main()
        components = []
        try:
            components = session.query(Component).all()
            self.main_component_cache = {'ids': {component.component_id: component.name for component in components}, 'names': {component.name: component.component_id for component in components}}
            session.close()
        except OperationalError:
            session.close()
        session = self.Session_backup()
        components_bu = session.query(Component).all()
        self.backup_component_cache = {'ids': {component.component_id: component.name for component in components_bu}, 'names': {component.name: component.component_id for component in components_bu}}
        for component in components:
            if component.name not in self.backup_component_cache['names']:
                c = Component(name=component.name)
                session.add(c)
        session.commit()
        components_bu = session.query(Component).all()
        self.backup_component_cache = {'ids': {component.component_id: component.name for component in components_bu}, 'names': {component.name: component.component_id for component in components_bu}}
        session.close()

    def convert_datapoint_queries(self):
        if self._is_backup():
            for _ in self.main_datapoint_queries.copy():
                query = self.main_datapoint_queries.pop(0)
                query['experiment_id'] = self.backup_experiment_id
                tag_name = self.main_tag_cache['ids'][query['tag_id']]
                query['tag_id'] = self.backup_tag_cache['names'][tag_name]
                component_name = self.main_component_cache['ids'].get(query['component_id'])
                query['component_id'] = self.backup_component_cache['names'][component_name]
                self.backup_datapoint_queries.append(query)
        else:
            for _ in self.backup_datapoint_queries.copy():
                query = self.backup_datapoint_queries.pop(0)
                query['experiment_id'] = self.main_experiment_id
                tag_name = self.backup_tag_cache['ids'][query['tag_id']]
                query['tag_id'] = self.main_tag_cache['names'][tag_name]
                component_name = self.backup_component_cache['ids'].get(query['component_id'])
                query['component_id'] = self.main_component_cache['names'][component_name]
                self.main_datapoint_queries.append(query)

    def convert_evaluation_queries(self):
        if self._is_backup():
            for _ in self.main_evaluation_queries.copy():
                query = self.main_evaluation_queries.pop(0)
                query['experiment_id'] = self.backup_experiment_id
                self.backup_evaluation_queries.append(query)
        else:
            for _ in self.backup_evaluation_queries.copy():
                query = self.backup_evaluation_queries.pop(0)
                query['experiment_id'] = self.main_experiment_id
                self.main_evaluation_queries.append(query)

    def convert_queries(self):
        self.convert_datapoint_queries()
        self.convert_evaluation_queries()

    @property
    def tag_cache(self):
        return self.main_tag_cache if not self._is_backup() else self.backup_tag_cache
    
    @property
    def component_cache(self):
        return self.main_component_cache if not self._is_backup() else self.backup_component_cache
    
    @property
    def datapoint_queries(self):
        return self.main_datapoint_queries if not self._is_backup() else self.backup_datapoint_queries
    
    @property
    def evaluation_queries(self):
        return self.main_evaluation_queries if not self._is_backup() else self.backup_evaluation_queries

    @property
    def Session(self):
        # logging.debug(f"Attempting to get Session")
        if not self.check_timeout():
            try:
                # Try to connect to the main database
                connection = self._engine.connect()
                connection.close()
                self.timeout = False
                if self.experiment_id is None:
                    self.experiment_id = self.main_experiment_id
                if self._is_backup():
                    self.experiment_id = self.main_experiment_id
                    self.convert_queries()
                logging.debug(f"Got main experiment")
                return self.Session_main
            except OperationalError:
                # If connection fails, use the backup database
                self.timeout = True
                self.timeout_time = time.time()
                if not self._is_backup():
                    self.experiment_id = self.backup_experiment_id
                    self.convert_queries()
                logging.debug(f"Got backup experiment")
                return self.Session_backup
        else:
            return self.Session_backup
    
    def check_timeout(self):
        # logging.debug(f"Timeout: {self.timeout}")
        if self.timeout:
            delta = time.time() - self.timeout_time
            return delta < 120
        else: 
            return False
        
    def check_run(self):
        keep_running =  not (self.stop_signal.is_set() and self.queue.empty())
        if not keep_running:
            logging.debug(f"Queue Loop Finished because stop_signal:{self.stop_signal.is_set()} and queue empty:{self.queue.empty()}")
        return keep_running

    def run(self):
        logging.debug(f"Starting Debug Worker")
        #Assumes postgresql
        if 'sqlite' in self.db_url:
            logging.debug(f"Creating sqlite engine")
            self._engine = create_engine(self.db_url)
        else:
            logging.debug(f"Creating postgresql engine")
            self._engine = create_engine(self.db_url, pool_pre_ping=True, connect_args={'connect_timeout': 10})
        logging.debug(f"Creating backup engine")
        self._engine_backup = create_engine(self.backup_db_url, pool_pre_ping=True)
        logging.debug(f"Creating session main")
        self.Session_main = sessionmaker(bind=self._engine)
        logging.debug(f"Creating session backup")
        self.Session_backup = sessionmaker(bind=self._engine_backup)
        logging.debug(f"Creating tags")
        self.load_tags()
        logging.debug(f"Creating components")
        self.load_components()
        logging.debug(f"Starting main loop")
        while self.check_run():
            #protect against empty self.queue
            logging.debug(f"Checking if queue is empty")
            if self.queue.empty():
                logging.debug(f"queue is empty")
                time.sleep(2)
            else:
                #two processes means maybe we do get empty self.queue so don't block
                try:
                    logging.debug(f"pulling from queue")
                    obj, method = self.queue.get(False)
                except Empty:
                    continue
                try:
                    if method == 'add_data_point':
                        try:
                            tag_id = self.get_tag_id(obj)
                            component_id = self.get_component_idz(obj)
                            # Create data point with tag ID
                            _ = self.Session
                            data_point = dict(experiment_id=self.experiment_id, tag_id=tag_id, component_id=component_id, step=obj.step,
                                                int_value=obj.int_value, float_value=obj.float_value,
                                                bool_value=obj.bool_value, string_value=obj.string_value,
                                                timestamp=obj.timestamp)
                            self.datapoint_queries.append(data_point)
                            if self._is_backup():
                                self.backup_component_names.append(obj.component_name)
                            else:
                                self.main_component_names.append(obj.component_name)
                        except Exception as e:
                            logging.debug(f"Failed to add obj: {obj} with method: {method} with error: {e}")
                            continue
                    elif method == "add_experiment_result":
                        try:
                            _ = self.Session
                            evaluation = dict(
                                experiment_id=self.experiment_id,
                                init_position=obj.init_position,
                                goal_value=obj.goal_value,
                                reward=obj.reward,
                                success=obj.success,
                                step=obj.step
                            )
                            self.evaluation_queries.append(evaluation)
                        except Exception as e:
                            logging.debug(f"Failed to add obj: {obj} with method: {method} with error: {e}")
                    else:
                        raise ValueError(f"Invalid method {method}")
                    logging.debug(f"Checking to add datapoints")
                    if len(self.datapoint_queries) > QUERY_BATCH:
                        try:
                            self.insert_bulk_data_points(self.datapoint_queries)
                        except Exception as e:
                            logging.debug(f"Failed to add obj: {obj} with method: {method} with error: {e}")
                    logging.debug(f"Checking to add evaluations")
                    if len(self.evaluation_queries) > 0:
                        try:
                            self.insert_bulk_evaluations(self.evaluation_queries)
                        except Exception as e:
                            logging.debug(f"Failed to add obj: {obj} with method: {method} with error: {e}")
                except Exception as e:
                    logging.debug(f"Failed to add obj: {obj} with method: {method} with error: {e}")
        logging.debug(f"Exiting Debug Worker")
        if len(self.datapoint_queries):
            self.insert_bulk_data_points(self.datapoint_queries)
        if len(self.evaluation_queries):
            self.insert_bulk_evaluations(self.evaluation_queries)
        

    
    def insert_bulk_evaluations(self, evaluation_queries):
        session = self.Session()
        session.bulk_insert_mappings(Evaluation, evaluation_queries)
        session.commit()
        session.close()
        evaluation_queries.clear()

    def insert_bulk_data_points(self, datapoint_queries):
        try:
            session = self.Session()
            session.bulk_insert_mappings(DataPoint, datapoint_queries)
            session.commit()
        except IntegrityError as e:
            session.rollback()
            if 'postgresql' in session.bind.engine.url.__str__():
                constraint_name = e.orig.diag.constraint_name
                if constraint_name == 'uix_exp_tag_comp_id_step':
                    pattern = r'\((.*?)\)'
                    matches = re.finditer(pattern, e.orig.diag.message_detail)
                    counter = 0
                    for match in matches:
                        if counter%2:
                            values = match.group(1).split(', ')
                            offending_data = dict(zip(keys, values))
                            logging.debug(f"REMOVING Offending data: {offending_data}")
                            session.query(DataPoint).filter(
                                and_(
                                    DataPoint.experiment_id == offending_data['experiment_id'],
                                    DataPoint.tag_id == offending_data['tag_id'],
                                    DataPoint.component_id == offending_data['component_id'],
                                    DataPoint.step == offending_data['step']
                                )
                            ).delete()
                            session.commit()
                        else:
                            keys = match.group(1).split(', ')
                        counter += 1
                    segments = self.split_list_into_four(datapoint_queries)
                    for segment in segments:
                        self.insert_bulk_data_points(segment)
            elif 'sqlite' in session.bind.engine.url.__str__():
                duplicates = session.query(DataPoint).group_by(
                    DataPoint.experiment_id, DataPoint.tag_id, DataPoint.component_id, DataPoint.step
                ).having(func.count('*') > 1).all()
                for duplicate in duplicates:
                    session.delete(duplicate)
                session.bulk_insert_mappings(DataPoint, datapoint_queries)
                session.commit()
        except OperationalError as e:
            session.close()
            time.sleep(5)
            segments = self.split_list_into_four(datapoint_queries)
            for segment in segments:
                self.insert_bulk_data_points(segment)
        except Exception as e:
            logging.debug(f"Failed to insert datapoints with error: {e}")
            raise e
        finally:
            session.close()
        datapoint_queries.clear()

    def split_list_into_four(self, lst):
        # Calculate the length of each segment
        length = len(lst)
        segment_size = length // 4
        extra = length % 4

        # Create the segments
        segments = []
        for i in range(4):
            start = i * segment_size + min(i, extra)
            end = start + segment_size + (1 if i < extra else 0)
            segments.append(lst[start:end])

        return segments

    def get_component_idz(self, obj):
        if not self._is_backup():
            return obj.component_id
        else:
            component_name = self.main_component_cache['ids'][obj.component_id]
            return self.backup_component_cache['names'][component_name]
        
    def get_tag_id(self, obj):
        # Check if tag ID is in the cache
        tag_id = self.tag_cache['names'].get(obj.tag_name, None)    
        # If not, check if tag with given name exists in the database
        session = self.Session()
        # If not, create it
        if tag_id is None:
            tag = Tag(name=obj.tag_name)
            try:
                session.add(tag)
                session.commit()
                if not self._is_backup():
                    bu_session = self.Session_backup()
                    bu_session.add(tag)
                    bu_session.commit()
                    bu_session.close()
            except Exception as e:
                session.rollback()
                time.sleep(5.0)
                tag = session.query(Tag).filter_by(name=obj.tag_name).first()
                if tag is None:
                    raise e  
            # Add tag name and ID to the cache
            tag_id = tag.tag_id
            self.tag_cache['names'][obj.tag_name] = tag_id
            session.close()
        return tag_id
    
    def add_exp_metadata(self, metadata_dict):
        self._add_exp_metadata(self.db_url, self.experiment_id, metadata_dict)
        if self.db_url != self.backup_db_url:
            self._add_exp_metadata(self.backup_db_url, self.backup_experiment_id, metadata_dict)
    
    def _add_exp_metadata(self, db_url, experiment_id, metadata_dict):
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        for key, value in metadata_dict.items():
            session.add(ExpMetadata(experiment_id=experiment_id, key=key, value=str(value)))
        session.commit()
        session.close()
        engine.dispose()

    def get_unique_experiment_name(self, base_name, session):
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
    
    def create_experiment(self, db_url, backup_db_url, name, start_time=None):
        self.db_url = db_url
        self.backup_db_url = backup_db_url
        try:
            experiment_id, name = self._create_experiment_direct(db_url, name, start_time)
        except OperationalError:
            logging.warning("Failed to connect to the main database. Using the backup database.")
            self.db_url = backup_db_url
            experiment_id, name = self._create_experiment_direct(backup_db_url, name, start_time)
        self.main_experiment_id = experiment_id
        if self.db_url != self.backup_db_url:
            backup_experiment_id, _ = self._create_experiment_direct(backup_db_url, name, start_time)
            self.backup_experiment_id = backup_experiment_id
        else:
            backup_experiment_id = experiment_id
            DBWorker.backup_experiment_id = experiment_id
        self.experiment_id = experiment_id
        return experiment_id, backup_experiment_id, name

    def _create_experiment_direct(self, db_url, name, start_time=None):
        if 'sqlite' in db_url:
            engine = create_engine(db_url)
        else:
            engine = create_engine(db_url, connect_args={'connect_timeout': 10})
        Session = sessionmaker(bind=engine)
        session = Session()
        Base.metadata.create_all(engine)
        successfully_created = False
        experiment = session.query(Experiment).filter_by(name=name).first()
        if experiment is not None:
            experiment_id = experiment.experiment_id
            successfully_created = True
        while not successfully_created:
            try:
                if start_time is None:
                    start_time = datetime.now()
                name = self.get_unique_experiment_name(name, session)
                new_experiment = Experiment(name=name, start_time=start_time)
                session.add(new_experiment)
                session.commit()
                experiment_id = new_experiment.experiment_id
                successfully_created = True
            except (psycopg2.errors.UniqueViolation, IntegrityError):
                session.rollback()
        session.close()
        engine.dispose()
        return experiment_id, name
    
    def experiment_end(self):
        self._experiment_end(self.db_url, self.experiment_id)
        self._experiment_end(self.backup_db_url, self.backup_experiment_id)

    def _experiment_end(self, db_url, experiment_id):
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        experiment = session.query(Experiment).get(experiment_id)
        if experiment is not None:
            experiment.end_time = datetime.now()
            session.commit()
        session.close()
        engine.dispose()

    def get_component_id(self, component_name):
        try:
            main_component_id = self._get_component_id(self.db_url, component_name)
        except OperationalError:
            main_component_id = None
        _ = self._get_component_id(self.backup_db_url, component_name)
        return main_component_id

    def _get_component_id(self, db_url, component_name):
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        component = session.query(Component).filter_by(name=component_name).first()           
        # If not, create it
        if component is None: 
            new_component = Component(name=component_name)
            session.add(new_component)
            session.commit()
            component_id = new_component.component_id  
            
        else:
            component_id = component.component_id
        session.close()
        engine.dispose()
        return component_id
