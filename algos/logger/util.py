from collections import namedtuple
import yaml
import os
# Get the directory containing the module
module_dir = os.path.dirname(os.path.abspath(__file__))

DataPointInfo = namedtuple('DataPointInfo', ['tag_name', 'int_value',
                                             'component_id', 'component_name','step',
                                             'float_value', 'bool_value', 
                                             'string_value', 'timestamp']
                        )

EvaluationPoint = namedtuple('EvaluationPoint', ['init_position', 
                                                   'goal_value', 
                                                   'reward',
                                                   'success', 
                                                   'step'])

def create_connection_string_from_yaml(yaml_file:str, db_name: str = None)->str:
    """Concerts a yaml file into a connection string

    :param yaml_file: Connection file
    :type yaml_file: str
    :return: The connection string for the SQL database
    :rtype: str
    """
    if os.path.isabs(yaml_file):    
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)

        db_config = config.get('database', {})
        ssl_config = config.get('ssl', {})
        if db_name:
            db_config['dbname'] = db_name

        connection_string = (
            f"{db_config['dbtype']}://"
            f"{db_config['username']}:{db_config['password']}@" if db_config['username'] else '') + (
            f"{db_config['host']}:{db_config['port']}" if db_config['host'] and db_config['port'] else ''
            ) + f"/{db_config['dbname']}"
            
        
        if ssl_config:
            connection_string += (
                f"?sslmode={ssl_config['mode']}&"
                f"sslcert={ssl_config['cert']}&"
                f"sslkey={ssl_config['key']}&"
                f"sslrootcert={ssl_config['rootcert']}"
            )

        return connection_string
    else:
        return ''
