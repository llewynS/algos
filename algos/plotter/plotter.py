import pandas as pd
from sqlalchemy import create_engine, func, and_, case, MetaData, select, column
from sqlalchemy.orm import sessionmaker, aliased
from itertools import groupby
from optuna.storages._rdb.models import StudyModel, TrialModel, TrialValueModel, TrialUserAttributeModel
from matplotlib.font_manager import FontProperties
from typing import List, Dict, Tuple, Union

from ..logger import get_default_storage, Experiment, Tag, Component, DataPoint, ExpMetadata, Evaluation, DataPointInfo, get_coverages

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from scipy import interpolate
import matplotlib.colors as mcolors
import os
import optuna
import shap
import pathlib

from collections import Counter

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

PALETTE_STRING = 'rocket'

def get_data_for_experiments_tag_and_component(experiments, tag_name, component_name, db_url=None):
    # Create a DB session
    if db_url is None:
        db_url = get_default_storage()
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Ensure experiment_numbers is a list
        if not isinstance(experiments, list):
            experiments = [experiments]
        allvalues = []
        timestamps = []
        # Query and plot for each experiment number
        for experiment in experiments:
            if isinstance(experiment, str):
                tempexp = experiment
                experiment = session.query(Experiment).filter_by(name=experiment).first()
            tag = session.query(Tag).filter_by(name=tag_name).first()
            component = session.query(Component).filter_by(name=component_name).first()

            
            if not experiment or not tag or not component:
                print(f"Experiment: {tempexp}, Tag {tag}, or Component {component} not found!")
                continue
            
            data_points = (
                session.query(DataPoint)
                .filter_by(experiment_id=experiment.experiment_id, tag_id=tag.tag_id, component_id=component.component_id)
                .order_by(DataPoint.step)
                .all()
            )
            
            # Extract the data for plotting
            timestamps.append([dp.step for dp in data_points])

            def determine_tag_type(dp):
                float_value = dp.float_value
                if float_value is not None:
                    return 'float_value'
                int_value = dp.int_value
                if int_value is not None:
                    return 'int_value'
                bool_value = dp.bool_value
                if bool_value is not None:
                    return 'bool_value'
                string_value = dp.string_value
                if string_value is not None:
                    return 'string_value'
                
            values = [getattr(dp, determine_tag_type(dp)) for dp in data_points]  # assuming you want to plot float values
            allvalues.append(values)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
    finally:
        session.close()
        engine.dispose()

    return experiments, timestamps, allvalues

def plot_data_for_experiments_tag_and_component(experiments, tag_name, component_name, db_url=None):
   
    plt.figure(figsize=(10, 6))
    
    experiments, timestamps, allvalues = get_data_for_experiments_tag_and_component(experiments, tag_name, component_name, db_url=db_url)
    for experiment, timestamps, values in zip(experiments, timestamps, allvalues):
        plt.plot(timestamps, values, marker='o', linestyle='-', label=f"Experiment {experiment.split('_')[-1]}")
    
    # Finalize the plot
    plt.title(f'Data for Tag: {tag_name}, Component: {component_name}')
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
        


def find_exps(env, reward_type, basename, db_url=None):
    if db_url is None:
        db_url = get_default_storage()
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Create aliases for the metadata table
    env_metadata = aliased(ExpMetadata)
    reward_metadata = aliased(ExpMetadata)

    # Query
    experiments = session.query(Experiment).join(
        env_metadata, Experiment.experiment_id == env_metadata.experiment_id
    ).join(
        reward_metadata, Experiment.experiment_id == reward_metadata.experiment_id
    ).filter(
        and_(env_metadata.key == 'env', env_metadata.value == env),
        and_(reward_metadata.key == 'reward_type', reward_metadata.value == reward_type),
        Experiment.name.like(f"{basename}_%")
    ).all()

    session.close()
    return experiments

def plot_evaluations_for_experiment(experiment_names, db_url=None):
    # Create a SQLAlchemy engine and session
    if db_url is None:
        db_url = get_default_storage()
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Initialize lists to store step and success rate values
    steps = []
    success_rates = []
    if not isinstance(experiment_names, list):
        experiment_names = [experiment_names]
    # Loop through each experiment name
    for experiment_name in experiment_names:
        # Query for the experiment ID using the experiment name
        experiment = session.query(Experiment).filter(Experiment.name == experiment_name).first()
        
        if experiment:
            success_rate, step = get_coverages(experiment.name)
            steps.append(step)
            success_rates.append(success_rate)
    # Close the database session
    session.close()
    engine.dispose()
    return experiment_names, steps, success_rates
    # Create a plot for each experiment
    for experiment_name, step_values, success_rate_values in zip(experiment_names, steps, success_rates):
        plt.plot(step_values, success_rate_values, label=experiment_name)

    # Add labels and legend
    plt.xlabel('Step')
    plt.ylabel('Success Rate')
    plt.legend()

    # Show the plot
    plt.show()

    


def get_best_experiments_in_study(study_name:str, cut_off_value: float, db_url=None):
    if db_url is None:
        db_url = get_default_storage()
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    columns_to_select = [
        StudyModel.study_name,
        TrialModel.number,
        TrialValueModel.value,
        TrialUserAttributeModel.value_json,
    ]
    # Define your custom SQL query using explicit column expressions with column function
    query = (
        session.query(*columns_to_select)
        .join(TrialModel, StudyModel.study_id == TrialModel.study_id)
        .join(TrialValueModel, TrialModel.trial_id == TrialValueModel.trial_id)
        .join(TrialUserAttributeModel, TrialModel.trial_id == TrialUserAttributeModel.trial_id)
        .filter(and_(StudyModel.study_name == study_name, TrialValueModel.value >= cut_off_value))
        .order_by(TrialValueModel.value.desc())
    )

    results = query.all()
    session.close()
    engine.dispose()
    results.sort(key=lambda x: x[2])  # Sort by TrialValueModel.value
    grouped_results = []
    for value, group in groupby(results, key=lambda x: x[2]):
        group_list = []
        for result in group:
            study_name, trial_number, value, user_attribute_value = result
            group_list.append([study_name, trial_number, value, user_attribute_value.strip('"')])
        grouped_results.append(group_list)
    return grouped_results

def get_all_experiments_in_study(study_name:str, db_url=None):
    if db_url is None:
        db_url = get_default_storage()
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    columns_to_select = [
        StudyModel.study_name,
        TrialUserAttributeModel.value_json,
    ]
    # Define your custom SQL query using explicit column expressions with column function
    query = (
        session.query(*columns_to_select)
        .join(TrialModel, StudyModel.study_id == TrialModel.study_id)
        .join(TrialUserAttributeModel, TrialModel.trial_id == TrialUserAttributeModel.trial_id)
        .filter(StudyModel.study_name == study_name)
    )

    results = query.all()
    session.close()
    engine.dispose()
    return [x[1].strip('"') for x in results]

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_with_smoothing(experiments, timestamps, allvalues, smoothing_window=10):
    
    avg_timestamps = [moving_average(t, smoothing_window) for t in timestamps]
    avg_values = [moving_average(v, smoothing_window) for v in allvalues]
    plot_data(experiments, 'Steps', avg_timestamps, 'Episodic Reward', avg_values, xlims=[min(timestamps[0]), max(timestamps[0])])

def plot_data(experiments, xlabel, xvalues, ylabel, yvalues, ylims=[], xlims=[], legend = True, file_path: Union[str, pathlib.Path] = None):
    set_sns_plot_settings(18,18,18)

    pallete = sns.color_palette(PALETTE_STRING, len(experiments))
    fig = plt.figure(figsize=(8, 6))
    for idx, (experiment, x_value, y_value) in enumerate(zip(experiments, xvalues, yvalues)):
            sns.lineplot(x=x_value, y=y_value, linestyle='-', linewidth=3, alpha=0.7, label=f"{experiment}", color = pallete[idx])
    # Finalize the plot
    if ylims:
        plt.ylim(ylims)
    if xlims:
        plt.xlim(xlims)
    plt.xlabel(xlabel, fontweight='bold')
    plt.ylabel(ylabel, fontweight='bold')
    
    if legend:
        plt.legend()
    else:
        plt.legend().remove()
    
    plt.grid(True)
    plt.xticks(rotation=-10)
    plt.tight_layout()
    if file_path:
        file_path = convert_fp_to_pathlib_create_directories(file_path)
        plt.savefig(file_path)
        plt.close() 
    plt.show()

def scatter_plot_data(experiments, xlabel, xvalues, ylabel, yvalues, ylims=[], xlims=[], leg_loc='best', show=True):
    set_sns_plot_settings()
    pallete = sns.color_palette(PALETTE_STRING, len(experiments))
    # fig = plt.figure(figsize=(8, 6))
    plt.figure(figsize=(8, 6)) 
    for idx, (experiment, x_value, y_value) in enumerate(zip(experiments, xvalues, yvalues)):
            sns.scatterplot(x=x_value, y=y_value, alpha=0.4, label=f"{experiment}", color = pallete[idx])
    # Finalize the plot
    if ylims:
        plt.ylim(ylims)
    if xlims:
        plt.xlim(xlims)
    plt.xlabel(xlabel, fontweight='bold')
    plt.ylabel(ylabel, fontweight='bold')
    plt.legend(loc=leg_loc)
    plt.grid(True)
    plt.xticks(rotation=-10)
    if show: 
        plt.show()

def plot_interpolated_studies_episodic_reward(study_names, agent_name, environment_name, mov_avg=None):
    # Plotting
    set_sns_plot_settings()
    plt.figure(figsize=(10, 6))
    pallete = sns.color_palette(PALETTE_STRING, len(study_names))
    for i,(study, agent) in enumerate(zip(study_names, agent_name)):
        exps = get_all_experiments_in_study(study)
        _ , timestamps, allvalues = get_data_for_experiments_tag_and_component(exps,'train/ep_rew', agent)
        if mov_avg:
            allvalues = [moving_average(v, mov_avg) for v in allvalues]
            timestamps = [moving_average(t, mov_avg) for t in timestamps]
        mean_rewards, std_dev_rewards, common_timesteps = interpolate_data(allvalues, timestamps)
        sns.lineplot(x=common_timesteps, y=mean_rewards, linestyle='-', linewidth=3, alpha=0.7,color=pallete[i],label=agent)
        plt.fill_between(common_timesteps, mean_rewards - std_dev_rewards, mean_rewards + std_dev_rewards, color=pallete[i], alpha=0.3)
    plt.xlabel('Time Step')
    plt.ylabel('Episodic reward')
    plt.legend()
    plt.show()


def interpolate_data(allvalues, timestamps):
    mins = []
    maxs = []
    for times in timestamps:
        mins.append(min(times))
        maxs.append(max(times))
    min_time = min(mins)
    max_time = max(maxs)-1000
    common_timesteps = np.linspace(min_time, max_time, 1000)
    interpolated_rewards = []
    for i in range(len(allvalues)):
        interp_func = interpolate.interp1d(timestamps[i], allvalues[i], kind='linear', bounds_error=False, fill_value='extrapolate')
        interpolated_rewards.append(interp_func(common_timesteps))
    interpolated_rewards = np.array(interpolated_rewards)

    # Calculate mean and standard deviation
    mean_rewards = np.mean(interpolated_rewards, axis=0)
    std_dev_rewards = np.std(interpolated_rewards, axis=0)
    return mean_rewards, std_dev_rewards, common_timesteps


def get_goal_values_for_successful_evaluations(experiment_name: str, db_url: str = None, oldsparse=False) -> List[float]:
    """
    Retrieves all goal values from successful evaluations for a given experiment name.

    :param experiment_name: The name of the experiment.
    :param db_url: Optional; The database URL. If None, will be obtained from config.
    :return: A list of goal values for successful evaluations.
    """
    # Create a SQLAlchemy engine and session
    if db_url is None:
        db_url = get_default_storage()
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Query for the experiment ID using the experiment name
    experiment = session.query(Experiment).filter(Experiment.name == experiment_name).first()

    if experiment:
        # Query for goal values where Evaluation.success is True for the specific experiment
        max_step_subquery = (
            session.query(Evaluation.goal_value, func.max(Evaluation.step).label("max_step"))
            .filter(Evaluation.experiment_id == experiment.experiment_id, Evaluation.success.is_(True))
            .group_by(Evaluation.goal_value)
            .subquery()
        )
        if oldsparse:
            successful_evaluations = (
                session.query(Evaluation.goal_value, Evaluation.reward)
                .join(max_step_subquery, and_(
                    Evaluation.goal_value == max_step_subquery.c.goal_value,
                    Evaluation.step == max_step_subquery.c.max_step,
                    Evaluation.experiment_id == experiment.experiment_id,
                ))
                .all()
            )
            successful_goal_values = [evaluation.goal_value for evaluation in successful_evaluations if evaluation.reward > 0]
            failed_goal_values = [evaluation.goal_value for evaluation in successful_evaluations if not evaluation.reward > 0]
        else:
            # Query for goal values with the largest step value for each goal
            successful_evaluations = (
                session.query(Evaluation.goal_value, Evaluation.success)
                .join(max_step_subquery, and_(
                    Evaluation.goal_value == max_step_subquery.c.goal_value,
                    Evaluation.step == max_step_subquery.c.max_step,
                    Evaluation.experiment_id == experiment.experiment_id,
                ))
                .all()
            )
            # Extract goal values
            successful_goal_values = [evaluation.goal_value for evaluation in successful_evaluations if evaluation.success]
            failed_goal_values = [evaluation.goal_value for evaluation in successful_evaluations if not evaluation.success]

    session.close()
    engine.dispose()
    def convert_to_int_list(data):
        result = []
        for item in data:
            # Remove the brackets and split by spaces
            item = item.strip('[]')
            elements = item.split()
            # Convert the elements to integers
            int_elements = [float(x) for x in elements]
            result.append(int_elements)
        return result

    return convert_to_int_list(successful_goal_values), convert_to_int_list(failed_goal_values)

def get_goal_values_for_evaluations_grouped_by_step(experiment_name: str, db_url: str = None, oldsparse=False) -> Dict[int, List[List[float]]]:
    """
    Retrieves all goal values from evaluations for a given experiment name, grouped by step.

    :param experiment_name: The name of the experiment.
    :param db_url: Optional; The database URL. If None, will be obtained from config.
    :return: A dictionary where keys are steps and values are lists of goal values for that step.
    """
    # Create a SQLAlchemy engine and session
    if db_url is None:
        db_url = get_default_storage()
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Query for the experiment ID using the experiment name
    experiment = session.query(Experiment).filter(Experiment.name == experiment_name).first()
    evaluations_grouped_by_step = {}

    if experiment:
        # Query for all evaluations, grouping by step
        evaluations = (
            session.query(Evaluation.step, Evaluation.goal_value, Evaluation.success, Evaluation.reward)
            .filter(Evaluation.experiment_id == experiment.experiment_id)
            .order_by(Evaluation.step)
            .all()
        )

        for evaluation in evaluations:
            step = evaluation.step
            goal_value = evaluation.goal_value
            success = evaluation.success
            reward = evaluation.reward

            if oldsparse:
                if reward > 0:
                    evaluations_grouped_by_step.setdefault(step, [[],[]])[0].append(goal_value)
                else:
                    evaluations_grouped_by_step.setdefault(step, [[],[]])[1].append(goal_value)
            else:
                if success:
                    evaluations_grouped_by_step.setdefault(step, [[],[]])[0].append(goal_value)
                else:
                    evaluations_grouped_by_step.setdefault(step, [[],[]])[1].append(goal_value)

    session.close()
    engine.dispose()

    def convert_to_int_list(data):
        result = []
        for item in data:
            # Remove the brackets and split by spaces
            item = item.strip('[]')
            elements = item.split()
            # Convert the elements to integers
            int_elements = [float(x) for x in elements]
            result.append(int_elements)
        return result

    # Convert goal values to int lists and group by step
    evaluations_grouped_by_step_converted = {step: [convert_to_int_list(goal_values[0]), convert_to_int_list(goal_values[1])] for step, goal_values in evaluations_grouped_by_step.items()}

    return evaluations_grouped_by_step_converted


def plot_maze_with_results(maze, successful_goal_pos):
    # Convert the maze to a numpy array for easier manipulation
    maze = np.array(maze)

    # Create a dictionary to count the number of successful hits for each goal position
    hit_counts = {}
    for pos in successful_goal_pos:
        pos_tuple = tuple(pos)
        if pos_tuple in hit_counts:
            hit_counts[pos_tuple] += 1
        else:
            hit_counts[pos_tuple] = 1

    # Create a color map for the goal hits
    cmap = mcolors.ListedColormap(['white', 'lightgreen', 'green', 'darkgreen', 'darkslategrey'])
    norm = mcolors.BoundaryNorm([0, 1, 2, 3, 4, 5], cmap.N)

    # Create the plot
    fig, ax = plt.subplots()
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            if (maze[y, x]) == '1':
                color = 'black'
            elif maze[y, x] == 'r':
                color = 'red'
            elif maze[y, x] == 'g':
                hits = hit_counts.get((y, x), 0)
                color = cmap(norm(hits))
            else:
                color = 'white'
            print(f"pos: {y, x}, color: {color}, maze[y, x]: {maze[y, x]}, hit_counts: {hit_counts.get((y, x), 0)}")
            rect = plt.Rectangle([x, y], 1, 1, facecolor=color)
            ax.add_patch(rect)
    
    # Set the ticks and labels
    ax.set_xticks(np.arange(maze.shape[1]) + 0.5)
    ax.set_yticks(np.arange(maze.shape[0]) + 0.5)
    ax.set_xticklabels(np.arange(maze.shape[1]))
    ax.set_yticklabels(np.arange(maze.shape[0]))
    ax.invert_yaxis()

    plt.grid(which='both')
    plt.show()

def plot_maze_with_results_for_each_step(maze, evaluations_grouped_by_step):
    # Convert the maze to a numpy array for easier manipulation
    maze = np.array(maze)

    # Create a color map for the goal hits
    cmap = mcolors.ListedColormap(['white', 'lightgreen', 'green', 'darkgreen', 'darkslategrey'])
    norm = mcolors.BoundaryNorm([0, 1, 2, 3, 4, 5], cmap.N)

    for step, goal_positions in evaluations_grouped_by_step.items():
        # Reset hit counts for each step
        hit_counts = {}
        for pos in goal_positions[0]:
            pos_tuple = tuple(pos)
            hit_counts[pos_tuple] = hit_counts.get(pos_tuple, 0) + 1

        # Create the plot for the current step
        fig, ax = plt.subplots()
        ax.set_title(f'Step: {step}')
        for y in range(maze.shape[0]):
            for x in range(maze.shape[1]):
                if maze[y, x] == '1':
                    color = 'black'
                elif maze[y, x] == 'r':
                    color = 'red'
                elif maze[y, x] == 'g':
                    hits = hit_counts.get((y, x), 0)
                    color = cmap(norm(hits))
                else:
                    color = 'white'
                rect = plt.Rectangle([x, y], 1, 1, facecolor=color)
                ax.add_patch(rect)

        # Set the ticks, labels, and grid
        ax.set_xticks(np.arange(maze.shape[1]) + 0.5)
        ax.set_yticks(np.arange(maze.shape[0]) + 0.5)
        ax.set_xticklabels(np.arange(maze.shape[1]))
        ax.set_yticklabels(np.arange(maze.shape[0]))
        ax.invert_yaxis()
        plt.grid(which='both')

        # Display the plot for the current step
        plt.show()


def get_all_goals_for_experiment(experiment_name):
    # Create a SQLAlchemy engine and session
    db_url = get_default_storage()
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Query for the experiment ID using the experiment name
    experiment = session.query(Experiment).filter(Experiment.name == experiment_name).first()
    tag = session.query(Tag).filter_by(name='train/goal').first()
    if experiment and tag:
        # Query for all goal values for the specific experiment from the data_points table
        goals = (
            session.query(DataPoint.string_value)
            .filter_by(experiment_id=experiment.experiment_id, tag_id=tag.tag_id)
            .all()
        )
        session.close()
        goals = [[float(value) for value in goal[0].replace('[', '').replace(']', '').split()] for goal in goals]
        return goals
    
def get_successful_goals_for_experiment(experiment_name):
    # Create a SQLAlchemy engine and session
    db_url = get_default_storage()
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Query for the experiment ID using the experiment name
    experiment = session.query(Experiment).filter(Experiment.name == experiment_name).first()
    # Get all successful evaluations for the experiment only for the most recent steps
    max_step = session.query(func.max(Evaluation.step)).filter(Evaluation.experiment_id == experiment.experiment_id).first()[0]
    successful_evaluations = (
        session.query(Evaluation.goal_value)
        .filter(Evaluation.experiment_id == experiment.experiment_id, Evaluation.step == max_step, Evaluation.success == True)
        .all()
    )
    session.close()

    goals = list(set(tuple([float(value) for value in goal[0].replace('[', '').replace(']', '').split()]) for goal in successful_evaluations))
    #only extract unique goals
    return goals
    
def get_all_goals_for_study(study_name, db_name = None):
    # Create a SQLAlchemy engine and session
    db_url = get_default_storage(db_name=db_name)
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Query for the study ID using the study name
    study_id = session.query(StudyModel.study_id).filter(StudyModel.study_name == study_name).first()[0]
    experiments = session.query(Experiment.name).filter(Experiment.study_id == study_id).all()
    session.close()
    d = {}
    for experiment in experiments:
        goals = get_all_goals_for_experiment(experiment.name)
        # print(f"Goals for experiment {experiment.name}: {goals}")
        d[experiment.name] = goals
    return d

def get_all_data_points_for_experiment(experiment_name, db_url=None):
    """
    Retrieves all data points for a given experiment name.

    :param experiment_name: The name of the experiment.
    :param db_url: Optional; The database URL. If None, will be obtained from config.
    :return: A list of data points for the given experiment.
    """
    # Create a SQLAlchemy engine and session
    if db_url is None:
        db_url = get_default_storage()
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Query for the experiment ID using the experiment name
    experiment = session.query(Experiment).filter(Experiment.name == experiment_name).first()
    data_points = []

    if experiment:
        # Query for all data points for the specific experiment
        data_points = (
            session.query(DataPoint)
            .filter(DataPoint.experiment_id == experiment.experiment_id)
            .all()
        )

    session.close()
    engine.dispose()

    return data_points

def get_all_evaluations_for_experiment(experiment_name, db_url=None):
    """
    Retrieves all evaluations for a given experiment name.

    :param experiment_name: The name of the experiment.
    :param db_url: Optional; The database URL. If None, will be obtained from config.
    :return: A list of evaluations for the given experiment.
    """
    # Create a SQLAlchemy engine and session
    if db_url is None:
        db_url = get_default_storage()
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Query for the experiment ID using the experiment name
    experiment = session.query(Experiment).filter(Experiment.name == experiment_name).first()
    evaluations = []

    if experiment:
        # Query for all evaluations for the specific experiment
        evaluations = (
            session.query(Evaluation)
            .filter(Evaluation.experiment_id == experiment.experiment_id)
            .all()
        )

    session.close()
    engine.dispose()

    return evaluations

def get_experiments_with_suffix(suffix, db_url=None):
    """
    Retrieves all experiments with names ending with the given suffix.

    :param suffix: The suffix to search for in experiment names.
    :param db_url: Optional; The database URL. If None, will be obtained from config.
    :return: A list of experiments with names ending with the given suffix.
    """
    # Create a SQLAlchemy engine and session
    if db_url is None:
        db_url = get_default_storage()
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Query for experiments with names ending with the given suffix
    experiments = session.query(Experiment).filter(Experiment.name.like(f"{suffix}%")).all()

    session.close()
    engine.dispose()

    return experiments

def plot_histogram_of_tag(results:dict, tag_name:str, bins:Union[int,float]=10):

    # Assuming `results` is your dictionary from the previous task
    proxy_results = results.copy()
    for key, value in results.items():
        if not value["hyperparameters"][0].get(tag_name):
            proxy_results.pop(key)

    bin_width = 0.005
    # Extract the number of layers for each experiment with objective value > 0.5
    layer_counts = []
    for study_name, study_data in proxy_results.items():
        for objective_value, params in zip(study_data['objective_values'], study_data['hyperparameters']):
            net_arch_length = round(params.get(tag_name, 0) / bin_width) * bin_width
            if objective_value > 0.5:
                layer_counts.append(net_arch_length)

    # Create a DataFrame with the number of layers
    layer_counts_df = pd.DataFrame({'num_layers': layer_counts})

    # Define bins and x-ticks
    bins = np.arange(min(layer_counts_df['num_layers']), max(layer_counts_df['num_layers']) + bin_width, bin_width)
    x_ticks = np.arange(min(layer_counts_df['num_layers']), max(layer_counts_df['num_layers']) + bin_width, bin_width)

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(layer_counts_df['num_layers'], bins=bins, alpha=0.7, edgecolor='black')
    plt.xlabel(f'Number of {tag_name} (Rounded to Nearest {bin_width})')
    plt.ylabel('Count of Experiments')
    plt.xticks(x_ticks, rotation=45)
    plt.grid(True)
    plt.show()

def save_trial_plot(study_name, file_path):
    file_path = convert_fp_to_pathlib_create_directories(file_path)
    sns.set_theme()
    sns.set_context('talk')
    sns.set_style('darkgrid', {"axes.facecolor": "0.9", "plot.facecolor": "0.9"})

    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['figure.facecolor'] = 'lightgrey'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['xtick.labelsize'] = 18  # Increase tick label size
    plt.rcParams['ytick.labelsize'] = 12  # Increase tick label size
    plt.rcParams['axes.labelsize'] = 18  # Increase axis label size
    plt.rcParams['axes.labelweight'] = 'bold'  # Increase axis 

    PALETTE_STRING = 'rocket'
    palette = sns.color_palette(PALETTE_STRING, 3)
    storage_url = get_default_storage()
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    
    # Extract trial numbers and objective values
    trials = study.trials_dataframe()
    trials = trials[trials['state'] == 'COMPLETE']
    trial_numbers = trials['number']
    objective_values = trials['value']

    # Calculate the cumulative maximum objective value
    cumulative_max = objective_values.cummax()
    cumulative_average = objective_values.expanding().mean()


    # Plot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=trial_numbers, y=objective_values, color=palette[0], alpha=0.7, label='Trial')
    sns.lineplot(x=trial_numbers, y=cumulative_max, color = palette[1], alpha=0.7, label='Cumulative Max')
    sns.lineplot(x=trial_numbers, y=cumulative_average, color=palette[2], label='Cumulative Average')

    plt.xlabel('Trial Number')
    plt.ylabel(r'$\mathbf{f_{objective}}$')
    # plt.title('Trial Number vs. Objective Value')
    plt.legend()
    # plt.legend().remove()
    plt.tight_layout()
    # # Save the figure to file
    plt.savefig(file_path / f'{study_name}_trials.jpg', dpi=600)
    # # Close the plot to suppress automatic display
    plt.close()
    # plt.show()

def save_importance_plot(study_name, file_path, plotting_dict):
    file_path = convert_fp_to_pathlib_create_directories(file_path)
    sns.set_theme()
    sns.set_context('talk')
    sns.set_style('darkgrid', {"axes.facecolor": "0.9", "plot.facecolor": "0.9"})

    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['figure.facecolor'] = 'lightgrey'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['xtick.labelsize'] = 18  # Increase tick label size
    plt.rcParams['ytick.labelsize'] = 12  # Increase tick label size
    plt.rcParams['axes.labelsize'] = 18  # Increase axis label size
    plt.rcParams['axes.labelweight'] = 'bold'  # Increase axis 

    PALETTE_STRING = 'rocket'

    storage_url = get_default_storage()
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    # Extract hyperparameter importance
    importance = optuna.importance.get_param_importances(study)

    # Convert the importance dictionary to a list of tuples and sort it
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    # Separate the parameter names and their importance values
    params, values = zip(*importance)
    plot_names = [plotting_dict[param][1] for param in params]
    # Plot using Seaborn
    sns.barplot(x=values, y=plot_names, hue=plot_names, palette=PALETTE_STRING, legend=False)
    plt.xlabel('Importance')
    plt.ylabel('Hyperparameter')
    plt.title('Hyperparameter Importance')
    plt.tight_layout()
    # Save the figure to file
    plt.savefig(file_path + f'\\{study_name}_importance.jpg', dpi=600)
    # Close the plot to suppress automatic display
    plt.close()

def plot_save_SHAP(results, plotting_dict, combinations, file_path, experiments):
    file_path = os.path.join(file_path, '-'.join(experiments))
    file_path = convert_fp_to_pathlib_create_directories(file_path)

    X = []
    Y = []
    for study_name, study_data in results.items():
        if study_name in experiments:
            for objective_value, params in zip(study_data['objective_values'], study_data['hyperparameters']):
                X.append(params)
                Y.append(objective_value)

    # Convert X and Y to DataFrames
    X = pd.DataFrame(X)
    # change the column names
    for column in X.columns:
        X.rename(columns={column: plotting_dict[column][1]}, inplace=True)
    Y = pd.DataFrame(Y, columns=['objective_value'])
    X.fillna(0, inplace=True)
    print(X.shape)
    print(Y.shape)

    random_state = 42
    # Train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, Y_train)

        # Predict on the test set
    Y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    print(f"Mean Squared Error for {'-'.join(experiments)}: {mse}")
    print(f"R^2 Score for {'-'.join(experiments)}: {r2}")

    cmap = sns.color_palette(PALETTE_STRING, as_cmap=True)
    # palette = sns.color_palette(PALETTE_STRING, as_cmap=True)

    # Create a custom colormap by trimming the top of the range
    # cmap = LinearSegmentedColormap.from_list('trimmed_palette', palette(np.linspace(0, 0.8, 256)))

    sns.set_theme()
    sns.set_context('talk')
    sns.set_style('darkgrid', {"axes.facecolor": "0.6", 
                               "plot.facecolor": "0.6", 
                               "grid.color": "0.5", 
                               "axes.edgecolor": "0.5"})

    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['figure.facecolor'] = 'lightgrey'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['xtick.labelsize'] = 20  # Increase tick label size
    plt.rcParams['ytick.labelsize'] = 20  # Increase tick label size
    plt.rcParams['axes.labelsize'] = 20  # Increase axis label size
    plt.rcParams['axes.labelweight'] = 'bold'  # Increase axis label weight
    # Initialize SHAP explainer
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # SHAP summary plot
    shap.summary_plot(shap_values, X, cmap=cmap, show=False)
    plt.gcf().tight_layout()
    plt.gcf().savefig(os.path.join(file_path, "summary.png"), dpi=600)
    plt.close()
    
    for combo in combinations:
        # SHAP dependence plot
        shap.dependence_plot(combo[0], shap_values.values, X, cmap=cmap, interaction_index=combo[1], show=False)
        plt.tight_layout()
        c1 = combo[0].replace('\\','').replace('$','').replace('mathbf{','').replace('}','').replace('{','')
        c2 = combo[1].replace('\\','').replace('$','').replace('mathbf{','').replace('}','').replace('{','')
        plt.savefig(os.path.join(file_path, f"{c1}_{c2}.png"), dpi=600)
        plt.close()

def plot_and_save_histograms(results: dict, file_path: Union[str, pathlib.Path], bin_width: Union[int, float] = 10,
                             objective_value_threshold: float = 0.8, matching_string: str = 'hidden-layers',
                             legend_shift_factor: int = 1):
    set_sns_plot_settings()
    # Ensure the file path exists
    file_path = convert_fp_to_pathlib_create_directories(file_path)

    hidden_layer_counts = {}

    for study_name, study_data in results.items():
        for objective_value, params in zip(study_data['objective_values'], study_data['hyperparameters']):
            for key, val in params.items():
                if val is None or "length" in key:
                    continue
                if matching_string in key:
                    if key not in hidden_layer_counts:
                        hidden_layer_counts[key] = []
                    net_arch_length = round(val / bin_width) * bin_width
                    if objective_value > objective_value_threshold:
                        hidden_layer_counts[key].append(net_arch_length)

    layers = len(list(hidden_layer_counts.keys()))
    num = 0
    # Plot hidden layers
    if len(hidden_layer_counts) > 0:
        fig, ax = plt.subplots()
        for key, counts in hidden_layer_counts.items():
            if len(counts) > 0:
                bin_offset = num * (bin_width//layers)
                counts_df = pd.DataFrame({'num_layers': counts})
                bins = np.arange(min(counts_df['num_layers']) , max(counts_df['num_layers']) + legend_shift_factor*bin_width, bin_width//layers) + bin_offset
                sns.histplot((counts_df['num_layers']+bin_offset), bins=bins, ax=ax, color=sns.color_palette(PALETTE_STRING)[num], kde=False, label=f"Layer {num}", edgecolor='gray', linewidth=1)
                num+=1
        ax.set_xlabel(r"Neurons in Each Layer(Rounded to Nearest {})".format(bin_width))
        ax.set_ylabel(r'$\mathbf{{f_{{objective}} > {}}}$'.format(objective_value_threshold))
        ax.legend()
        plt.tight_layout()
        the_str = matching_string.replace('-', '_')
        print(f'Saving to {os.path.join(file_path, f"{the_str}_histogram.jpg")}')

        # Save the figure
        fig.savefig(os.path.join(file_path, f"{the_str}_histogram.jpg"), dpi=600)
        plt.close(fig)

def plot_heatmap_on_maze(maze, counts):
    """
    Plots a heatmap of counts on the maze.

    :param maze: A 2D list representing the maze with walls (1), paths (0), robot start ('r'), and goal ('g').
    :param counts: A Counter object where keys are locations (tuples) and values are counts.
    """
    # Convert the maze to a numpy array with numeric values for easier manipulation
    maze_numeric = []
    for row in maze:
        maze_numeric.append([1 if cell == 1 else 2 if cell == 'r' else 3 if cell == 'g' else 0 for cell in row])
    maze_numeric = np.array(maze_numeric, dtype=int)

    # Create a continuous color map for the counts
    heatmap_cmap = plt.cm.viridis
    heatmap_norm = mcolors.Normalize(vmin=0, vmax=max(counts.values()))

    # Create the plot
    fig, ax = plt.subplots()
    ax.set_title('Heatmap of Counts on Maze')

    # Plot the maze with text annotations
    for y in range(maze_numeric.shape[0]):
        for x in range(maze_numeric.shape[1]):
            if maze[y][x] == 1:
                ax.text(x, y, 'W', ha='center', va='center', color='black')
            elif maze[y][x] == 'r':
                ax.text(x, y, 'S', ha='center', va='center', color='red')
            elif maze[y][x] == 'g':
                ax.text(x, y, 'G', ha='center', va='center', color='blue')

    # Create a heatmap array with the same shape as the maze
    heatmap = np.zeros_like(maze_numeric, dtype=float)

    # Fill the heatmap array with the counts from the counter
    for (i, j), count in counts.items():
        heatmap[i, j] = count

    # Overlay the heatmap with transparency
    heatmap_img = ax.imshow(heatmap, cmap=heatmap_cmap, norm=heatmap_norm, interpolation='none', alpha=0.6)

    # Add a color bar to describe the heatmap
    cbar = plt.colorbar(heatmap_img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Counts')

    # Set the ticks, labels, and grid
    ax.set_xticks(np.arange(maze_numeric.shape[1]) + 0.5)
    ax.set_yticks(np.arange(maze_numeric.shape[0]) + 0.5)
    ax.set_xticklabels(np.arange(maze_numeric.shape[1]))
    ax.set_yticklabels(np.arange(maze_numeric.shape[0]))
    ax.invert_yaxis()
    plt.grid(which='both')

    # Display the plot
    plt.show()

def get_N_best_experiments_in_study(study_name: str, N: int, 
                                    excluded_studies:List[str]=[], 
                                    db_url: str = None) -> List[str]:
    exps = get_best_experiments_in_study(study_name, 0.0, db_url=db_url)[::-1]
    experis = []
    for i in range(len(exps)):
        for j in range(len(exps[i])):
            if len(experis)>=N:
                return experis
            if exps[i][j][-1] in excluded_studies:
                continue
            experis.append(exps[i][j][-1])
    return experis
    
def plot_experiments_for_tags(experis:str, tag_comps:List[Tuple[str,str, str]], file_path: Union[str, pathlib.Path], exp_rename:str=''):
    
    for tag, comp, y_label in tag_comps:
        experiments, timestamps, allvalues = get_data_for_experiments_tag_and_component(experis[:], tag, comp)
        if exp_rename:
            for i in range(len(experiments)):
                experiments[i] = f"{exp_rename}_{i}"
        plot_data(experiments, 'Steps', timestamps, y_label, allvalues, xlims=[min(timestamps[0]), max(timestamps[0])], file_path=file_path)

def plot_coverage_for_experiments(experis:str, file_path: Union[str, pathlib.Path], exp_rename:str=''):
    e,s,sr = plot_evaluations_for_experiment(experis[:])
    if exp_rename:
        for i in range(len(e)):
            e[i] = f"{exp_rename}_{i}"
    plot_data(e, 'Steps', s, 'Coverage of Goal Space', sr, legend=True, file_path=file_path)

def plot_N_best_experiments_ep_rew_coverage(study_name: str, N: int, 
                                            file_path: Union[str, pathlib.Path], 
                                            db_url: str = None, exp_rename:str='', 
                                            excluded_studies:List[str]=[]):
    file_path = convert_fp_to_pathlib_create_directories(file_path)
    experis = get_N_best_experiments_in_study(study_name, N, 
                                              excluded_studies=excluded_studies, 
                                              db_url=db_url)
    if not experis:
        return
    plot_exp_rew_coverage(experis, file_path, study_name, exp_rename=exp_rename)

def plot_exp_rew_coverage(experis: List[str], file_path: Union[str, pathlib.Path], file_name:str, exp_rename:str=''):
    file_path = convert_fp_to_pathlib_create_directories(file_path)
    plot_experiments_for_tags(experis, [('train/ep_rew', 'SACSB3Agent', 'Episodic Reward')], file_path/f'{file_name}_ep_rew.png', exp_rename=exp_rename)
    plot_coverage_for_experiments(experis, file_path / f'{file_name}_coverage.png', exp_rename=exp_rename)

def convert_fp_to_pathlib_create_directories(file_path: Union[str, pathlib.Path]):
    #check if filepath is string and convert to path
    is_dir = False
    if isinstance(file_path, str):
        is_dir = file_path[-1] == '/'
        file_path = pathlib.Path(file_path)
    #check if path is a directory and if it exists and create if not using pathlib if not directory create the directory above the file
    if not file_path.is_dir() and not is_dir:
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)
    else:
        if not file_path.exists():
            file_path.mkdir(parents=True)
    return file_path

def set_sns_plot_settings(label_size:int=20, x_tick_size:int=18, y_tick_size:int=18):
    sns.set_theme()
    sns.set_context('talk')
    sns.set_style('darkgrid', {"axes.facecolor": "0.9", "plot.facecolor": "0.9"})

    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['figure.facecolor'] = 'lightgrey'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['xtick.labelsize'] = x_tick_size  # Increase tick label size
    plt.rcParams['ytick.labelsize'] = y_tick_size  # Increase tick label size
    plt.rcParams['axes.labelsize'] = label_size  # Increase axis label size
    plt.rcParams['axes.labelweight'] = 'bold'  # Increase axis label weight


def get_all_metadata_for_experiment(experiment:str) -> Dict[str,str]:
    db_url = get_default_storage()
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    experiment = session.query(Experiment).filter(Experiment.name == experiment).first()
    metadatas = {}
    if experiment:
       metadata = session.query(ExpMetadata).filter_by(experiment_id=experiment.experiment_id).all()
       for data in metadata:
           metadatas[data.key] = data.value
    session.close()
    engine.dispose()
    return metadatas

