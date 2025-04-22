import pathlib
import subprocess
import time
import os
import logging
from typing import List
import optuna
import yaml

from ..optimiser.sftp import PexpectSFTPClient, FilePathGenerator
from ..optimiser.scriptgen import ScriptGenerator, ScriptRunner
from ..optimiser.experimentrunner import ExperimentRunner
from ..logger import get_coverage, create_experiment_direct

class SBatchGenerator(ScriptGenerator):
    _clist = None
    _sftp = None

    def __init__(self, job_name: str, config_path: str = None, config_kwargs: dict = None):
        self.config_path = config_path
        self.job_name = job_name
        self._config_kwargs = config_kwargs

    def create_config_string(self, exp_name:str) -> str:
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        config_str = ''
        defaults = config['defaults']
        if self._config_kwargs is not None:
            defaults = {**defaults, **self._config_kwargs}
        defaults['job_name'] = exp_name
        config_str += config['sbatch'].format(**defaults).rstrip() + " "
        return config_str

    def generate_script(self, hp_cmd_str: str, exp_name:str) -> str:
        file_str = self.create_config_string(exp_name)
        return file_str + hp_cmd_str + "\n"

class SBatchRunner(ScriptRunner):
    def __init__(self, filepath: pathlib.Path, generator: SBatchGenerator):
        self.filepath = filepath
        self._generator = generator

    def save_sbatch_file(self, file_name:str, file_str: str):
        if not self.filepath.exists():
            self.filepath.mkdir(parents=True, exist_ok=True)
        file_path = self.filepath / file_name
        with open(file_path, 'w', newline='\n') as f:
            f.write(file_str)
        return file_path

    def run_sbatch_script(self, file_path) -> int:
        process = subprocess.Popen(args=['sbatch',
                                            str(file_path)],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        stdout = str(stdout, encoding='utf-8')
        print("STDOUT: " ,stdout)
        print("STD ERROR: " , str(stderr, encoding='utf-8'))
        return int(stdout.split(' ')[-1].strip())

    def wait_script_complete(self, sbatch_num: int) -> bool:
        process = subprocess.Popen(args=['squeue', '-j',
                                            str(sbatch_num)],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
        stdout, _ = process.communicate()
        stdout = [
            x for x in str(stdout, encoding='utf-8').split('\n')
            if x != ''
        ]
        return len(stdout) > 1
        
    def get_script_result(self, exp_name, storage_string:str) -> float:
        return get_coverage(exp_name, storage_string)

    def run(self, hp_cmd_str: str, trial:optuna.Trial, storage_string:str) -> float:
        lst = hp_cmd_str.split(' ')
        exp_name = lst[lst.index('--name') + 1]
        sbatch_str = self._generator.generate_script(hp_cmd_str, exp_name)
        
        file_path = self.save_sbatch_file(exp_name, sbatch_str)
        #don't forget to generate script and add input argument
        sbatch_num = self.run_sbatch_script(file_path)
        last_step = 0
        while self.wait_script_complete(sbatch_num):
            # I should add a trial.report here. The function should take 
            try:
                result, steps = self.get_script_result(exp_name)
                if steps != last_step:
                    trial.report(result, steps)
                    last_step = steps
            except:
                pass
            time.sleep(60)
        val = self.get_script_result(exp_name, storage_string)
        if val is None:
            return val
        return val[0]

class RemoteSBatchSFTPRunner(SBatchRunner):
    def __init__(self, filepath: pathlib.Path, sftp:PexpectSFTPClient, generator: SBatchGenerator, file_path_generator: FilePathGenerator):
        super().__init__(filepath, generator)
        self._sftp = sftp
        self._file_path_generator = file_path_generator

    def save_sbatch_file(self, file_name:str, file_str: str):
        self.local_sh, self.remote_path = self._file_path_generator(file_name)
        super().save_sbatch_file(self.local_sh, file_str)
        if not self._sftp.exists(self.remote_path.parent):
            self._sftp.makedirs(str(self.remote_path.parent))
        self._sftp.put(str(self.local_sh), str(self.remote_path))
        return self.remote_path

    def run_sbatch_script(self, file_path:str) -> int:
        sbatch_str = self._sftp.execute(
                f'sbatch {file_path}')
        retry = 0
        while retry < 5:
            if sbatch_str is None:
                logging.error(f"Failed to run sbatch script {file_path}")
                retry += 1
                time.sleep(60)
            try:
                ret_code = int(str(sbatch_str).split(' ')[-1].strip())
                logging.info(f"Running sbatch script {file_path} with sbatch number {ret_code}")
                return ret_code
            except ValueError as e:
                logging.error(f"Failed to extract sbatch number from {sbatch_str}")
                if "Unable to contact slurm controller" in str(sbatch_str):
                    retry += 1
                    time.sleep(60)
                else:
                    raise e
        if retry == 5:
            raise Exception(f"Failed to run sbatch script {file_path}")
        
    
    def wait_script_complete(self, sbatch_num: int) -> bool:
        cmd = self._sftp.execute(f'squeue -j {sbatch_num}')
        if cmd is None:
            return True
        return len(cmd.split('\n')) > 1
    
class LocalRunner(ExperimentRunner):
    def __init__(self, file_path: pathlib.Path = None, storage_url: str = None):
        self._file_path = file_path
        self._storage_url = storage_url
        self._cmd = f'python "{self._file_path}" '

    def generate_cmd(self, hp_cmd_str: str) -> str:
        return self._cmd + hp_cmd_str


    def spawn_process(self, cmd, stdout_file, stderr_file) -> int:
        print('Executing command: ', cmd)
        process = subprocess.Popen(cmd,
                                    stdout=stdout_file,
                                    stderr=stderr_file,
                                    shell=True,
                                    text=True, 
                                    env=os.environ.copy())
        return process
        
    def get_script_result(self, exp_name) -> float:
        return get_coverage(exp_name, self._storage_url)

    def run(self, hp_cmd_str: str, trial_num:int, _:str) -> float:
        cmd = self.generate_cmd(hp_cmd_str)
        lst = hp_cmd_str.split(' ')
        exp_name = lst[lst.index('--name') + 1]
        fp = f"./localruns/{exp_name}/"
        if not os.path.exists(fp):
            os.makedirs(fp)
        stdout_file = open(f"{fp}/{exp_name}.out", "w")
        stderr_file = open(f"{fp}/{exp_name}.err", "w")
        #write cmd to a cmd file
        with open(f"{fp}/{exp_name}.cmd", "w") as f:
            f.write(cmd)
        #don't forget to generate script and add input argument
        process = self.spawn_process(cmd, stdout_file, stderr_file)
        std_out_reader = open(f"{fp}/{exp_name}.out", "r")
        most_recent_evaluation = 0
        while 1:
            ret_code = process.poll()
            # read stdout_file to see if new evaluations to report
            lines = reversed(std_out_reader.readlines())
            for i in lines:
                if "EVALUATING@" in i:
                    step = int(i.split('@')[1])
                    if step > most_recent_evaluation:
                        most_recent_evaluation = step
                        proxy = None
                        retries = 0
                        while proxy is None:
                            proxy = self.get_script_result(exp_name)
                            if proxy is not None:
                                result, step = proxy
                                trial_num.report(result, step)
                            else:
                                time.sleep(60)
                                retries += 1
                                if retries > 4:
                                    break
            if ret_code is not None:
                break
            time.sleep(20)
        if ret_code != 0:
            if process.stderr is not None:
                stderr_output = process.stderr.read()
            else:
                stderr_output = "Dunno, read the corresponding .err file"
            std_out_reader.close()
            stdout_file.close()
            stderr_file.close()
            raise Exception(f"Experiment {exp_name} failed with error code {ret_code} with error {stderr_output}")
        std_out_reader.close()
        stdout_file.close()
        stderr_file.close()
        results = self.get_script_result(exp_name)
        return results if results is None else results[0]

