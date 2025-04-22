import pexpect
import logging
import sys
import re
import time
import pathlib
import pysftp
from stat import S_ISDIR, S_ISREG
from typing import List, Union
from threading import Event, Lock

ERRORLIMIT = 60

def make_dir(dir_path: pathlib.Path):
    """Make dir if it doens't exist and don't error if it does

    :param dir_path: path to create
    :type dir_path: pathlib.Path
    """    
    try:
        dir_path.mkdir()
    except FileExistsError:
        pass


def check_file_times(from_file:str, to_file:str) -> bool:
    """Check if from_file is newer than to_file

    :param from_file: the source file
    :type from_file: str
    :param to_file: the destination file
    :type to_file: str
    :return: True if from_file is newer than to_file, False otherwise
    :rtype: bool
    """    
    from_time = from_file.lstat().st_mtime if isinstance(from_file,pathlib.Path)\
                else from_file.st_mtime
    to_time = to_file.lstat().st_mtime if isinstance(to_file,pathlib.Path)\
                else to_file.st_mtime
    return int(from_time) > int(to_time)

def check_create(file_path: pathlib.Path) -> bool:
    """Check if file_path exists and create it if it doesn't

    :param file_path: the path to check
    :type file_path: pathlib.Path
    :return: True if file_path exists, False otherwise
    :rtype: bool
    """    
    if not file_path.exists():
        paths_to_create = [file_path]
        while not paths_to_create[0].parent.exists():
            paths_to_create.insert(0, paths_to_create[0].parent)
        [f.mkdir() for f in paths_to_create if not f.suffix]
        return False
    return True

class SFTP(pysftp.Connection):
          
    def __init__(self, host=None, username=None, private_key=None,
                 password=None, port=22, private_key_pass=None, ciphers=None,
                 log=False, cnopts=None, default_path=None):
        if cnopts is None:
            cnopts, hostkeys = self.hostkey_resolution(host)
        super().__init__(host, username, private_key, password, port,
                         private_key_pass, ciphers, log, cnopts, default_path)
        if hostkeys != None:
            print("Connected to new host, caching its hostkey")
            hostkeys.add(host, self.remote_server_key.get_name(),
                         self.remote_server_key)
            hostkeys.save(pysftp.helpers.known_hosts())

    def hostkey_resolution(self, host):
        cnopts = pysftp.CnOpts()
        hostkeys = None
        if cnopts.hostkeys.lookup(host) == None:
            print("New host - will accept any host key")
            # Backup loaded .ssh/known_hosts file
            hostkeys = cnopts.hostkeys
            # And do not verify host key of the new host
            cnopts.hostkeys = None
        return cnopts, hostkeys
    
    def put_r(self, local, remote='.', preserve_mtime=True, confirm=True,
              include=set()):
        local = local if isinstance(local,
                                    pathlib.Path) else pathlib.Path(local)
        if not self.exists(remote):
            self.makedirs(remote)
        if local.is_dir():
            for f in local.iterdir():
                if (set(f.parts) & include):
                    remote_path = remote + '/' + f.name
                    if f.is_dir():
                        if not self.exists(remote_path):
                            self.makedirs(remote_path)
                        self.put_r(f, remote_path, preserve_mtime, confirm,
                                   include)
                    else:
                        if not (self.exists(remote_path)) or check_file_times(
                                f, self.stat(remote_path)):
                            print(f'Copying {f} to {remote_path}')
                            self.put(f, remote_path,
                                     preserve_mtime=preserve_mtime,
                                     confirm=confirm)
        else:
            if not (set(local.parts) & include):
                return
            remote_path = remote + '/' + local.name
            if not (self.exists(remote_path)) or check_file_times(
                    local, self.stat(remote_path)):
                print(f'Copying {local} to {remote_path}')
                self.put(local, remote_path, preserve_mtime=preserve_mtime,
                         confirm=confirm)

    def get_r(self, remote, local, preserve_mtime=True, include=set()):
        local = local if isinstance(local,
                                    pathlib.Path) else pathlib.Path(local)
        make_dir(local)
        for entry in self.listdir_attr(remote):
            #remote must use posix file system
            remote_path = remote + '/' + entry.filename
            if not (set(remote_path.split('/')) & include):
                continue
            #local can be either windows or posix
            local_path = local / entry.filename
            mode = entry.st_mode
            if S_ISDIR(mode):
                make_dir(local_path)
                self.get_r(remote_path, local_path, preserve_mtime, include)
            elif S_ISREG(mode):
                if not (local_path.exists()) or check_file_times(
                        entry, local_path):
                    print(f'Copying {remote_path} to {local_path}')
                    self.get(remote_path, local_path,
                             preserve_mtime=preserve_mtime)
                    
class FilePathGenerator:
    def __init__(self, local_path: pathlib.Path, remote_path: pathlib.Path):
        if not isinstance(local_path, pathlib.Path):
            local_path = pathlib.Path(local_path)
        self._local_path = local_path
        if not isinstance(remote_path, pathlib.Path):
            remote_path = pathlib.Path(remote_path)
        self._remote_path = remote_path

    def generate_local_remote_path(self, file_name:str):
        stem = pathlib.Path(*self._local_path.parts[-2:])
        local_file = self._local_path / file_name
        check_create(local_file.parent)
        remote_path = self._remote_path / stem / file_name
        return local_file, remote_path

    def __call__(self, file_num: int) -> List[pathlib.Path]:
        return self.generate_local_remote_path(file_num)

class PexpectTimeoutError(Exception):
    pass

class PexpectConnectionError(Exception):
    pass

class PexpectSFTPClient:
    """An SFTP client that uses pexpect to execute commands on the remote host.

    This was created so that ssh configurations can be used to connect to remote hosts. 
    The issue with using paramiko is that it is poorly suited to complex ssh configurations.
    It also allows multiple commands in a single session which can be very useful for setting up
    environment variables before executing the sbatch command.
    """
    _stop = Event()
    _counter = 0
    _counter_lock = Lock()
    def __init__(self, config_shortname:str, ssh_prompt:str):
        """Initialise the PexpectSFTPClient.

        :param config_shortname: The host name of the ssh configuration
        :type config_shortname: str
        :param ssh_prompt: The default ssh prompt on your remote host. Caution: it is different on different hosts.
        :type ssh_prompt: str
        """        
        self.config_shortname = config_shortname
        self.ssh_cmd = f"ssh {config_shortname}"
        self.sftp_cmd = f"sftp {config_shortname}"
        self.sftp_prompt = r'sftp> '
        self.ssh_prompt = ssh_prompt
        
        # # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.disabled = True

    def check_lock(self, e = None, child = None):
        with PexpectSFTPClient._counter_lock:
            if e is None:
                #Reset counter
                PexpectSFTPClient._counter = 0
            else:
                ssh_output = child.before
                #Multiplex not happening
                error_message = ''
                if 'passcode' in ssh_output: # or 'restricted to employees, students, or individuals authorized' in ssh_output: issues with rangpur connectivity mean this is not so good.
                    PexpectSFTPClient._counter = ERRORLIMIT
                    error_message = 'MFA - Passcode required'
                #Wrong host
                elif 'Could not resolve hostname' in ssh_output:
                    PexpectSFTPClient._counter = ERRORLIMIT
                    error_message = 'Could not resolve hostname'
                else:
                    PexpectSFTPClient._counter += 1
                if PexpectSFTPClient._counter >= ERRORLIMIT:
                    error_message = f'Error limit reached' if not error_message else error_message
                    PexpectSFTPClient._stop.set()
                    raise PexpectConnectionError(f'{error_message} : {e}')
                
                
    def _execute_ssh_command(self, command: str)->str:
        """Execute a command on the ssh machine

        :param command: The command you wish to execute e.g ls -l
        :type command: str
        :raises PexpectTimeoutError: The command has timed out 
        :raises PexpectConnectionError: There was an error connecting to the remote host
        :return: The stdout of the command
        :rtype: str
        """
        ssh_agent_pid = None
        if self._stop.is_set():
            raise PexpectConnectionError('SSH connection failed in another thread') 
        try:
            result = None
            child = pexpect.spawn(self.ssh_cmd, encoding='utf-8', timeout=60)
            # Uncomment the below line if you need to debug your connection
            # child.logfile = sys.stdout  # Logs the output to stdout for debugging
        
            # This may or may not work on your remote machine.
            # Get the PID of the ssh-agent to kill it before exiting.
            # try:
            #     child.expect("Agent pid (\d+)", timeout=20)
            #     output = child.after  # This contains the matched output
            #     # Extract the PID using regular expression
            #     match = re.search(r'Agent pid (\d+)', output)
            #     if match:
            #         ssh_agent_pid = int(match.group(1))
            #     else:
            #         print('No agent PID found')
            # except pexpect.TIMEOUT:
            #     print('No agent PID found')
            #     self.check_lock()            
            # Execute the command
            try:
                child.expect(self.ssh_prompt)
                # output = child.before.strip()
                #This is specific to me and I should make this a function in an extended class rather than here
                if command.split(' ')[0] == 'sbatch':
                    child.sendline('source ~/.bash_profile')
                    child.expect(self.ssh_prompt)
                # Run command specified
                child.sendline(command)
                child.expect(self.ssh_prompt)
                # Get the output excluding the first and last line
                result = child.before.strip().split('\n')
                result = '\n'.join(result[1:-1])
                child.sendline(f'pkill -u uqlsalt -f ssh-agent')
                child.expect(self.ssh_prompt)
                time.sleep(1)
                child.sendline('exit\r')
                child.expect(pexpect.EOF)
                child.close()
                self.check_lock()
                return result
            # Handle timeout. Usually failed ssh connection
            except (pexpect.EOF, pexpect.TIMEOUT) as e:
                self.logger.error(f"Error occurred: {e}")
                self.check_lock(e, child)
                child.close()
                return None
        # I'm not sure if this actually catches anything as I haven't hit it but it's there just in case
        except (pexpect.EOF, pexpect.TIMEOUT) as e:
            self.logger.error(f"Error occurred: {e}")
            self.check_lock(e, child)
        except pexpect.exceptions.ExceptionPexpect as e:
            self.logger.error(f"Error occurred: {e}")
            return None
        
        
    def _execute_sftp_command(self, command: str) -> str:
        """Executes SFTP for file exchange

        :param command: The sftp command to be executed
        :type command: str
        :return: The output of the sftp command
        :rtype: str
        """
        if self._stop.is_set():
            raise PexpectConnectionError('SSH connection failed in another thread')       
        child = pexpect.spawn(self.sftp_cmd, encoding='utf-8', timeout=60)
        # child.logfile = sys.stdout  # Logs the output to stdout for debugging
        
        try:
            child.expect(self.sftp_prompt)
            child.sendline(command)
            child.expect(self.sftp_prompt)
            result = child.before.strip()
            child.sendline('bye')
            child.expect(pexpect.EOF)
            self.check_lock()
            child.close()
            return result
        except (pexpect.EOF, pexpect.TIMEOUT) as e:
            self.logger.error(f"Error occurred: {e}")
            self.check_lock(e, child)
            child.close()
            return None
            
        
    def exists(self, path: Union[str, pathlib.Path]) -> bool:
        """Check if a file or path exists

        :param path: The dir or file you wish to check
        :type path: Union[str, pathlib.Path]
        :return: true if path exists, false otherwise
        :rtype: bool
        """                
        result = self._execute_ssh_command(f'test -e {path} && echo EXISTS || echo NO')
        #Recheck if None (feels dangerous.)
        if result is None:
            return False
        return "EXISTS" in result

    def makedirs(self, remote_directory: str):
        """Make all directories in remote_directory on remote host

        :param remote_directory: The path you wish to create
        :type remote_directory: str
        """        
        directories = remote_directory.strip().split('/')
        for i in range(1, len(directories) + 1):
            path = '/'.join(directories[:i])
            if not self.exists(path):
                self.logger.info(f"Creating directory: {path}")
                self._execute_sftp_command(f'mkdir {path}')

    def put(self, local_path: Union[str, pathlib.Path], remote_path: Union[str, pathlib.Path]):
        """Put file from local host onto remote host

        :param local_path: path to file on local
        :type local_path: Union[str, pathlib.Path]
        :param remote_path: path to file on remote
        :type remote_path: Union[str, pathlib.Path]
        """        
        self.logger.info(f"Uploading {local_path} to {remote_path}")
        self._execute_sftp_command(f'put {local_path} {remote_path}')

    def execute(self, command: str) -> str:
        """Execute a command on the remote host

        :param command: The command to execute on the remote host
        :type command: str
        :return: The output of the command on the remote host
        :rtype: str
        """        
        self.logger.info(f"Executing command: {command}")
        return self._execute_ssh_command(command)