import argparse
import threading
import time

from ..experimentrunners import LocalRunner

def create_dict_from_exp_file(exp_file):
    params = exp_file.split(' ')[1:]  # Remove the python file bit from the front
    params_dict = {}
    i = 0
    while i < len(params):
        key = params[i]
        if key.startswith('--'):
            values = []
            i += 1
            while i < len(params) and not params[i].startswith('--'):
                values.append(params[i])
                i += 1
            if len(values) == 1:
                params_dict[key] = values[0]
            else:
                params_dict[key] = values
        else:
            i += 1
    return params_dict

def run_local(num, args, unknowns):
    if args['exp_filepath'] == '':
        raise ValueError('Experiment filepath is required')
    #read the experiment file
    with open(args['exp_filepath'], 'r') as f:
        exp_file = f.read().strip()
    params = create_dict_from_exp_file(exp_file)
    for i in range(0, len(unknowns)-1, 2):
        if unknowns[i] in params:
            params[unknowns[i]] = unknowns[i+1]
    params['--name'] = params['--name'] + f'_{num}'
    local_runner = LocalRunner()
    hp_cmd_string = ' '.join([f'{key} {" ".join(params[key]) if isinstance(params[key], list) else params[key]}' for key in params])
    local_runner.run(hp_cmd_string, num)

def main(args, unknowns):
    threads = []
    for i in range(args['num_runs']):
        thread = threading.Thread(target=run_local, args=(i, args.copy(), unknowns.copy()))
        thread.start()
        threads.append(thread)
        time.sleep(20)
    for thread in threads:
        thread.join()        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-filepath', default='')
    parser.add_argument('--num-runs', type=int, default=5)
    args, unknowns = parser.parse_known_args()
    args = vars(args)
    main(args, unknowns)