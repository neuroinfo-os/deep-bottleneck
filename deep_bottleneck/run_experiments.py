"""This script can be used to submit several experiments to the grid.
All the experiments need to be specified as separate JSON."""
import os
import argparse


def main():
    create_output_directory()
    args = parse_command_line_args()
    start_experiments(args.configpath, bool(args.local_execution))


def create_output_directory():
    output_path = 'output/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configpath',
                        help='The folder containing the experiment configurations or a single configuration file.')
    parser.add_argument('-l', '--local_execution',
                        default=True,
                        help='Whether the experiments should be run locally or on the grid.')
    args = parser.parse_args()
    return args


def start_experiments(config_dir_or_file, local_execution):
    """Recursively walk through the config dir and submit all experiment
    configurations in there to the grid.
    """
    n_experiments = 0
    if os.path.isdir(config_dir_or_file):
        for root, _, files in os.walk(config_dir_or_file):
            for file in files:
                start_experiment(root, file, local_execution)
                n_experiments += 1

    else:
        root, file = os.path.split(config_dir_or_file)
        start_experiment(root, file, local_execution)
        n_experiments += 1

    print(f'Submitted {n_experiments} experiments.')


def start_experiment(root, file, local_execution):
    submission_name, extension = os.path.splitext(file)
    is_valid_config_file = extension == '.json'
    if is_valid_config_file:
        config_path = os.path.join(root, file)
        experiment_name, _ = os.path.splitext(config_path)
        if local_execution:
            command = f'python experiment.py --name {experiment_name} with {config_path} seed=0'
        else:
            command = f'qsub -N {submission_name} experiment.sge {experiment_name} {config_path}'
        print(f'Executing command: {command}')
        os.system(command)


if __name__ == '__main__':
    main()
