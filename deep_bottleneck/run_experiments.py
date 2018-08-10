"""This script can be used to submit several experiments to the grid.

All the experiments need to be specified as seperate JSON or yml files."""
import os
import argparse


def main():
    create_output_directory()
    args = parse_command_line_args()
    start_experiments(args.directory)


def create_output_directory():
    output_path = 'output/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', help='The folder containing the experiment configurations')
    args = parser.parse_args()
    return args


def start_experiments(config_dir):
    """Recursively walk through the config dir and submit all experiment
    configurations in there to the grid.
    """
    n_experiments = 0
    for root, dirs, files in os.walk(config_dir):
        for name in files:
            submission_name, extension = os.path.splitext(name)
            is_valid_config_file = extension == '.json' or extension == '.yml'
            if is_valid_config_file:
                config_path = os.path.join(root, name)
                command = f'qsub -N {submission_name} experiment.sge {submission_name} {config_path}'
                print(f'Executing command: {command}')
                os.system(command)
                n_experiments += 1

    print(f'Submitted {n_experiments} experiments.')


if __name__ == '__main__':
    main()
