import json
import os
import argparse

# create output folder
output_path = "output/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

#argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', help="The folder containing the JSON-files")
args = parser.parse_args()


# start Nets
for root, dirs, files in os.walk(args.directory):
    for name in files:
        submition_name, extension = os.path.splitext(name)
        if extension == ".json" or extension == ".yml":
            json_path = os.path.join(root, name)
            command = f"qsub -N {submition_name} experiment.sge {submition_name} {json_path}"
            print(command)
            os.system(command)