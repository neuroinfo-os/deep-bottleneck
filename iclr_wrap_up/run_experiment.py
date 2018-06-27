import json
import os

# create output folder
output_path = "output/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# get hyperparameter
h_name = "hyperparameter.json"
h_file = open(h_name)
h_data = json.load(h_file)
h_len = len(h_data["hyperparameter"])

# start Nets
for i in range(h_len):
	#print(h_data["hyperparameter"][i])
	name = h_data["hyperparameter"][i]["name"]
	command = "qsub -N " + name + " experiment_grid.sge " + str(i)
	print(command)
	os.system(command)