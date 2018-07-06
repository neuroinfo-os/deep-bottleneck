import json
import re
import os

COHORT_NR = 6

di = {
"epochs": 8000,
"batch_size" : 256,
"architecture" : [10, 7, 5, 4, 3],
"learning_rate" : 0.0004,
"calculate_mi_for": "full_dataset",
"activation_fn" :"relu",
"model" : "models.feedforward",
"dataset" : "datasets.harmonics",
"estimator" : "mi_estimator.upper",
"discretization_range" : 0.001,
"callbacks" : [],
"optimizer" : "adam",
"n_runs" : 5,
"regularization" : False
}
folder_str = "cohort_"+str(COHORT_NR)
os.makedirs(folder_str, exist_ok=True)

activation_functions = ["relu", "tanh"]
architectures = [[10, 7, 5, 4, 3],
                 [10, 9, 7, 7, 3],
                 [10, 9, 7, 5, 3],
                 [10, 9, 7, 3, 3],
                 [10, 7, 7, 4, 3],
                 [10, 7, 5, 4],
                 [10, 7, 5],
                 [10],
                 [1, 1, 1, 1],
                 []
                 ]


for activation_fn in activation_functions:
    current_folder_str = folder_str + "/" + str(activation_fn)
    os.makedirs(current_folder_str, exist_ok=True)
    for architecture in architectures:
        di["activation_fn"] = activation_fn
        di["architecture"] = architecture
        js_string = json.dumps(di)
        js_string = re.sub(r",(?![^\[]*\])", ",\n", js_string)

        name_str = "cohort_" + str(COHORT_NR) + "_" + di["activation_fn"] + "_architecture_" + "_".join(str(a) for a in di["architecture"])  + ".json"


        with open(os.path.join(current_folder_str, name_str), "w") as outfile:
            outfile.write(js_string)

        print(name_str)


