import json
import re
import os

COHORT_NR = 5

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


for activation_fn in ["relu", "tanh", "sigmoid", "softsign", "softplus", "leaky_relu", "hard_sigmoid", "selu", "relu6", "elu", "linear"]:
   di["activation_fn"] = activation_fn
   js_string = json.dumps(di)
   js_string = re.sub(r",(?![^\[]*\])", ",\n", js_string)

   name_str = "cohort_" + str(COHORT_NR) + "_activationfn_" + di["activation_fn"] + ".json"


   with open(os.path.join(folder_str, name_str), "w") as outfile:
       outfile.write(js_string)

   print(name_str)