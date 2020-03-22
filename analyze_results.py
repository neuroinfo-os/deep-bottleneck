import papermill as pm

pm.execute_notebook(
   'result.ipynb',
   'output.ipynb',
   parameters=dict(ids=list(range(12, 21)))
)