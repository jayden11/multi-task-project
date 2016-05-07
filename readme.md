### Multi-Task Project
## Where everything is

- run_model.py is where you call the model. it defines the parameters, calls the reader etc. At the moment it calls four networks - the hyperparameter training network, the hyperparameter validation network, the final training network and the final testing network.
- run_epoch.py is calle by individual models, returns the predictions and accuracy to run_model
- graph.py is the computation graph

## how to run

you gotta flag:

python run_model.py --model_type "JOINT" --dataset_path "../../data/conll"

note - only works with python 2 at the moment. Working on it.
