import numpy as np

# fname = 'outputs_gitlab/Map3_emu_nell64_nphi512.npz'
fname = 'outputs_gitlab/Map2_emu.npz'
loaded_variable_dict = np.load(fname, allow_pickle=True)['arr_0'].tolist()

n_parameters = loaded_variable_dict['n_parameters']
parameters = loaded_variable_dict['parameters']
feature_dimensions = loaded_variable_dict['feature_dimensions']

scaling_division = loaded_variable_dict['scaling_division']
scaling_subtraction = loaded_variable_dict['scaling_subtraction']

weights_ = loaded_variable_dict['weights']
alphas_ = loaded_variable_dict['alphas']
betas_ = loaded_variable_dict['betas']
biases_ = loaded_variable_dict['biases']

# Extract training normalization statistics
try: param_train_mean = loaded_variable_dict['parameters_mean']
except: param_train_mean = loaded_variable_dict['param_train_mean']
try: param_train_std = loaded_variable_dict['parameters_std']
except: param_train_std = loaded_variable_dict['param_train_std']
try: feature_train_mean = loaded_variable_dict['features_mean']
except: feature_train_mean = loaded_variable_dict['feature_train_mean']
try: feature_train_std = loaded_variable_dict['features_std']
except: feature_train_std = loaded_variable_dict['feature_train_std']

hyper_params = list(zip(alphas_, betas_))
weights_ = [w.T for w in weights_]
weights = list(zip(weights_, biases_))
weights = np.array(weights, dtype=object)

np.savez(fname.replace('.npz', '_converted.npz'),
        n_parameters = loaded_variable_dict['n_parameters'],
        parameters = loaded_variable_dict['parameters'],
        feature_dimensions = loaded_variable_dict['feature_dimensions'],
        scaling_division = loaded_variable_dict['scaling_division'],
        scaling_subtraction = loaded_variable_dict['scaling_subtraction'],
        weights = weights,
        hyper_params = hyper_params,
        param_train_mean = param_train_mean,
        param_train_std = param_train_std,
        feature_train_mean = feature_train_mean,
        feature_train_std = feature_train_std)