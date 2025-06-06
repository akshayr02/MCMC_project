## Authtor: Pierre Burger
## Date: 2023-10-02
## Description:
# This module provides a JAX-based emulator for predicting arbirtray features from input parameters using a neural network.
# It is an adapted version of CosmoPower Jax emulator (https://github.com/dpiras/cosmopower-jax), which is a JAX-based emulator for predicting power spectra from input parameters using a neural network.

import pickle
import numpy as onp
import jax.numpy as jnp
from jax import random
from jax.nn import sigmoid
from jax import jacfwd, jacrev, jit
from functools import partial



class cosmoemu_jax:
    """General-purpose JAX-based emulator for predicting features from input parameters using a neural network.

    Supports loading pretrained networks stored in a specific npz format. Designed for rapid evaluation
    and differentiation of outputs with respect to inputs.

    Parameters
    ----------
    filepath : string
        Full path to the .pkl file containing the pretrained model.
    verbose : bool, default=True
        Whether to print information during initialization.
    """
    def __init__(self, filepath=None, verbose=True): 
        if verbose:
            print(filepath)

        # Load the npz model file
        loaded_variable_dict = onp.load(filepath, allow_pickle=True)

        if verbose:
            print(loaded_variable_dict.keys())

        # Extract relevant values from the loaded dictionary
        n_parameters = loaded_variable_dict['n_parameters']
        parameters = loaded_variable_dict['parameters']
        feature_dimensions = loaded_variable_dict['feature_dimensions']

        scaling_division = loaded_variable_dict['scaling_division']
        scaling_subtraction = loaded_variable_dict['scaling_subtraction']

        # Extract training normalization statistics
        param_train_mean = loaded_variable_dict['param_train_mean']
        param_train_std = loaded_variable_dict['param_train_std']
        feature_train_mean = loaded_variable_dict['feature_train_mean']
        feature_train_std = loaded_variable_dict['feature_train_std']
      
        # Group activation hyperparameters and transpose weights to match JAX's convention
        hyper_params = loaded_variable_dict['hyper_params']
        weights = loaded_variable_dict['weights'].tolist()

        # Store model components as attributes
        self.weights = weights
        self.hyper_params = hyper_params
        self.param_train_mean = param_train_mean
        self.param_train_std = param_train_std
        self.feature_train_mean = feature_train_mean
        self.feature_train_std = feature_train_std
        self.n_parameters = n_parameters
        self.parameters = parameters
        self.scaling_division = scaling_division
        self.scaling_subtraction = scaling_subtraction
        self.modes = jnp.arange(0, feature_dimensions)  # useful indexing range

    def _dict_to_ordered_arr_jax(self, input_dict):
        """Convert dictionary of input parameters to ordered array based on trained model."""
        if self.parameters is not None:
            return jnp.stack([input_dict[k] for k in self.parameters], axis=1)
        else:
            return jnp.stack([input_dict[k] for k in input_dict], axis=1)

    @partial(jit, static_argnums=0)
    def _activation(self, x, a, b):
        """Custom activation function as used in training.
        Parameters `a` and `b` control the nonlinearity.
        """
        return jnp.multiply(jnp.add(b, jnp.multiply(sigmoid(jnp.multiply(a, x)), jnp.subtract(1., b))), x)

    @partial(jit, static_argnums=0)
    def _predict(self, weights, hyper_params, param_train_mean, param_train_std,
                 feature_train_mean, feature_train_std, input_vec):
        """Forward pass through the neural network to produce feature predictions."""
        # Normalize input vector
        layer_out = [(input_vec - param_train_mean) / param_train_std]

        # Apply each hidden layer: linear -> activation
        for i in range(len(weights[:-1])):
            w, b = weights[i]
            alpha, beta = hyper_params[i]
            act = jnp.dot(layer_out[-1], w.T) + b  # Linear transformation
            layer_out.append(self._activation(act, alpha, beta))  # Apply activation

        # Final linear layer without activation
        w, b = weights[-1]
        preds = jnp.dot(layer_out[-1], w.T) + b

        # De-normalize predictions
        preds = preds * feature_train_std + feature_train_mean
        return preds.squeeze()

    @partial(jit, static_argnums=0)
    def predict(self, input_vec):
        """Predict features from input parameters using the emulator."""
        if isinstance(input_vec, dict):
            input_vec = self._dict_to_ordered_arr_jax(input_vec)

        # Ensure proper shape
        if len(input_vec.shape) == 1:
            input_vec = input_vec.reshape(-1, self.n_parameters)
        assert len(input_vec.shape) == 2

        return self._predict(self.weights, self.hyper_params, self.param_train_mean, 
                             self.param_train_std, self.feature_train_mean, self.feature_train_std,
                             input_vec)

    @partial(jit, static_argnums=0)
    def rescaled_predict(self, input_vec):
        """Return emulator prediction scaled to match physical values."""
        return self.predict(input_vec) * self.scaling_division + self.scaling_subtraction

    @partial(jit, static_argnums=0)
    def ten_to_rescaled_predict(self, input_vec):
        """Return 10^rescaled prediction, useful for log-scaled outputs."""
        return 10 ** self.rescaled_predict(input_vec)

    @partial(jit, static_argnums=0)
    def derivative(self, input_vec, mode='forward'):
        """Compute derivatives of emulator outputs w.r.t. input parameters."""
        if isinstance(input_vec, dict):
            input_vec = self._dict_to_ordered_arr_jax(input_vec)
        if len(input_vec.shape) == 1:
            input_vec = input_vec.reshape(1, self.n_parameters)
        assert len(input_vec.shape) == 2

        # Choose forward or reverse-mode autodiff
        if mode == 'forward':
            if input_vec.shape[0] == 1:
                return jnp.swapaxes(jacfwd(self.predict)(input_vec), 1, 2)
            return jnp.diagonal(jnp.swapaxes(jacfwd(self.predict)(input_vec), 1, 2))
        elif mode == 'reverse':
            if input_vec.shape[0] == 1:
                return jnp.swapaxes(jacrev(self.predict)(input_vec), 1, 2)
            return jnp.diagonal(jnp.swapaxes(jacrev(self.predict)(input_vec), 1, 2))
        else:
            raise ValueError(f"Differentiation mode '{mode}' not recognized.")

    @partial(jit, static_argnums=0)
    def derivative_rescaled(self, input_vec, mode='forward'):
        """Derivative of rescaled outputs with respect to input parameters."""
        if isinstance(input_vec, dict):
            input_vec = self._dict_to_ordered_arr_jax(input_vec)
        if len(input_vec.shape) == 1:
            input_vec = input_vec.reshape(1, self.n_parameters)
        assert len(input_vec.shape) == 2

        if mode == 'forward':
            if input_vec.shape[0] == 1:
                return jnp.swapaxes(jacfwd(self.rescaled_predict)(input_vec), 1, 2)
            return jnp.diagonal(jnp.swapaxes(jacfwd(self.rescaled_predict)(input_vec), 1, 2))
        elif mode == 'reverse':
            if input_vec.shape[0] == 1:
                return jnp.swapaxes(jacrev(self.rescaled_predict)(input_vec), 1, 2)
            return jnp.diagonal(jnp.swapaxes(jacrev(self.rescaled_predict)(input_vec), 1, 2))
        else:
            raise ValueError(f"Differentiation mode '{mode}' not recognized.")

    @partial(jit, static_argnums=0)
    def derivative_ten_to_rescaled(self, input_vec, mode='forward'):
        """Derivative of 10^rescaled outputs with respect to input parameters."""
        if isinstance(input_vec, dict):
            input_vec = self._dict_to_ordered_arr_jax(input_vec)
        if len(input_vec.shape) == 1:
            input_vec = input_vec.reshape(1, self.n_parameters)
        assert len(input_vec.shape) == 2

        if mode == 'forward':
            if input_vec.shape[0] == 1:
                return jnp.swapaxes(jacfwd(self.ten_to_rescaled_predict)(input_vec), 1, 2)
            return jnp.diagonal(jnp.swapaxes(jacfwd(self.ten_to_rescaled_predict)(input_vec), 1, 2))
        elif mode == 'reverse':
            if input_vec.shape[0] == 1:
                return jnp.swapaxes(jacrev(self.ten_to_rescaled_predict)(input_vec), 1, 2)
            return jnp.diagonal(jnp.swapaxes(jacrev(self.ten_to_rescaled_predict)(input_vec), 1, 2))
        else:
            raise ValueError(f"Differentiation mode '{mode}' not recognized.")
