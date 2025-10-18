# %%
import jax.numpy as jnp
from jax import random

from nn import init_mlp_params, forward_drop as forward, update_drop as update
from vis import plot_predictions

# %%
key = random.PRNGKey(0)
x = jnp.linspace(-3, 3, 200).reshape(-1, 1)
key, subkey = random.split(key)
y = jnp.tanh(x) + 0.1 * random.normal(key, x.shape)

mlp_dims = [1, 16, 16, 1]
key, subkey = random.split(key)
mlp_params = init_mlp_params(subkey, mlp_dims)
n_epochs = 2000
dropout = 0.05
lr = 2e-3
for epoch in range(n_epochs):
    mlp_params = update(mlp_params, x, y, subkey, dropout, lr=lr)

x_test = jnp.linspace(-5, 5, 200).reshape(-1, 1)
y_test = jnp.tanh(x_test)

y_pred_samples = []
n_mc_samples = 100
for _ in range(n_mc_samples):
    key, subkey = random.split(key)
    y_pred = forward(mlp_params, x_test, subkey, dropout)
    y_pred_samples.append(y_pred)
y_pred_samples = jnp.stack(y_pred_samples)
y_pred_mean = jnp.mean(y_pred_samples, axis=0)
y_pred_std = jnp.std(y_pred_samples, axis=0)

plot_predictions(
    x,
    y,
    x_test,
    y_test,
    y_pred_mean,
    y_pred_std=y_pred_std,
    title="MC Dropout Predictions",
).display()

# %%
x_rich = jnp.concatenate([x, x / 6 + 4], axis=0)
y_rich = jnp.concatenate([y, jnp.tanh(x / 6 + 4) + 0.1 * random.normal(key, x.shape)], axis=0)

mlp_params_rich = init_mlp_params(subkey, mlp_dims)
for epoch in range(n_epochs):
    key, subkey = random.split(key)
    mlp_params_rich = update(mlp_params_rich, x_rich, y_rich, subkey, dropout, lr=lr)

y_pred_samples_rich = []
n_mc_samples = 100
for _ in range(n_mc_samples):
    key, subkey = random.split(key)
    y_pred_rich = forward(mlp_params_rich, x_test, subkey, dropout)
    y_pred_samples_rich.append(y_pred_rich)
y_pred_samples_rich = jnp.stack(y_pred_samples_rich)
y_pred_mean_rich = jnp.mean(y_pred_samples_rich, axis=0)
y_pred_std_rich = jnp.std(y_pred_samples_rich, axis=0)

plot_predictions(
    x_rich,
    y_rich,
    x_test,
    y_test,
    y_pred_mean_rich,
    y_pred_std=y_pred_std_rich,
    title="MC Dropout Predictions (Rich Data)",
).display()

# %%
