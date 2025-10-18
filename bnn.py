# %%
import jax
import jax.numpy as jnp
from jax import jit
from jax import random
import pandas as pd
from tqdm.auto import tqdm

from nn import init_mlp_params, forward, mse_loss_fn, update
from vis import plot_predictions, plot_losses


# %%
def mcmc(key, x, y, theta_0, log_likelihood_fn, log_prior_fn, n_samples, step_size=0.5):
    samples = []
    theta = theta_0
    for _ in tqdm(range(n_samples)):
        key, subkey = random.split(key)
        theta_prop = jax.tree.map(
            lambda t: t + step_size * random.normal(subkey, t.shape), theta
        )
        log_alpha = (
            log_likelihood_fn(theta_prop, x, y)
            - log_likelihood_fn(theta, x, y)
            + log_prior_fn(theta_prop)
            - log_prior_fn(theta)
        )
        key, subkey = random.split(key)
        accept = jnp.log(random.uniform(subkey)) < log_alpha
        if accept:
            theta = theta_prop
        samples.append(theta)
    return samples


@jit
def log_likelihood_fn(params, x, y, sigma=0.1):
    y_pred = forward(params, x)
    return -jnp.mean((y_pred - y) ** 2) / (2 * sigma**2)


@jit
def log_prior_fn(params, sigma=1.0):
    log_prob = 0.0
    for w, b in params:
        log_prob += jnp.sum(
            -0.5 * (w / sigma) ** 2 - jnp.log(jnp.sqrt(2 * jnp.pi) * sigma)
        )
        log_prob += jnp.sum(
            -0.5 * (b / sigma) ** 2 - jnp.log(jnp.sqrt(2 * jnp.pi) * sigma)
        )
    return log_prob


def predict_with_samples(x, samples):
    y_preds = []
    for params in samples:
        y_pred = forward(params, x)
        y_preds.append(y_pred)
    return jnp.stack(y_preds)


# %%
key = random.PRNGKey(0)
key, subkey = random.split(key)
mlp_dims = [1, 8, 8, 1]
mlp_params = init_mlp_params(subkey, mlp_dims)

x = jnp.linspace(-3, 3, 200).reshape(-1, 1)
y = jnp.tanh(x) + 0.1 * random.normal(key, x.shape)

losses = []
n_epochs = 1000
for epoch in range(n_epochs):
    key, subkey = random.split(key)
    mlp_params = update(mlp_params, x, y, lr=0.01)
    if epoch % 100 == 0 or epoch == n_epochs - 1:
        loss = mse_loss_fn(mlp_params, x, y)
        losses.append((epoch, loss.item()))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
loss_df = pd.DataFrame(losses, columns=["epoch", "loss"])

plot_losses(loss_df).display()

# %%
x_test = jnp.linspace(-5, 5, 1000).reshape(-1, 1)
y_test_true = jnp.tanh(x_test)
y_test_pred = forward(mlp_params, x_test)

plot_predictions(x, y, x_test, y_test_true, y_test_pred).display()

# %%
key, subkey = random.split(key)
theta_0s = [init_mlp_params(subkey, mlp_dims) for _ in range(5)]
n_samples = 2000
n_burn = 100
key, *subkeys = random.split(key, len(theta_0s) + 1)
samples = [
    mcmc(
        subkey,
        x,
        y,
        theta_0,
        lambda params, x, y: log_likelihood_fn(params, x, y, sigma=0.1),
        lambda params: log_prior_fn(params, sigma=3.0),
        n_samples,
        step_size=0.1,
    )
    for theta_0, subkey in zip(theta_0s, subkeys)
]
samples = [s for chain in samples for s in chain]  # flatten the list

# %%
y_preds = predict_with_samples(x_test, samples[n_burn:])
y_pred_mean = jnp.mean(y_preds, axis=0)
y_pred_std = jnp.std(y_preds, axis=0)

fig = plot_predictions(x, y, x_test, y_test_true, y_pred_mean, y_pred_std)
fig.save("bnn.svg")
fig.display()

# %%
x_rich = jnp.concatenate([x, x / 6 + 4], axis=0)
key, subkey = random.split(key)
y_rich = jnp.tanh(x_rich) + 0.1 * random.normal(subkey, x_rich.shape)
key, subkey = random.split(key)
theta_0s = [init_mlp_params(subkey, mlp_dims) for _ in range(5)]

key, *subkeys = random.split(key, len(theta_0s) + 1)
samples_rich = [
    mcmc(
        subkey,
        x_rich,
        y_rich,
        theta_0,
        lambda params, x, y: log_likelihood_fn(params, x, y, sigma=0.1),
        lambda params: log_prior_fn(params, sigma=5.0),
        n_samples,
        step_size=0.1,
    )
    for theta_0, subkey in zip(theta_0s, subkeys)
]
samples_rich = [s for chain in samples_rich for s in chain]  # flatten the list
y_preds_rich = predict_with_samples(x_test, samples_rich[n_burn:])
y_pred_mean_rich = jnp.mean(y_preds_rich, axis=0)
y_pred_std_rich = jnp.std(y_preds_rich, axis=0)

fig_r = plot_predictions(
    x_rich, y_rich, x_test, y_test_true, y_pred_mean_rich, y_pred_std_rich
)
fig_r.save("bnn_rich.svg")
fig_r.display()

# %%
