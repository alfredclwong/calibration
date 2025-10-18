# %%
import jax
import jax.numpy as jnp
from jax import jit
from jax import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from vis import plot_predictions


# %%
@jit
def sq_dist(x1, x2):
    return jnp.sum(x1**2, 1).reshape(-1, 1) + jnp.sum(x2**2, 1) - 2 * jnp.dot(x1, x2.T)


@jit
def rbf_kernel(x1, x2, length_scale=1.0, variance=1.0):
    return variance * jnp.exp(-0.5 / length_scale**2 * sq_dist(x1, x2))


def plot_kernels(x, length_scales, variances):
    x = jnp.linspace(-3, 3, 100).reshape(-1, 1)
    length_scales = [0.2, 0.5, 1.0, 2.0]
    variances = [0.5, 1.0, 2.0]

    fig = make_subplots(
        rows=len(length_scales),
        cols=len(variances),
        subplot_titles=[
            f"ls={ls}, var={var}" for ls in length_scales for var in variances
        ],
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.05,
        horizontal_spacing=0.0,
    )
    for i, ls in enumerate(length_scales):
        for j, var in enumerate(variances):
            K = rbf_kernel(x, x, length_scale=ls, variance=var) + 1e-6 * jnp.eye(len(x))
            fig.add_trace(
                go.Heatmap(
                    z=K,
                    x=np.round(x.flatten(), 2),
                    y=np.round(x.flatten(), 2),
                    colorscale="RdBu_r",
                ),
                row=i + 1,
                col=j + 1,
            )
    fig.update_yaxes(
        autorange="reversed",
        constrain="domain",
    )
    fig.update_xaxes(
        scaleanchor="y",
        scaleratio=1,
        constrain="domain",
    )
    fig.update_layout(
        width=900,
        height=900,
        coloraxis=dict(colorscale="RdBu_r"),
        showlegend=True,
    )
    for trace in fig.data:
        trace.update(coloraxis="coloraxis")
    return fig


# %%
x = jnp.linspace(-3, 3, 100).reshape(-1, 1)
length_scales = [0.2, 0.5, 1.0, 2.0]
variances = [0.5, 1.0, 2.0]
plot_kernels(x, length_scales, variances).show()


# %%
def sample_gp(key, x, K, n_samples=5):
    L = jnp.linalg.cholesky(K)
    key, subkey = random.split(key)
    u = random.normal(subkey, (len(x), n_samples))
    samples = jnp.dot(L, u).T
    return samples


def sample_gp_prior(key, x, n_samples=5, length_scale=1.0, variance=1.0):
    K = rbf_kernel(x, x, length_scale=length_scale, variance=variance)
    samples = sample_gp(key, x, K, n_samples)
    return samples


def sample_gp_posterior(
    key,
    x_train,
    y_train,
    x_test,
    n_samples=5,
    length_scale=1.0,
    variance=1.0,
    noise_variance=0.1,
):
    K_tt = rbf_kernel(
        x_train, x_train, length_scale, variance
    ) + noise_variance * jnp.eye(len(x_train))
    K_ts = rbf_kernel(x_train, x_test, length_scale, variance)
    K_ss = rbf_kernel(x_test, x_test, length_scale, variance) + 1e-6 * jnp.eye(
        len(x_test)
    )

    K_tt_inv = jnp.linalg.inv(K_tt)
    mu_s = jnp.dot(K_ts.T, jnp.dot(K_tt_inv, y_train)).reshape(-1, 1)
    cov_s = K_ss - jnp.dot(K_ts.T, jnp.dot(K_tt_inv, K_ts))

    var_s = jnp.diag(cov_s).reshape(-1, 1)
    samples = mu_s.T + sample_gp(key, x_test, cov_s, n_samples)
    return mu_s, var_s, samples


# %%
key = random.PRNGKey(0)

key, subkey = random.split(key)
x = jax.random.uniform(subkey, (50, 1), minval=-3, maxval=3)
y = jnp.tanh(x) + 0.1 * random.normal(key, x.shape)
x_test = jnp.linspace(-5, 5, 100).reshape(-1, 1)
y_test = jnp.tanh(x_test)

mu, var, gp_post_samples = sample_gp_posterior(
    key,
    x,
    y,
    x_test,
    n_samples=3,
    length_scale=1.0,
    variance=1.0,
    noise_variance=0.1,
)
std = jnp.sqrt(var)
fig = plot_predictions(
    x,
    y,
    x_test,
    y_test,
    mu,
    y_pred_std=std,
    title="GP Posterior Predictions",
)
fig.save("gp.svg")
fig.display()

# %%
x_rich = jnp.concatenate([x, x / 6 + 4], axis=0)
key, subkey = random.split(key)
y_rich = jnp.tanh(x_rich) + 0.1 * random.normal(subkey, x_rich.shape)

mu_r, var_r, gp_post_samples_r = sample_gp_posterior(
    key,
    x_rich,
    y_rich,
    x_test,
    n_samples=3,
    length_scale=1.0,
    variance=1.0,
    noise_variance=0.1,
)
std_r = jnp.sqrt(var_r)
fig_r = plot_predictions(
    x_rich,
    y_rich,
    x_test,
    y_test,
    mu_r,
    y_pred_std=std_r,
    title="GP Posterior Predictions with Richer Data",
)
fig_r.save("gp_rich.svg")
fig_r.display()

# %%
