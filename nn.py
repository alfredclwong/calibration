import jax.numpy as jnp
from jax import random, jit, grad
from jax.scipy.special import logsumexp


def init_kaiming_layer(key, n_in, n_out):
    w_key, b_key = random.split(key)
    w = random.normal(w_key, (n_in, n_out)) * jnp.sqrt(2.0 / n_in)
    b = random.normal(b_key, (n_out,))
    return w, b


def init_mlp_params(key, sizes):
    keys = random.split(key, len(sizes))
    return [init_kaiming_layer(k, m, n) for k, m, n in zip(keys, sizes[:-1], sizes[1:])]


@jit
def relu(x):
    return jnp.maximum(0, x)


@jit
def drop(key, x, dropout):
    keep_prob = 1.0 - dropout
    mask = random.bernoulli(key, p=keep_prob, shape=x.shape)
    return x * mask / keep_prob


@jit
def forward(params, x):
    n = len(params)
    for i, (w, b) in enumerate(params):
        x = jnp.dot(x, w) + b
        if i < n - 1:
            x = relu(x)
    return x


@jit
def forward_drop(params, x, key, dropout):
    n = len(params)
    for i, (w, b) in enumerate(params):
        key, subkey = random.split(key)
        # w = drop(subkey, w, dropout)
        x = jnp.dot(x, w) + b
        if i < n - 1:
            x = relu(x)
            x = drop(subkey, x, dropout)
    return x


@jit
def logits_to_logprobs(logits):
    return logits - logsumexp(logits)


@jit
def mse_loss_fn(params, x, y):
    y_pred = forward(params, x)
    return jnp.mean((y_pred - y) ** 2)


@jit
def mse_loss_fn_drop(params, x, y, key, dropout):
    y_pred = forward_drop(params, x, key, dropout)
    return jnp.mean((y_pred - y) ** 2)


@jit
def update(params, x, y, lr=0.001):
    grads = grad(mse_loss_fn)(params, x, y)
    return [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)]


@jit
def update_drop(params, x, y, key, dropout, lr=0.001):
    grads = grad(mse_loss_fn_drop)(params, x, y, key, dropout)
    return [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)]
