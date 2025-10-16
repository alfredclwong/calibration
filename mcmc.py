# %%
import torch
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go

# %%
z = torch.distributions.Normal(0, 1)
ll = lambda x: z.log_prob((x + 1) * (x - 1) ** 2)
x = torch.linspace(-3, 3, 100)
y = torch.exp(ll(x))

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=x.numpy(), y=y.numpy(), mode="lines", name="pdf"))
fig.update_layout(height=400, width=600)
fig.show()

# %%
n = 10
T = 10000
T_burn = 50
prior = torch.distributions.Normal(0, 5)
theta = prior.sample((n,))
thetas = [theta.numpy()]
for t in tqdm(range(T)):
    theta_prop = theta + 0.5 * torch.randn(n)
    log_alpha = ll(theta_prop) - ll(theta) + prior.log_prob(theta_prop) - prior.log_prob(theta)
    theta = torch.where(torch.log(torch.rand(n)) < log_alpha, theta_prop, theta)
    thetas.append(theta.numpy())
thetas = np.array(thetas)

# %%
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=x.numpy(), y=y.numpy(), mode="lines", name="pdf"),
)
fig.add_trace(
    go.Histogram(
        x=thetas[T_burn:].flatten(),
        nbinsx=100,
        name="samples",
        yaxis="y2",
        opacity=0.6,
    ),
)
fig.update_layout(
    yaxis=dict(title="pdf", range=[0, 0.5]),
    yaxis2=dict(title="samples", overlaying="y", side="right", range=[0, 2500])
)
fig.show()

# %%
