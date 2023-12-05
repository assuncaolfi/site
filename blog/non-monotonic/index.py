# %% [markdown]
# ---
# title: Decomposable non-monotonic models
# date: 2023-11-17
# categories: [causal, cognitive, modeling]
# ---

# %% [markdown]
"""
_This post is a work in progress._

Recently, I helped design an experiment measuring a binary response against a
continuous delay time. If the user did not do the thing at time zero, then we
delayed for a variable number of minutes before reminding them to do it. This
delay had a non-monotonic relationship to the response: as the delay increased,
the response responded differently. Initially, the response increased; then it
peaked; and finally it decreased.

Causally, we may decompose this process into two: assuming the user forgot to/
could not do the thing at the time, as the delay increases, they  1) become
more available for and 2) lose interest in doing the thing. This is a common
phenomena in different time-based scenarios. In sports, the "aging curve" refers
to how a player's performance increases with age, then decreases. As the player
gets older, they get 1) better at the sport and 2) physically weaker.

Andrew Gelman wrote about this a couple of times in his blog: see 
[this post from 2018](https://statmodeling.stat.columbia.edu/2018/09/07/bothered-non-monotonicity-heres-one-quick-trick-make-happy/)
and [this one from 2023](https://statmodeling.stat.columbia.edu/2023/01/01/how-to-model-a-non-monotonic-relation/),
as well as their comments, which also informed this post. Gelman proposed that
we should model these processes like this:

$$g(x) = g_1(x) + g_2(x),$$

where  
$g_1(x)$ is a monotonically increasing function with a right asymptote; and  
$g_2(x)$ is a monotonically decreasing function with a left asymptote.

In this post, we will go over some of these models and test them on a dataset.

## Mind-in-Eyes

The dataset for the study is available... [@Hartshorne2015]

We could control for other variables, such as the computer type (desk or
laptop), but let's assume there are no confounding effects at play here.
"""

# %%
# | label: mind-in-eyes-read
from rich.pretty import pprint
import polars as pl

mind_eyes = (
    pl.read_csv("data/mind-in-eyes.csv")
    .with_columns(total_correct=pl.col("total.correct"))
    .select("age", "total_correct")
    .with_columns(
        y=(pl.col("total_correct") - pl.col("total_correct").mean())
        / pl.col("total_correct").std()
    )
)

# %%
# | label: mind-in-eyes-plot
from site import theme

theme.set()

fig = (
    so.Plot(mind_eyes.to_pandas(), x="age", y="y")
    .label(x="Age", y="Correct answers (z-score)", title="Mind-in-Eyes Task")
    .add(so.Dots())
)
fig

# %% [markdown]
"""
## Empirical models

After struggling with splines,
Splines, Gaussian Processes...

## Decomposable models

Some commenters on Andrew's blog...

All intervals are 80% credibility...

$$
\begin{align}
g(x) = g_1(x) + g_2(x) \\
y \sim \mathrm{Normal}(g(x), \sigma) \\
\sigma \sim \mathrm{HalfNormal}(1)
\end{align}
$$

### Laurent polynomial

Degree = 2, order = -1.

$$
g(x) = ax^{-1} + bx + cx^2
$$

### Siler

$$
g(x) = \alpha_1 \exp(-\lambda_1 x) + \alpha_2 + \alpha_3 \exp(\lambda_2 x)
$$

With priors:

$$
\begin{align}
\alpha \sim \mathrm{Normal}(0, 2) \\
\lambda \sim \mathrm{HalfNormal}(0.01) \\
\end{align}
$$

"""

# %%
# | label: parametric-models-siler-prior-predictive
# | warning: false
import arviz as az
import numpy as np
import pymc as pm


def g(x):
    y = α[0] * pm.math.exp(-1 * λ[0] * x) + α[1] + α[2] * pm.math.exp(λ[1] * x)
    return y


def add_bands(fig: so.Plot, prediction: az.InferenceData, distribution: str):
    summary = az.summary(prediction, hdi_prob=0.8)
    summary["age"] = domain
    fig = fig.add(
        so.Band(),
        data=summary,
        x="age",
        y="mean",
        ymin="hdi_10%",
        ymax="hdi_90%",
    ).label(title=f"{distribution} Predictive")
    return fig


ages = mind_eyes.get_column("age")
y = mind_eyes.get_column("y")
domain = np.arange(ages.min(), ages.max() + 1)

with pm.Model() as siler:
    α = pm.Normal("alpha", 0, 2, size=3)
    λ = pm.HalfNormal("lambda", 0.01, size=2)
    σ = pm.HalfNormal("sigma", 1)
    x = pm.Data("x", ages)
    mu = pm.Deterministic("mu", g(x))
    pm.Normal("y", mu=mu, sigma=σ, observed=y)
    prediction = pm.Deterministic("prediction", g(domain))
    traces = pm.sample_prior_predictive(samples=2000, random_seed=37)

add_bands(fig, traces.prior.prediction, "Prior")

# %%
# | label: parametric-models-siler-posterior-predictive
# | warning: false
with siler:
    traces = pm.sample(progressbar=False, random_seed=37)

# print(az.summary(traces))
add_bands(fig, traces.posterior.prediction, "Posterior")

# %% [markdown]
"""
### McElreath

$$
g(x) = \exp(-ax) (1 - exp(-bx))^c
$$

## Comparison

Let's compare using LOO...
"""
