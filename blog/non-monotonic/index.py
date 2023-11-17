# %% [markdown]
# ---
# title: Non-monotonic models
# date: 2023-11-17
# categories: [modeling, inference]
# ---

# %% [markdown]
"""
_This post is a work in progress._

Recently, I helped design an experiment using a continuous treatment variable,
with a non-monotonic relationship with the response.

I came across a 2018 blog post by Andrew Gelman on [additive models
for non-monotonic functions](https://statmodeling.stat.columbia.edu/2018/09/07/
bothered-non-monotonicity-heres-one-quick-trick-make-happy/).

In this post, we will go over some of these models and test them on the Mind-in-
Eyes Task dataset [@Hartshorne2015].

## Polynomials

$$
g(x) = ax^{-1} + bx + cx^2
$$

## Siler

$$
g(x) = a \exp(-b x) + c + d \exp(ex)
$$

## McElreath

$$
g(x) = \exp(-ax) (1 - exp(-bx))^c
$$
"""
