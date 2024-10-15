import jax
import json
import re
import polars as pl
jax.config.update("jax_compilation_cache_dir", "jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

from jax import jit, vmap
from jax import random as jrand
from genspn.io import dataframe_to_arrays
from genspn.distributions import make_trace
import jax.numpy as jnp
from genspn.smc import smc


def get_arrays(df, test_ratio):
    schema, (numerical_array, categorical_array) = dataframe_to_arrays(df)
    test_ratio = 0.2
    n_train = int((1 - test_ratio) * len(df))
    if numerical_array is None:
        return schema, (categorical_array[:n_train], categorical_array[n_train:])
    elif categorical_array is None:
        return schema, (numerical_array[:n_train], numerical_array[n_train:])
    else:
        return schema, ((numerical_array[:n_train], categorical_array[:n_train]), (numerical_array[n_train:], categorical_array[n_train:]))


def learn_structure(df, seed=42, max_clusters=200, alpha=2, d=0.1, gibbs_iters=20, test_ratio=0.2):
    import warnings
    warnings.simplefilter("ignore", category=Warning)

    schema, (train_data, test_data) = get_arrays(df, test_ratio)

    key = jax.random.PRNGKey(seed)

    key, subkey = jax.random.split(key)
    trace = make_trace(subkey, alpha, d, schema, train_data, max_clusters)

    key, subkey = jax.random.split(key)
    trace, sum_logprobs = smc(subkey, trace, test_data, max_clusters, train_data, gibbs_iters, max_clusters)

    idx = jnp.argmax(sum_logprobs)
    print(f"Model index chosen: {idx}")
    cluster = trace.cluster[idx]
    mixture_weights = cluster.pi/jnp.sum(cluster.pi)
    components=cluster.f[:max_clusters]

    model_spec = {
        # No normal params.
        #"mu":components.dists[0].mu,
        #"sigma":components.dists[0].std,
        "logprobs": components.dists[0].logprobs,
        "cluster_weights": mixture_weights
    }
    return model_spec, schema
