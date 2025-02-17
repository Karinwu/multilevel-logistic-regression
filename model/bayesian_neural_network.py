import argparse
import os
import time
from typing import Optional

from jax import vmap
import jax.numpy as jnp
import jax.random
import numpy as np
import numpyro as npr
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.reparam import LocScaleReparam
import pandas as pd

