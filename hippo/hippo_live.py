## import packages
from typing import Any, Callable, Tuple, Union, Dict

import numpy as np
import einops
import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array, Float, Int
from scipy import special as ss

# from src.models.hippo.trans import legs, initializer
from trans import legs, initializer


class HiPPOCell(nn.Module):
    def discretize(self, A, B, step, alpha, dtype, **kwargs):
        raise NotImplementedError

    @staticmethod
    def initialize_state(rng, batch_size, channels, hidden_size, init_fn):
        raise NotImplementedError


class HiPPOLTI(HiPPOCell):

    step_size: float  # this is normally denoted as dt or delta within the HiPPO/S4 Literature
    basis_size: float
    alpha: float
    A_init: Callable = legs
    B_init: Callable = legs
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, c_t_1, x):
        hidden_features = c_t_1.shape[-1]
        channels = c_t_1.shape[-2]

        A, _ = self.A_init(hidden_features, dtype=self.dtype)
        _, B = self.B_init(hidden_features, dtype=self.dtype)

        _B = jnp.tile(
            B, (channels,)
        )  # make channel number of B matrices to account for making channel number of HiPPOs. HiPPO only works on a 1-D signal

        # discretize the A and B matrices
        A_d, B_d = self.discretize(
            A, _B, step=self.step_size, alpha=self.alpha, dtype=self.dtype
        )

        # initialize the A and B matrices as functions so we can make them trainable parameters (if we want)
        A_init_fn = initializer(A_d)
        B_init_fn = initializer(B_d)

        # make the A and B matrices trainable parameters
        A_d = self.param("A_d", A_init_fn, (hidden_features, hidden_features))
        B_d = self.param("B_d", B_init_fn, (hidden_features, channels))

        # formulate the HiPPO operator

        # c_t = A_d @ c_t_1 + B_d @ x

        # A_d   (128, 128)
        # c_t_1 (1, 128)
        # B_d   (128, 1)
        # x     (1,)
        '''
        c = nChannels
        n = nHidden
        m = nHidden
        '''
        c_t = (
             jnp.einsum("mn, cn -> cn", A_d, c_t_1) + jnp.einsum("nc, c -> cn", B_d, x)
        )

        return c_t, c_t

    def discretize(self, A, B, step, alpha=0.5, dtype=jnp.float32):
        assert alpha in [0, 0.5, 1, 2], "alpha must be 0, 0.5, 1, 2"

        I = jnp.eye(A.shape[0], dtype=dtype)

        # Generalized Bilinear Transformation
        # referencing equation 13 within the discretization method section of the HiPPO paper
        if alpha <= 1:
            step_size = step
            part1 = I - (step_size * alpha * A)
            part2 = I + (step_size * (1 - alpha) * A)

            GBT_A = jnp.linalg.solve(part1, part2)
            GBT_B = jnp.linalg.solve(part1, step_size * B)

        # Zero-Order Hold
        else:
            # refer to this for why this works
            # https://en.wikipedia.org/wiki/Discretization#:~:text=A%20clever%20trick%20to%20compute%20Ad%20and%20Bd%20in%20one%20step%20is%20by%20utilizing%20the%20following%20property

            n = A.shape[0]
            b_n = B.shape[1]
            A_B_square = jnp.block(
                [[A, B], [jnp.zeros((b_n, n)), jnp.zeros((b_n, b_n))]]
            )
            A_B = jax.scipy.linalg.expm(A_B_square * self.step_size)

            if A_B.dtype != dtype:
                A_B = A_B.astype(dtype)

            GBT_A = A_B[0:n, 0:n]
            GBT_B = A_B[0:-b_n, -b_n:]

        return GBT_A, GBT_B

    @staticmethod
    def initialize_state(rng, batch_size, channels, hidden_size, init_fn):
        mem_shape = (batch_size, channels, hidden_size)
        return init_fn(rng, mem_shape)
