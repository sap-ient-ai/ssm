## import packages
import numpy as np
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float

from scipy import special as ss
from typing import Any, Callable, Tuple, Union

# Factory for constant initializer in Flax
def initializer(x: Union[Float[Array, "N N"], Float[Array, "N 1"]]) -> Callable:
    def _init(key, shape: Tuple):
        assert shape == x.shape
        return x

    return _init


def legt(N: int, dtype=jnp.float32):
    q = jnp.arange(N, dtype=dtype)
    k, n = jnp.meshgrid(q, q)
    _case = jnp.power(-1, n - k)

    # building the B matrix based off equation the LMU derivation within paper, specifically when lambda is equal to 1
    pre_D = jnp.sqrt(jnp.diag(2 * q + 1))
    B = jnp.diag(pre_D)[:, None]

    # building the B matrix based off equation the LMU derivation within paper, specifically when lambda is equal to 1
    A_base = jnp.sqrt(2 * n + 1) * jnp.sqrt(2 * k + 1)
    A = jnp.where(k <= n, A_base, A_base * _case)

    return -A, B


def legs(N: int, dtype=jnp.float32):
    q = jnp.arange(N, dtype=dtype)
    k, n = jnp.meshgrid(q, q)

    # building the B matrix based off equation 29 within paper
    pre_D = jnp.sqrt(jnp.diag(2 * q + 1))
    B = jnp.diag(pre_D)[:, None]

    # building the A matrix based off equation 30 within paper
    A_base = jnp.sqrt(2 * n + 1) * jnp.sqrt(2 * k + 1)
    A = jnp.where(n > k, A_base, jnp.where(n == k, n + 1, 0.0))

    return -A, B


def lmu(
    N: int, dtype: Any = jnp.float32
) -> Tuple[Float[Array, "N N"], Float[Array, "N 1"], str]:
    q = jnp.arange(N, dtype=dtype)
    k, n = jnp.meshgrid(q, q)
    _case = jnp.power(-1.0, (n - k))

    A_base = 2 * n + 1
    B = jnp.diag((2 * q + 1) * jnp.power(-1, n))[:, None]
    A = jnp.where(
        k <= n, A_base * _case, A_base
    )  # if n >= k, then _case_2 * A_base is used, otherwise A_base

    return -A.astype(dtype), B.astype(dtype), "lmu"


def lagt(
    N: int, alpha: float = 0.0, beta: float = 1.0, dtype: Any = jnp.float32
) -> Tuple[Float[Array, "N N"], Float[Array, "N 1"], str]:
    L = jnp.exp(
        0.5
        * jnp.asarray(
            ss.gammaln(np.arange(N) + alpha + 1) - ss.gammaln(np.arange(N) + 1)
        )
    )
    inv_L = 1.0 / L[:, None]
    pre_A = (jnp.eye(N) * ((1 + beta) / 2)) + jnp.tril(jnp.ones((N, N)), -1)
    pre_B = jnp.asarray(ss.binom(alpha + np.arange(N), np.arange(N)))[:, None]

    A = -inv_L * pre_A * L[None, :]
    B = (
        jnp.exp(-0.5 * jnp.asarray(ss.gammaln(1 - alpha)))
        * jnp.power(beta, (1 - alpha) / 2)
        * inv_L
        * pre_B
    )

    return A.astype(dtype), B.astype(dtype), "lagt"


def fru(
    N: int, dtype: Any = jnp.float32
) -> Tuple[Float[Array, "N N"], Float[Array, "N 1"], str]:
    A = jnp.diag(
        jnp.stack([jnp.zeros(N // 2), jnp.zeros(N // 2)], axis=-1).reshape(-1)[1:],
        1,
    )
    B = jnp.zeros(A.shape[1], dtype=dtype)

    B = B.at[0::2].set(jnp.sqrt(2))
    B = B.at[0].set(1)

    q = jnp.arange(A.shape[1], dtype=dtype)
    k, n = jnp.meshgrid(q, q)

    n_odd = n % 2 == 0
    k_odd = k % 2 == 0

    _case_1 = (n == k) & (n == 0)
    _case_2_3 = ((k == 0) & (n_odd)) | ((n == 0) & (k_odd))
    _case_4 = (n_odd) & (k_odd)
    _case_5 = (n - k == 1) & (k_odd)
    _case_6 = (k - n == 1) & (n_odd)

    A = jnp.where(
        _case_1,
        -1.0,
        jnp.where(
            _case_2_3,
            -jnp.sqrt(2),
            jnp.where(
                _case_4,
                -2,
                jnp.where(
                    _case_5,
                    jnp.pi * (n // 2),
                    jnp.where(_case_6, -jnp.pi * (k // 2), 0.0),
                ),
            ),
        ),
    )
    B = B[:, None]

    return A.astype(dtype), B.astype(dtype), "fru"


def fout(
    N: int, dtype: Any = jnp.float32
) -> Tuple[Float[Array, "N N"], Float[Array, "N 1"], str]:
    A = jnp.diag(
        jnp.stack([jnp.zeros(N // 2), jnp.zeros(N // 2)], axis=-1).reshape(-1)[1:],
        1,
    )
    B = jnp.zeros(A.shape[1], dtype=dtype)

    B = B.at[0::2].set(jnp.sqrt(2))
    B = B.at[0].set(1)

    q = jnp.arange(A.shape[1], dtype=dtype)
    k, n = jnp.meshgrid(q, q)

    n_odd = n % 2 == 0
    k_odd = k % 2 == 0

    _case_1 = (n == k) & (n == 0)
    _case_2_3 = ((k == 0) & (n_odd)) | ((n == 0) & (k_odd))
    _case_4 = (n_odd) & (k_odd)
    _case_5 = (n - k == 1) & (k_odd)
    _case_6 = (k - n == 1) & (n_odd)

    A = jnp.where(
        _case_1,
        -1.0,
        jnp.where(
            _case_2_3,
            -jnp.sqrt(2),
            jnp.where(
                _case_4,
                -2,
                jnp.where(
                    _case_5,
                    jnp.pi * (n // 2),
                    jnp.where(_case_6, -jnp.pi * (k // 2), 0.0),
                ),
            ),
        ),
    )
    A = 2 * A
    B = 2 * B
    B = B[:, None]

    return A.astype(dtype), B.astype(dtype), "fout"


def foud(
    N: int, dtype: Any = jnp.float32
) -> Tuple[Float[Array, "N N"], Float[Array, "N 1"], str]:
    A = jnp.diag(
        jnp.stack([jnp.zeros(N // 2), jnp.zeros(N // 2)], axis=-1).reshape(-1)[1:],
        1,
    )
    B = jnp.zeros(A.shape[1], dtype=dtype)

    B = B.at[0::2].set(jnp.sqrt(2))
    B = B.at[0].set(1)

    q = jnp.arange(A.shape[1], dtype=dtype)
    k, n = jnp.meshgrid(q, q)

    n_odd = n % 2 == 0
    k_odd = k % 2 == 0

    _case_1 = (n == k) & (n == 0)
    _case_2_3 = ((k == 0) & (n_odd)) | ((n == 0) & (k_odd))
    _case_4 = (n_odd) & (k_odd)
    _case_5 = (n - k == 1) & (k_odd)
    _case_6 = (k - n == 1) & (n_odd)

    A = jnp.where(
        _case_1,
        -1.0,
        jnp.where(
            _case_2_3,
            -jnp.sqrt(2),
            jnp.where(
                _case_4,
                -2,
                jnp.where(
                    _case_5,
                    2 * jnp.pi * (n // 2),
                    jnp.where(_case_6, 2 * -jnp.pi * (k // 2), 0.0),
                ),
            ),
        ),
    )

    A = 0.5 * A
    B = 0.5 * B
    B = B[:, None]

    return A.astype(dtype), B.astype(dtype), "foud"
