import einops
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.activation import sigmoid, tanh

from jaxtyping import Array, Float, Int
from typing import Any, Callable, List, Optional, Tuple, Dict

# from src.models.hippo.hippo_live import HiPPOCell
from hippo_live import HiPPOCell

class Cell(nn.Module):
    @staticmethod
    def initialize_carry(rng, batch_size, hidden_size, init_fn=nn.initializers.zeros):
        return jnp.zeros((batch_size, hidden_size))


class LSTMCell(nn.Module):
    features: int
    bias: bool = True
    gate_fn: Callable[..., Any] = sigmoid
    activation_fn: Callable[..., Any] = tanh
    dtype: Any = jnp.float32

    def setup(self):
        self.dense_i = nn.Dense(
            self.features * 4,
            use_bias=self.bias,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            param_dtype=self.dtype,
            name="dense_i",
        )
        self.dense_h = nn.Dense(
            self.features * 4,
            use_bias=self.bias,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            param_dtype=self.dtype,
            name="dense_h",
        )

    def __call__(self, carry, inputs):
        h_t, c_t = carry
        gates_i = self.dense_i(inputs)
        gates_h = self.dense_h(h_t)

        # get the gate outputs
        i_t, f_t, g_t, o_t = jnp.split(
            gates_i + gates_h, indices_or_sections=4, axis=-1
        )
        i_t = self.gate_fn(i_t)
        f_t = self.gate_fn(f_t)
        o_t = self.gate_fn(o_t)
        g_t = self.activation_fn(g_t)

        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * self.activation_fn(c_t)

        return h_t, c_t


class GateHiPPOCell(Cell):
    hippo_cell: Callable[..., HiPPOCell]
    hippo_args: Dict
    tau_args: Dict  # referencing tau as the general symbol used in the paper as some general RNN, can be a normal RNN, a LSTM, or a GRU
    _tau: Callable[..., Cell] = LSTMCell
    bias: bool = True
    dtype: Any = jnp.float32

    def setup(self):
        self.hippo = self.hippo_cell(**self.hippo_args)
        self.tau = self._tau(**self.tau_args)
        self.hippo_proj = nn.Dense(
            features=1,
            use_bias=self.bias,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            param_dtype=self.dtype,
            name="hippo_proj",
        )
        assert isinstance(self.tau, LSTMCell), "Only a RNN Cell is supported"
        assert isinstance(self.hippo, HiPPOCell), "Only a HiPPO Cell is supported"

    def __call__(self, carry, x):
        h_t, c_t_1 = carry
        tau_x = jnp.append(c_t_1, x)

        # get the hidden state from the RNN of choice
        h_t, _ = self.tau(carry, tau_x)

        # project the hidden state to a scalar to behave as the scalar input of the underlying signal
        f_t = einops.rearrange(
            self.hippo_proj(einops.rearrange(h_t, "features -> 1 features")), "1 t -> t"
        )

        # get the hidden state from the HiPPO and put it into a form HiPPO can use
        c_t_1 = einops.rearrange(c_t_1, "N -> 1 N")

        # we do not backpropagate through the HiPPO, so we stop the gradient
        c_t, _ = jax.lax.stop_gradient(self.hippo(c_t_1, f_t))

        # reshape cell state so next iteration of RNN (tau) can use it
        c_t = einops.rearrange(c_t, "1 N -> N")

        return h_t, c_t

    @staticmethod
    def initialize_carry(
        self, rng, batch_size, hidden_size, init_fn=nn.initializers.zeros
    ):
        key1, key2 = jax.random.split(rng)
        mem_shape = (batch_size, hidden_size)
        return init_fn(key1, mem_shape), init_fn(key2, mem_shape)


# BatchedGatedHiPPOCell is a vectorized form of GateHiPPOCell
class BatchedGatedHiPPOCell(GateHiPPOCell):
    hippo_cell: Callable[..., HiPPOCell]
    hippo_args: Dict
    tau_args: Dict
    # ^ referencing tau as the general symbol used in the paper as some general RNN, can be a normal RNN, a LSTM, or a GRU
    _tau: Callable[..., Cell] = LSTMCell
    bias: bool = True
    dtype: Any = jnp.float32

    def setup(self) -> None:
        self.cell = nn.vmap(
            target=GateHiPPOCell,
            in_axes=(0, 0),
            variable_axes={"params": 0},
            split_rngs={"params": True},
        )(
            hippo_cell=self.hippo_cell,
            hippo_args=self.hippo_args,
            tau_args=self.tau_args,
            # mlp_args=self.mlp_args,
            _tau=self._tau,
            bias=self.bias,
            dtype=self.dtype,
        )

    def __call__(self, carry, x):
        return self.cell(carry, x)
    

class CharRNN(nn.Module):

    vocab_size: int
    hidden_size: int
    rnn_cells: List[Cell]
    cell_args: List[Dict]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_size,
            embedding_init=nn.initializers.normal(stddev=0.05),
            dtype=self.dtype,
            name="embedding",
        )
        self.cells = [
            cell(**args) for cell, args in zip(self.rnn_cells, self.cell_args)
        ]
        self.output_proj = nn.Dense(
            features=self.vocab_size,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,  #(),
            dtype=self.dtype,
            name="output_proj",
        )

    def __call__(self, carry, x, targets=None):
        embeddings = self.embedding(targets if targets is not None else x)
        new_carries = carry
        logits_sequence = []

        for t in range(x.shape[1]):
            embedded = (
                embeddings[:, t, :]
                if targets is not None or t == 0
                else new_carries[-1][0]
            )

            for i, cell in enumerate(self.cells):
                cell_input = embedded if i == 0 else new_carries[i - 1][0]
                new_carries[i] = cell(new_carries[i], cell_input)

            logit = self.output_proj(new_carries[-1][0])
            logits_sequence.append(logit)

        logits = jnp.stack(logits_sequence, axis=1)
        return logits, new_carries

    def initialize_carries(
        self,
        rng,
        batch_size: int,
        hidden_sizes: List[int],
        init_fn=nn.initializers.zeros,
    ):
        carry = []
        for hs in hidden_sizes:
            mem_shape = (batch_size, hs)
            key1, key2, key3 = jax.random.split(rng, num=3)
            carry.append((init_fn(key2, mem_shape), init_fn(key3, mem_shape)))
            rng = key1

        return carry