"""
brain.py — genome -> feedforward neural network ("the agent's brain").

Genome: a list of Gene(source_type, source_id, sink_type, sink_id, weight).
  source_type in {"SENSOR", "INTERNAL"}; sink_type in {"INTERNAL", "ACTION"}.

Node sets (v1 fixed sizes):
  SENSOR nodes:   N_SENSORS (from sensors.py, includes the bias node)
  INTERNAL nodes: N_INTERNAL (tunable constant below)
  ACTION nodes:   one per coin, N_COINS (from sensors.COINS)

Topology (v1): feedforward, SINGLE hidden layer. Allowed connections ONLY:
  SENSOR -> INTERNAL, SENSOR -> ACTION (skip connection), INTERNAL -> ACTION.
  NOT allowed: INTERNAL -> INTERNAL (no recurrence/depth), anything -> SENSOR,
  ACTION -> anything.

Forward pass each hour (pure function of (genome, sensors), no hidden state carried
between calls):
  internal_j = tanh( sum of weight * source_value over incoming SENSOR connections )
  action_k   = tanh( sum of weight * source_value over incoming SENSOR/INTERNAL connections )
"""

from collections import namedtuple

import numpy as np

from sensors import N_SENSORS, COINS

N_INTERNAL = 8                      # tunable: size of the single hidden layer
N_COINS = len(COINS)
WEIGHT_MIN, WEIGHT_MAX = -4.0, 4.0  # tunable: random_genome's weight sampling range

Gene = namedtuple("Gene", ["source_type", "source_id", "sink_type", "sink_id", "weight"])

# The only connection (source_type, sink_type) pairs a valid gene may have.
VALID_CONNECTIONS = (("SENSOR", "INTERNAL"), ("SENSOR", "ACTION"), ("INTERNAL", "ACTION"))


def random_genome(rng, n_genes):
    """A random sparse genome: n_genes independently-sampled connections, each drawn
    uniformly from the allowed connection types, with random valid endpoint ids and a
    random weight in [WEIGHT_MIN, WEIGHT_MAX]. Duplicate connections are allowed (their
    contributions simply sum in the forward pass) -- no dedup, kept simple.
    """
    genes = []
    for _ in range(n_genes):
        source_type, sink_type = VALID_CONNECTIONS[rng.integers(len(VALID_CONNECTIONS))]
        source_id = int(rng.integers(N_SENSORS) if source_type == "SENSOR" else rng.integers(N_INTERNAL))
        sink_id = int(rng.integers(N_INTERNAL) if sink_type == "INTERNAL" else rng.integers(N_COINS))
        weight = float(rng.uniform(WEIGHT_MIN, WEIGHT_MAX))
        genes.append(Gene(source_type, source_id, sink_type, sink_id, weight))
    return genes


def build_network(genome):
    """Compile a genome into per-node incoming-connection lists for a fast forward pass.
    Validates that every gene uses an allowed connection type (fails loudly on a
    malformed genome rather than silently ignoring it).
    """
    internal_inputs = [[] for _ in range(N_INTERNAL)]
    action_inputs = [[] for _ in range(N_COINS)]
    for g in genome:
        assert (g.source_type, g.sink_type) in VALID_CONNECTIONS, f"invalid connection type: {g}"
        if g.sink_type == "INTERNAL":
            internal_inputs[g.sink_id].append((g.source_id, g.weight))  # source_type must be SENSOR
        else:  # ACTION
            action_inputs[g.sink_id].append((g.source_type, g.source_id, g.weight))
    return {"internal_inputs": internal_inputs, "action_inputs": action_inputs}


def forward(network, sensors):
    """Pure function of (network, sensors) -> action vector (len N_COINS). No hidden
    state carried between calls -- single hidden layer, feedforward, no recurrence.
    """
    internal = np.empty(N_INTERNAL)
    for j, inputs in enumerate(network["internal_inputs"]):
        s = 0.0
        for source_id, weight in inputs:
            s += weight * sensors[source_id]
        internal[j] = np.tanh(s)

    actions = np.empty(N_COINS)
    for k, inputs in enumerate(network["action_inputs"]):
        s = 0.0
        for source_type, source_id, weight in inputs:
            s += weight * (sensors[source_id] if source_type == "SENSOR" else internal[source_id])
        actions[k] = np.tanh(s)

    return actions
