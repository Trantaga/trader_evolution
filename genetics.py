"""
genetics.py — genetic operators for the trading-agent genome: crossover, mutation,
tournament selection. Genome format (source_type, source_id, sink_type, sink_id, weight)
and the fixed node skeleton (N_SENSORS, N_INTERNAL, N_COINS) are reused unchanged from
brain.py -- this file never redefines network semantics, only recombines/perturbs genes.

A gene's (source_type, source_id, sink_type, sink_id) tuple is its CONNECTION IDENTITY --
stable across all genomes since the node skeleton is fixed -- and is the locus used to
align parents during crossover.
"""

import numpy as np

from brain import Gene, VALID_CONNECTIONS, N_INTERNAL, N_COINS, WEIGHT_MIN, WEIGHT_MAX
from sensors import N_SENSORS

# --------------------------------------------------------------------------------------
# CONFIG -- tunable constants
# --------------------------------------------------------------------------------------
P_INHERIT_UNMATCHED = 0.3   # crossover: prob. a disjoint gene from the LESS-FIT parent survives
WEIGHT_MUT_SIGMA = 0.5      # mutate: stddev of the additive weight perturbation
P_PERTURB = 0.8              # mutate: prob. per gene of an additive weight perturbation (common)
P_RESET = 0.05                # mutate: prob. per gene of a full weight reset (rare)
P_ADD_CONN = 0.1              # mutate: prob. per genome of adding one new random connection
P_REMOVE_CONN = 0.1           # mutate: prob. per genome of removing one random connection
MIN_GENES = 5                  # guard: genomes never shrink below this many connections
MAX_GENES = 60                 # guard: genomes never grow past this many connections
TOURNAMENT_K = 3               # selection: tournament size


def connection_id(gene):
    """A gene's stable recombination locus: its (source,sink) identity, not its weight."""
    return (gene.source_type, gene.source_id, gene.sink_type, gene.sink_id)


# Fixed integer codes for node types, used only to build a canonical (hash-independent)
# sort order below -- NOT a change to the genome representation itself (genes still store
# "SENSOR"/"INTERNAL"/"ACTION" strings; this mapping is purely for ordering).
_NODE_TYPE_CODE = {"SENSOR": 0, "INTERNAL": 1, "ACTION": 2}


def connection_sort_key(cid):
    """Canonical, hash-independent total order over connection identities. Python
    randomizes string hashes per process (PYTHONHASHSEED), so iterating a `set` of
    connection-id tuples (which contain "SENSOR"/"INTERNAL"/"ACTION" strings) visits
    them in a different order every process launch -- this key function replaces any
    such hash-order iteration with one based only on fixed integer codes and integer
    ids, which is identical across every process and every PYTHONHASHSEED.
    """
    source_type, source_id, sink_type, sink_id = cid
    return (_NODE_TYPE_CODE[source_type], source_id, _NODE_TYPE_CODE[sink_type], sink_id)


def _canonical_order(genome):
    """Sort a genome's genes into the canonical connection order. Called at the end of
    crossover() and mutate() so genome gene order is always deterministic afterward --
    downstream build_network() and any serialization (telemetry, the determinism test)
    are then order-stable regardless of the hash-dependent order genes may have been
    assembled in along the way."""
    return sorted(genome, key=lambda g: connection_sort_key(connection_id(g)))


def _random_connection_locus(rng):
    source_type, sink_type = VALID_CONNECTIONS[rng.integers(len(VALID_CONNECTIONS))]
    source_id = int(rng.integers(N_SENSORS) if source_type == "SENSOR" else rng.integers(N_INTERNAL))
    sink_id = int(rng.integers(N_INTERNAL) if sink_type == "INTERNAL" else rng.integers(N_COINS))
    return source_type, source_id, sink_type, sink_id


def _random_gene(rng):
    source_type, source_id, sink_type, sink_id = _random_connection_locus(rng)
    weight = float(rng.uniform(WEIGHT_MIN, WEIGHT_MAX))
    return Gene(source_type, source_id, sink_type, sink_id, weight)


# --------------------------------------------------------------------------------------
# Crossover
# --------------------------------------------------------------------------------------
def crossover(parent_a, parent_b, fitness_a, fitness_b, rng):
    """Align parents by connection identity.
      - Shared connection (in both parents): child inherits the weight from a RANDOM
        parent (50/50).
      - Disjoint connection (in one parent only): always inherited from the FITTER
        parent; from the less-fit parent only with probability P_INHERIT_UNMATCHED.
    Child node skeleton is the fixed one (brain.py) -- only the connection set/weights
    recombine. Gene-count guards (MIN_GENES/MAX_GENES) are enforced afterward.
    """
    genes_a = {connection_id(g): g for g in parent_a}
    genes_b = {connection_id(g): g for g in parent_b}
    all_ids = set(genes_a) | set(genes_b)

    fitter, less_fit = (genes_a, genes_b) if fitness_a >= fitness_b else (genes_b, genes_a)

    child = []
    # deterministic order: iterate in canonical (hash-independent) order, not raw set
    # order -- a plain `for cid in all_ids` would consume rng.random() below in an order
    # that varies with Python's per-process string-hash randomization, silently breaking
    # "same master_seed -> same result" across process launches (see connection_sort_key).
    for cid in sorted(all_ids, key=connection_sort_key):
        if cid in genes_a and cid in genes_b:
            child.append(genes_a[cid] if rng.random() < 0.5 else genes_b[cid])
        elif cid in fitter:
            child.append(fitter[cid])
        elif rng.random() < P_INHERIT_UNMATCHED:
            child.append(less_fit[cid])

    while len(child) < MIN_GENES:
        child.append(_random_gene(rng))
    if len(child) > MAX_GENES:
        idx = rng.choice(len(child), size=MAX_GENES, replace=False)
        child = [child[i] for i in idx]

    # deterministic order: canonicalize the output gene order regardless of how it was
    # assembled above, so build_network()/serialization are order-stable downstream.
    return _canonical_order(child)


# --------------------------------------------------------------------------------------
# Mutation
# --------------------------------------------------------------------------------------
def mutate(genome, rng):
    """Apply, independently: per-gene weight perturbation (common) and reset (rare), then
    genome-level add-connection and remove-connection (both rare, at most one of each per
    call). Gene count stays within [MIN_GENES, MAX_GENES]. Node COUNTS are fixed in v1
    (no add-node -- brain growth is a later v2 feature).
    """
    new_genes = []
    for g in genome:
        w = g.weight
        if rng.random() < P_PERTURB:
            w = w + rng.normal(0, WEIGHT_MUT_SIGMA)
        if rng.random() < P_RESET:
            w = rng.uniform(WEIGHT_MIN, WEIGHT_MAX)
        w = float(np.clip(w, WEIGHT_MIN, WEIGHT_MAX))
        new_genes.append(g._replace(weight=w))
    genome = new_genes

    if rng.random() < P_ADD_CONN and len(genome) < MAX_GENES:
        # audited: this set is only used for `in` membership tests below (O(1) hash
        # lookup, order-independent) -- never iterated, so its hash-dependent internal
        # order doesn't affect which locus gets tried/added or consume rng in an
        # order-dependent way. Safe as-is.
        existing = {connection_id(g) for g in genome}
        for _ in range(10):  # a few tries to dodge an exact duplicate; skip quietly if unlucky
            locus = _random_connection_locus(rng)
            if locus not in existing:
                source_type, source_id, sink_type, sink_id = locus
                genome.append(Gene(source_type, source_id, sink_type, sink_id,
                                    float(rng.uniform(WEIGHT_MIN, WEIGHT_MAX))))
                break

    if rng.random() < P_REMOVE_CONN and len(genome) > MIN_GENES:
        idx = int(rng.integers(len(genome)))
        genome = genome[:idx] + genome[idx + 1:]

    # deterministic order: canonicalize before returning (mirrors crossover()) -- the
    # add-connection branch above appends at the end, which would otherwise leave the
    # gene list in an order that depends on how add/remove happened to interact.
    return _canonical_order(genome)


def validate_genome(genome, n_sensors=N_SENSORS, n_internal=N_INTERNAL, n_coins=N_COINS,
                     min_genes=MIN_GENES, max_genes=MAX_GENES):
    """Independent structural check (used by the smoke test, not by evolution itself):
    gene count within guards, every connection type allowed, every node id in range."""
    if not (min_genes <= len(genome) <= max_genes):
        return False, f"gene count {len(genome)} outside [{min_genes},{max_genes}]"
    for g in genome:
        if (g.source_type, g.sink_type) not in VALID_CONNECTIONS:
            return False, f"illegal connection type: {g}"
        max_source = n_sensors if g.source_type == "SENSOR" else n_internal
        if not (0 <= g.source_id < max_source):
            return False, f"source_id out of range: {g}"
        max_sink = n_internal if g.sink_type == "INTERNAL" else n_coins
        if not (0 <= g.sink_id < max_sink):
            return False, f"sink_id out of range: {g}"
    return True, "ok"


# --------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------
def tournament_select(population, fitnesses, rng, k=TOURNAMENT_K):
    idxs = rng.integers(0, len(population), size=k)
    winner = idxs[int(np.argmax(fitnesses[idxs]))]
    return population[winner], float(fitnesses[winner])
