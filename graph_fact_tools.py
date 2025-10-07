from __future__ import annotations
from typing import List, Tuple
import numpy as np

Array = np.ndarray
MatrixList = List[List[Array]]

# ---------- Building block utilities ----------

def rows_with_two_ones(A: Array) -> int:
    return int(np.sum(np.sum(A, axis=1) == 2))

def cols_with_two_ones(A: Array) -> int:
    return int(np.sum(np.sum(A, axis=0) == 2))

def is_building_block(A: Array) -> bool:
    """
    'Building block' per the spec:
      • Across ALL rows and columns combined, there is at most ONE (row OR column) whose sum is exactly 2.
      • Every other row/column must have at most one '1'.
      • No row/column may have > 2 ones.
    """
    row_sums = np.sum(A, axis=1)
    col_sums = np.sum(A, axis=0)

    if np.any(row_sums > 2) or np.any(col_sums > 2):
        return False

    num_rows_eq2 = np.count_nonzero(row_sums == 2)
    num_cols_eq2 = np.count_nonzero(col_sums == 2)

    # At most one among all rows+cols has exactly two 1s
    if (num_rows_eq2 + num_cols_eq2) > 1:
        return False

    # All others must be <= 1 is implied by the >2 check + ==2 counting
    return True

def violating_cols(A: Array) -> List[int]:
    """Columns with more than two 1s."""
    return [int(j) for j, s in enumerate(np.sum(A, axis=0)) if s >= 2]

def violating_rows(A: Array) -> List[int]:
    """Rows with more than two 1s."""
    return [int(i) for i, s in enumerate(np.sum(A, axis=1)) if s >= 2]

# ---------- Split operations & E builders ----------

def split_column(A: Array, c: int, i: int) -> Tuple[Array, Array]:
    """
    Create A' by replacing column c with two columns:
      • col c': same as col c except entry (i,c') set to 0
      • col c'' : a standard-basis column e_i (only 1 at row i)

    Also build E of shape (n+1) x n by taking I_n and inserting a duplicate
    of the i-th ROW immediately after it (so a single column — column i — has two 1s).
    """
    m, n = A.shape
    A_prime = np.zeros((m, n + 1), dtype=int)

    # copy columns before c
    if c > 0:
        A_prime[:, :c] = A[:, :c]

    # new col c (same as old c but zero at row i)
    A_prime[:, c] = A[:, c]
    A_prime[i, c] = 0

    # new col c+1 is e_i
    A_prime[:, c + 1] = 0
    A_prime[i, c + 1] = 1

    # copy columns after c into shifted positions
    if c + 1 < n:
        A_prime[:, c + 2:] = A[:, c + 1:]

    # Build E: (n+1) x n identity with duplicated i-th row inserted just after row i
    E = np.eye(n, dtype=int)
    E = np.insert(E, c, E[c, :], axis=0)  # duplicate row i

    return A_prime

def split_row(A: Array, r: int, j: int) -> Tuple[Array, Array]:
    """
    Create A' by replacing row r with two rows:
      • row r': same as row r except entry (r', j) set to 0
      • row r'' : a standard-basis row e_j^T (only 1 at column j)

    Also build E of shape m x (m+1).
    To achieve m x (m+1), we duplicate the j-th COLUMN of I_m,
    giving a single ROW (the j-th) of E with two 1s (in columns j and j+1).
    """
    m, n = A.shape
    A_prime = np.zeros((m + 1, n), dtype=int)

    # copy rows before r
    if r > 0:
        A_prime[:r, :] = A[:r, :]

    # new row r (same as old r but zero at column j)
    A_prime[r, :] = A[r, :]
    A_prime[r, j] = 0

    # new row r+1 is e_j^T
    A_prime[r + 1, :] = 0
    A_prime[r + 1, j] = 1

    # copy rows after r into shifted positions
    if r + 1 < m:
        A_prime[r + 2:, :] = A[r + 1:, :]

    # Build E: m x (m+1) by duplicating column j of I_m
    E = np.eye(m, dtype=int)
    E = np.insert(E, r , E[:, r], axis=1)  # duplicate column j

    return A_prime

# --- NEW: isomorphism / dedup utilities ---
import networkx as nx

def chain_to_graph(chain):
    """
    Build a layered simple graph G from a chain [A1,...,Ak].
    Nodes: (layer, idx) where layer = 0..k, idx indexes rows/cols in that layer.
    Edges: for each 1 in A_i at (r,c), add edge between (i-1, r) and (i, c).
    Node attribute 'layer' is set so isomorphisms must preserve layers.
    """
    G = nx.Graph()
    k = len(chain)
    # Add nodes for each layer with 'layer' attribute
    # Layer 0 has size = rows of A1, layer i has size = cols of A_i = rows of A_{i+1}, etc.
    # We infer layer sizes from matrices
    layer_sizes = []
    # layer 0 size
    m0, n0 = chain[0].shape
    layer_sizes.append(m0)
    # intermediate layers 1..k-1
    for i in range(k-1):
        m, n = chain[i].shape
        layer_sizes.append(n)
    # layer k size
    mk_1, nk_1 = chain[-1].shape
    layer_sizes.append(nk_1)

    for layer, size in enumerate(layer_sizes):
        for idx in range(size):
            G.add_node((layer, idx), layer=layer)

    # Add edges for each matrix
    for i, A in enumerate(chain, start=1):
        # layer i-1 rows, layer i cols
        rows, cols = A.shape
        # sanity with inferred sizes
        assert rows == layer_sizes[i-1] and cols == layer_sizes[i], \
            "Chain has inconsistent shapes across layers."
        # add edges for ones
        rs, cs = A.nonzero()
        for r, c in zip(rs, cs):
            G.add_edge((i-1, int(r)), (i, int(c)))
    return G

def chain_wl_hash(chain):
    """
    Compute a Weisfeiler-Lehman graph hash that is invariant to within-layer permutations
    but respects the 'layer' attribute so layers cannot mix.
    """
    G = chain_to_graph(chain)
    # WL hash will incorporate node labels if provided via 'node_attr'
    return nx.weisfeiler_lehman_graph_hash(G, node_attr='layer')

def chains_isomorphic(chainA, chainB):
    """
    Exact isomorphism check with layer labels preserved.
    Useful only when wl-hashes collide; otherwise rely on hashes.
    """
    if len(chainA) != len(chainB):
        return False
    # Quick dimension check
    dimsA = [(A.shape[0], A.shape[1]) for A in chainA]
    dimsB = [(B.shape[0], B.shape[1]) for B in chainB]
    if dimsA != dimsB:
        # Different layer sizes => can still be isomorphic via permutations? No: sizes per layer must match.
        return False

    GA = chain_to_graph(chainA)
    GB = chain_to_graph(chainB)
    node_match = nx.algorithms.isomorphism.categorical_node_match('layer', None)
    GM = nx.algorithms.isomorphism.GraphMatcher(GA, GB, node_match=node_match)
    return GM.is_isomorphic()

def dedupe_chains(L):
    """
    L is a list of chains (each chain is a list of matrices).
    We bucket by WL hash; inside each bucket keep one representative, using exact
    isomorphism only to resolve same-hash collisions.
    Returns a new list with one representative per isomorphism class.
    """
    buckets = {}
    unique = []
    for chain in L:
        h = chain_wl_hash(chain)
        if h not in buckets:
            buckets[h] = [chain]
            unique.append(chain)
        else:
            # Check against existing reps in this bucket
            is_dup = False
            for rep in buckets[h]:
                if chains_isomorphic(chain, rep):
                    is_dup = True
                    break
            if not is_dup:
                buckets[h].append(chain)
                unique.append(chain)
    return unique


# ---------- Fact(L) recursion ----------

def all_building_blocks(L: MatrixList) -> bool:
    return all(is_building_block(A) for lst in L for A in lst)

def deep_copy_matrix_list(L: MatrixList) -> MatrixList:
    return [[A.copy() for A in lst] for lst in L]

def Fact(L: MatrixList) -> MatrixList:
    """
    Recursive procedure per the spec.
    L is a list of lists of matrices [ [A1,...,Ak], [B1,...,Bt], ... ],
    Each matrix is a numpy array with 0/1 entries.
    """
    # Base case
    if all_building_blocks(L):
        return L

    L1: MatrixList = []

    for lst in L:
        # If this whole list is already building blocks, keep it
        if all(is_building_block(A) for A in lst):
            L1.append([A.copy() for A in lst])
            continue

        # Otherwise, expand every non-building A_s in this list
        for s, A_s in enumerate(lst):
            if is_building_block(A_s):
                continue

            m, n = A_s.shape

            # --- Column branch: columns with > 2 ones ---
            for c in violating_cols(A_s):
                # for each i with A_s[i, c] != 0
                rows_with_one = np.nonzero(A_s[:, c] != 0)[0]
                # Build F: (n+1) x n identity with duplicated i-th row
                F = np.eye(n, dtype=int)
                F = np.insert(F, c, F[c, :], axis=0)  # duplicate row i
                for i in rows_with_one:
                    A_prime = split_column(A_s, c, int(i))
                    # Append {A_1,...,A_{s-1}, A'_s, E, A_{s+1},...,A_k}
                    new_list = [M.copy() for M in lst[:s]] + [A_prime, F] + [M.copy() for M in lst[s+1:]]
                    L1.append(new_list)

            # --- Row branch: rows with > 2 ones ---
            for r in violating_rows(A_s):
                # for each j with A_s[r, j] != 0
                cols_with_one = np.nonzero(A_s[r, :] != 0)[0]
                # Build E: m x (m+1) by duplicating column j of I_m
                E = np.eye(m, dtype=int)
                E = np.insert(E, r , E[:, r], axis=1)  # duplicate column j
                for j in cols_with_one:
                    A_prime = split_row(A_s, r, int(j))
                    # Append {A_1,...,A_{s-1}, E, A'_s, A_{s+1},...,A_k}
                    new_list = [M.copy() for M in lst[:s]] + [E, A_prime] + [M.copy() for M in lst[s+1:]]
                    L1.append(new_list)

            # We expand only the first non-building A_s in this list for this layer,
            # mirroring the “for each A_s in l that is not a building block” but
            # ensuring we don’t duplicate work within the same list multiple times.
            # If you instead want to expand *every* non-building A_s in this list
            # at this level, remove this 'break'.
            #break

    # if L1 == []:
    #     print('something went wrong')
    #     print(L[0])
    #     return L

    L1 = dedupe_chains(L1)

    # Recurse
    return Fact(L1)

# ---------- Convenience: validation helpers ----------

def assert_binary(A: Array):
    if not np.array_equal(A, (A != 0).astype(int)):
        raise ValueError("Matrix entries must be 0/1.")

def validate_L(L: MatrixList):
    for lst in L:
        for A in lst:
            if not isinstance(A, np.ndarray):
                raise TypeError("All matrices must be numpy arrays.")
            if A.ndim != 2:
                raise ValueError("All matrices must be 2D.")
            assert_binary(A)

# ---------- Public entry point ----------

def factorize(L: MatrixList) -> MatrixList:
    """
    Validates input and runs the recursive Fact(L).
    """
    validate_L(L)
    # Work on a deep copy to avoid mutating caller's data
    L_copy = deep_copy_matrix_list(L)
    return Fact(L_copy)
