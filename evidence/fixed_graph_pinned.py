"""Independent float32 k-reciprocal comparator; no labels enter this function."""
import numpy as np

K1 = 20
K2 = 6
LAMBDA = .3

def assemble_owned(blocks):
    """Consume CPU score blocks, retaining no feature data or disk cache."""
    qg = blocks.pop('qg'); qq = blocks.pop('qq'); gg = blocks.pop('gg')
    nq, ng = qg.shape
    assert qq.shape == (nq, nq) and gg.shape == (ng, ng)
    assert all(x.dtype == np.float32 and np.isfinite(x).all() and np.min(x) >= 0 for x in [qg, qq, gg])
    matrix = np.empty((nq + ng, nq + ng), dtype=np.float32)
    matrix[:nq, :nq] = qq; matrix[:nq, nq:] = qg
    matrix[nq:, :nq] = qg.T; matrix[nq:, nq:] = gg
    return matrix

def rerank_owned(full, query_count, *, k1=K1, k2=K2, blend=LAMBDA, sort_block=64):
    """Consume an owned full distance array; use only exact firstK ranks.

    Returns final scores, normalization-only scores, and query neighborhood indices.
    Give a temporary/owned array if full-matrix lifetime must end inside the call.
    Every numerical operator follows the float32 distance-matrix convention; the
    only memory changes are batched full sorting and copying only the query rows.
    """
    assert full.dtype == np.float32 and full.ndim == 2 and full.shape[0] == full.shape[1]
    n = len(full); assert 0 < query_count < n and 0 <= blend <= 1
    assert 1 <= k2 <= k1 + 1 <= n and sort_block >= 1
    np.power(full, 2, out=full)
    scale = np.max(full, axis=0)
    assert np.isfinite(scale).all() and np.min(scale) > 0
    np.divide(full, scale[None, :], out=full)
    distance = full.T
    del full
    rank = np.empty((n, max(k1 + 1, k2)), dtype=np.int32)
    for start in range(0, n, sort_block):
        # Full quicksort, exactly as the author; truncation occurs afterwards.
        rank[start:start + sort_block] = np.argsort(distance[start:start + sort_block], axis=1)[:, :rank.shape[1]].astype(np.int32)
    weights = np.zeros((n, n), dtype=np.float32)
    half = int(np.around(k1 / 2.)) + 1
    query_context = []
    for node in range(n):
        forward = rank[node, :k1 + 1]
        reverse = rank[forward, :k1 + 1]
        reciprocal = forward[np.where(reverse == node)[0]]
        expanded = reciprocal.copy()
        for neighbor in reciprocal:
            candidate = rank[neighbor, :half]
            candidate_reverse = rank[candidate, :half]
            reciprocal_candidate = candidate[np.where(candidate_reverse == neighbor)[0]]
            if len(np.intersect1d(reciprocal_candidate, reciprocal)) > (2. / 3) * len(reciprocal_candidate):
                expanded = np.append(expanded, reciprocal_candidate)
        expanded = np.unique(expanded)
        affinity = np.exp(-distance[node, expanded])
        weights[node, expanded] = affinity / np.sum(affinity)
        if node < query_count:
            query_context.append(dict(reciprocal=reciprocal.tolist(), expanded=expanded.tolist()))
    query_distance = distance[:query_count].copy()
    del distance
    if k2 != 1:
        averaged = np.zeros_like(weights, dtype=np.float32)
        for node in range(n):
            averaged[node] = np.mean(weights[rank[node, :k2]], axis=0)
        weights = averaged
        del averaged
    del rank
    reverse_index = [np.where(weights[:, column] != 0)[0] for column in range(n)]
    jaccard = np.zeros_like(query_distance, dtype=np.float32)
    for node in range(query_count):
        overlap = np.zeros((1, n), dtype=np.float32)
        active = np.where(weights[node] != 0)[0]
        for column in active:
            images = reverse_index[column]
            overlap[0, images] = overlap[0, images] + np.minimum(weights[node, column], weights[images, column])
        jaccard[node] = 1 - overlap / (2. - overlap)
    scores = jaccard * (1 - blend) + query_distance * blend
    return scores[:, query_count:], query_distance[:, query_count:], query_context
