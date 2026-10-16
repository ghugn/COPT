"""Reference exact clique solver, not a competitive production solver.

Color bounds are recomputed on the remaining set at every iteration. This
deliberately avoids reusing prefix bounds after a learned reordering.
"""
import random
import time


def vertices(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def er_graph(n, density, seed):
    rng = random.Random(seed)
    adj = [0] * n
    for u in range(n):
        for v in range(u):
            if rng.random() < density:
                adj[u] |= 1 << v
                adj[v] |= 1 << u
    return adj


def is_clique(adj, clique):
    return len(set(clique)) == len(clique) and all(
        adj[u] & (1 << v) for i, u in enumerate(clique) for v in clique[i + 1:]
    )


def color_bound(adj, mask):
    """Number of independent color classes: valid bound for the whole mask."""
    remaining, count = mask, 0
    while remaining:
        count += 1
        available = remaining
        while available:
            bit = available & -available
            v = bit.bit_length() - 1
            remaining ^= bit
            available &= ~bit & ~adj[v]
    return count


def solve(adj, candidates=None, prefix=(), incumbent=(), priority=None,
          seconds=2.0, node_limit=200000, capture=0, capture_seed=0,
          capture_min=12):
    """Replay one subtree with a shared valid global incumbent.

    priority is applied only at the replay root; descendants use degree.
    exhausted means this subtree cannot improve the returned incumbent; it
    proves the whole graph optimum ONLY when the replay state is the root.
    """
    candidates = (1 << len(adj)) - 1 if candidates is None else candidates
    prefix, incumbent = tuple(prefix), tuple(incumbent)
    assert is_clique(adj, prefix) and is_clique(adj, incumbent)
    assert all(v not in prefix and all(adj[v] & (1 << u) for u in prefix)
               for v in vertices(candidates))
    start = time.perf_counter()
    best = incumbent if len(incumbent) >= len(prefix) else prefix
    nodes, seen = 0, 0
    snapshots, trace = [], [(0.0, len(best))]
    rng = random.Random(capture_seed)

    def expand(clique, mask, depth):
        nonlocal nodes, best, seen
        if time.perf_counter() - start >= seconds or nodes >= node_limit:
            raise TimeoutError
        nodes += 1
        if len(clique) > len(best):
            best = clique
            trace.append((time.perf_counter() - start, len(best)))
        if len(clique) + color_bound(adj, mask) <= len(best):
            return
        if capture and depth and mask.bit_count() >= capture_min:
            seen += 1
            state = {"prefix": list(clique), "candidates": mask,
                     "incumbent": list(best), "depth": len(clique)}
            if len(snapshots) < capture:
                snapshots.append(state)
            else:
                j = rng.randrange(seen)
                if j < capture:
                    snapshots[j] = state
        while mask:
            if time.perf_counter() - start >= seconds:
                raise TimeoutError
            if len(clique) + color_bound(adj, mask) <= len(best):
                return
            if depth == 0 and priority is not None:
                v = max(vertices(mask), key=lambda u: (priority[u], -u))
            else:
                v = max(vertices(mask), key=lambda u: ((adj[u] & mask).bit_count(), -u))
            expand(clique + (v,), mask & adj[v], depth + 1)
            mask &= ~(1 << v)

    exhausted = True
    try:
        expand(prefix, candidates, 0)
    except TimeoutError:
        exhausted = False
    elapsed = time.perf_counter() - start
    assert is_clique(adj, best)
    return {"best_size": len(best), "clique": list(best), "nodes": nodes,
            "exhausted": exhausted, "search_s": elapsed, "trace": trace,
            "snapshots": snapshots}


def solve_online(adj, scorer=None, gate=None, top_k=None,
                 seconds=5.0, node_limit=2000000):
    """
    Milestone M3: Bộ giải Online hoàn chỉnh tích hợp Cost-Aware Gate.
    Chạy tìm kiếm Maximum Clique từ đầu đến cuối trên đồ thị bất kỳ.
    - Tại nút Root: hỏi Gate xem có nên gọi COPT không.
    - Nếu gọi: tính priority từ COPT và phân nhánh trên Top-k (theo tư tưởng ASAP).
    - Toàn bộ thời gian overhead được tính sòng phẳng vào tổng thời gian.
    - Đảm bảo 100% tính chính xác nhờ Greedy Coloring Upper Bound.
    """
    start_total = time.perf_counter()
    candidates = (1 << len(adj)) - 1
    best = ()
    nodes = 0
    trace = [(0.0, 0)]
    neural_called = False
    overhead_s = 0.0

    priority = None
    if gate is not None and scorer is not None:
        feats = gate.extract_features(adj, candidates, clique_len=0, best_len=0)
        if gate.should_call(feats):
            scores, elapsed, err = scorer.score(adj, candidates)
            overhead_s = elapsed
            if err is None and scores is not None:
                priority = scores
                neural_called = True

    def expand(clique, mask, depth):
        nonlocal nodes, best
        if time.perf_counter() - start_total >= seconds or nodes >= node_limit:
            raise TimeoutError
        nodes += 1
        if len(clique) > len(best):
            best = clique
            trace.append((time.perf_counter() - start_total, len(best)))
        if len(clique) + color_bound(adj, mask) <= len(best):
            return

        branches_taken = 0
        while mask:
            if time.perf_counter() - start_total >= seconds:
                raise TimeoutError
            if len(clique) + color_bound(adj, mask) <= len(best):
                return

            if depth == 0 and priority is not None:
                if top_k is not None and branches_taken >= top_k:
                    v = max(vertices(mask), key=lambda u: ((adj[u] & mask).bit_count(), -u))
                else:
                    v = max(vertices(mask), key=lambda u: (priority[u], -u))
            else:
                v = max(vertices(mask), key=lambda u: ((adj[u] & mask).bit_count(), -u))

            branches_taken += 1
            expand(clique + (v,), mask & adj[v], depth + 1)
            mask &= ~(1 << v)

    exhausted = True
    try:
        expand((), candidates, 0)
    except TimeoutError:
        exhausted = False
    total_elapsed = time.perf_counter() - start_total
    assert is_clique(adj, best)

    return {
        "best_size": len(best),
        "clique": list(best),
        "nodes": nodes,
        "exhausted": exhausted,
        "total_s": total_elapsed,
        "overhead_s": overhead_s,
        "search_s": total_elapsed - overhead_s,
        "neural_called": neural_called,
        "trace": trace
    }

