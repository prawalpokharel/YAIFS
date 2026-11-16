"""
Dependency-criticality placement for YAIFS.

The policy decides where each application instance runs so that the loss of a
node or a cluster breaks as little as possible. Every node gets a criticality
score from three signals:

* ``C``: composite structural centrality on the topology graph
  (0.35 PageRank + 0.30 betweenness + 0.20 degree + 0.15 closeness, divided by
  the largest value);
* ``H``: flow-edge entropy, the normalised Shannon entropy of the node's traffic
  over its incident links when every expected user location sends to every
  eligible node along shortest paths;
* ``R``: a declared role, 1 for control-plane (gateway) nodes and 0 otherwise.

The score is ``alpha*C + beta*H + gamma*R``. Hosts are then chosen with the
objective of the Centrality-Entropy Index (CEI) placement study::

    J = (1 - lam) * (H(p) / log n - g * Phi) - lam * E

where ``p`` is the load share of each eligible node, ``Phi`` the
score-weighted load share and ``E`` the share of node power switched on.

The same planner also implements simpler rules (random, lowest delay, least
central, spread over failure domains), so every rule shares one delay filter
and one "chain as a unit" step and differs only in how it picks a host.

The method is described in P. Pokharel, "Self-Tuning Governance of the
Energy-Resilience Tradeoff in GPU Cluster Scheduling", IEEE IEMCON 2026.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import networkx as nx

from yafs.placement import Placement
from yafs.topology import Topology

RULES = ("random", "latency", "least_central", "spread", "cei")
DEFAULT_WEIGHTS = {"alpha": 0.40, "beta": 0.35, "gamma": 0.25}


@dataclass
class ChainRequest:
    """One application instance to place: its modules travel together."""

    app: object
    modules: list
    latency_requirement: float = math.inf
    user_nodes: list = field(default_factory=list)
    entry_bytes: float = 0.0
    instructions: float = 0.0
    load: float = 1.0
    instance: int = 0


def pagerank(graph, alpha=0.85, max_iter=100, tol=1.0e-6):
    """PageRank by power iteration on an undirected, unweighted graph.

    Same result as ``networkx.pagerank(graph, alpha)``, without its scipy
    dependency, which YAIFS does not otherwise need.
    """
    nodes = list(graph)
    n = len(nodes)
    rank = {v: 1.0 / n for v in nodes}
    degree = {v: graph.degree(v) for v in nodes}
    for _ in range(max_iter):
        dangling = alpha * sum(rank[v] for v in nodes if degree[v] == 0) / n
        new = {v: (1.0 - alpha) / n + dangling for v in nodes}
        for v in nodes:
            if degree[v]:
                share = alpha * rank[v] / degree[v]
                for u in graph.neighbors(v):
                    new[u] += share
        if sum(abs(new[v] - rank[v]) for v in nodes) < n * tol:
            return new
        rank = new
    return rank


def composite_centrality(graph):
    """Weighted PageRank, betweenness, degree and closeness, scaled to [0, 1]."""
    if graph.number_of_nodes() == 0:
        return {}
    pr = pagerank(graph, alpha=0.85)
    betweenness = nx.betweenness_centrality(graph)
    degree = nx.degree_centrality(graph)
    closeness = nx.closeness_centrality(graph)
    raw = {
        n: 0.35 * pr[n] + 0.30 * betweenness[n] + 0.20 * degree[n] + 0.15 * closeness[n]
        for n in graph
    }
    top = max(raw.values())
    return {n: (value / top if top > 0 else 0.0) for n, value in raw.items()}


def flow_edge_entropy(graph, sources, targets):
    """Normalised entropy of each node's traffic over its incident links."""
    sources = [s for s in dict.fromkeys(sources) if s in graph]
    targets = [t for t in dict.fromkeys(targets) if t in graph]
    traffic = {}
    for src in sources:
        paths = nx.single_source_shortest_path(graph, src)
        for dst in targets:
            path = paths.get(dst)
            if not path or len(path) < 2:
                continue
            for a, b in zip(path, path[1:]):
                edge = (a, b) if str(a) <= str(b) else (b, a)
                traffic[edge] = traffic.get(edge, 0) + 1
    entropy = {}
    for node in graph:
        loads = [
            traffic.get((node, nb) if str(node) <= str(nb) else (nb, node), 0)
            for nb in graph.neighbors(node)
        ]
        loads = [x for x in loads if x > 0]
        total = sum(loads)
        if len(loads) < 2 or total == 0:
            entropy[node] = 0.0
            continue
        h = -sum((x / total) * math.log(x / total) for x in loads)
        entropy[node] = h / math.log(graph.degree(node))
    return entropy


def node_scores(graph, eligible, demand_nodes, weights=None):
    """Score every node with alpha*C + beta*H + gamma*R and keep the parts."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    cen = composite_centrality(graph)
    ent = flow_edge_entropy(graph, demand_nodes, eligible)
    scores = {}
    for node, attrs in graph.nodes(data=True):
        role = 1.0 if attrs.get("node_role") == "control-plane" else 0.0
        c, h = cen.get(node, 0.0), ent.get(node, 0.0)
        scores[node] = {
            "C": c,
            "H": h,
            "R": role,
            "score": w["alpha"] * c + w["beta"] * h + w["gamma"] * role,
        }
    return scores


def estimate_delay(graph, request, host, distances):
    """Mean one-way delay from the request's users to a host, plus processing."""
    users = request.user_nodes or [host]
    total = 0.0
    for user in users:
        total += distances.get(user, {}).get(host, math.inf)
    ipt = float(graph.nodes[host].get(Topology.NODE_IPT, 1.0) or 1.0)
    return total / len(users) + request.instructions / ipt


def _objective(loads, scores, watts, eligible, lam, g):
    total = sum(loads.values())
    n = len(eligible)
    if total <= 0 or n < 2:
        return 0.0
    shares = [loads[node] / total for node in eligible if loads[node] > 0]
    h = -sum(p * math.log(p) for p in shares) / math.log(n)
    top = max(scores[node]["score"] for node in eligible) or 1.0
    phi = sum(scores[node]["score"] * loads[node] / total for node in eligible) / top
    energy = sum(watts[node] for node in eligible if loads[node] > 0) / sum(watts[n_] for n_ in eligible)
    return (1 - lam) * (h - g * phi) - lam * energy


class CriticalityPlanner:
    """Plan the hosts of a list of chain requests with one of ``RULES``.

    Args:
        graph: the topology graph (``sim.topology.G``).
        eligible: nodes that may host services.
        rule: one of ``RULES``.
        weights: ``alpha``, ``beta``, ``gamma`` of the node score.
        lam, g: weights of the placement objective.
        tiers: when above 0, split eligible nodes into this many centrality
            tiers and use the least central tier that can host the chain.
        domains: node to failure-domain map, used by ``spread``.
        demand_nodes: expected user locations, used by the entropy signal.
        seed: seed of the ``random`` rule.
    """

    def __init__(self, graph, eligible, rule="cei", weights=None, lam=0.35, g=0.5,
                 tiers=0, domains=None, demand_nodes=None, seed=0):
        if rule not in RULES:
            raise ValueError(f"Unknown rule {rule!r}; use one of {RULES}")
        self.graph = graph
        self.eligible = sorted(eligible, key=str)
        if not self.eligible:
            raise ValueError("No eligible node to place services on")
        self.rule = rule
        self.lam, self.g, self.tiers = lam, g, int(tiers)
        self.domains = domains or {node: node for node in self.eligible}
        self.rng = random.Random(seed)
        demand = demand_nodes if demand_nodes else self.eligible
        self.scores = node_scores(graph, self.eligible, demand, weights)
        self.watts = {n: float(graph.nodes[n].get("WATT", 1.0) or 1.0) for n in self.eligible}
        self.loads = {n: 0.0 for n in self.eligible}
        self.distances = dict(nx.all_pairs_dijkstra_path_length(graph, weight=Topology.LINK_PR))
        self._domain_cursor = 0

    def _feasible(self, request, forbidden=()):
        candidates = [h for h in self.eligible if h not in set(forbidden)] or list(self.eligible)
        delays = {h: estimate_delay(self.graph, request, h, self.distances) for h in candidates}
        ok = [h for h in candidates if delays[h] <= request.latency_requirement]
        if not ok:
            best = min(delays.values())
            ok = [h for h in candidates if delays[h] == best]
        return ok, delays

    def _tier_filter(self, hosts):
        if self.tiers <= 0:
            return hosts
        ranked = sorted(self.eligible, key=lambda n: (self.scores[n]["C"], str(n)))
        size = math.ceil(len(ranked) / self.tiers)
        for start in range(0, len(ranked), size):
            band = set(ranked[start:start + size])
            inside = [h for h in hosts if h in band]
            if inside:
                return inside
        return hosts

    def choose(self, request, forbidden=()):
        """Pick a host for ``request``; ``forbidden`` hosts (for example nodes that
        already run a replica of the same app) are skipped while others remain."""
        hosts, delays = self._feasible(request, forbidden)
        hosts = self._tier_filter(hosts)
        key_load = lambda h: (self.loads[h], str(h))  # noqa: E731
        if self.rule == "random":
            host = self.rng.choice(hosts)
        elif self.rule == "latency":
            host = min(hosts, key=lambda h: (delays[h], self.loads[h], str(h)))
        elif self.rule == "least_central":
            host = min(hosts, key=lambda h: (self.scores[h]["C"], self.loads[h], str(h)))
        elif self.rule == "spread":
            order = sorted({self.domains[h] for h in hosts}, key=str)
            domain = order[self._domain_cursor % len(order)]
            self._domain_cursor += 1
            host = min((h for h in hosts if self.domains[h] == domain), key=key_load)
        else:
            def gain(h):
                self.loads[h] += request.load
                value = _objective(self.loads, self.scores, self.watts, self.eligible, self.lam, self.g)
                self.loads[h] -= request.load
                return value
            # Near-ties go to the lower delay, then the node name, so runs repeat exactly.
            host = max(hosts, key=lambda h: (round(gain(h), 12), -delays[h], str(h)))
        self.loads[host] += request.load
        return host

    def plan(self, requests):
        """Return ``{(app, instance): host}`` in request order.

        Replicas of one app never share a node while another node is eligible."""
        out, used = {}, {}
        for r in requests:
            host = self.choose(r, forbidden=used.get(r.app, ()))
            used.setdefault(r.app, set()).add(host)
            out[(r.app, r.instance)] = host
        return out


def allocation_definition(app, modules, host):
    """``initialAllocation`` payload for ``Simulation.deploy_application_vnfs``."""
    return {
        "initialAllocation": [
            {"app": app, "module_name": module, "id_resource": host} for module in modules
        ]
    }


class CriticalityPlacement(Placement):
    """Core-layer placement that deploys every module of an app on one planned host.

    ``plan`` maps ``(app_name, instance)`` to a node, as returned by
    :meth:`CriticalityPlanner.plan`. Each app gets one instance per planned key.
    """

    def __init__(self, plan, **kwargs):
        super().__init__(**kwargs)
        self.plan = plan

    def initial_allocation(self, sim, app_name):
        app = sim.apps[app_name]
        for (planned_app, _instance), host in self.plan.items():
            if planned_app != app_name:
                continue
            for module, service in app.services.items():
                sim.deploy_module(app_name, module, service, [host])
