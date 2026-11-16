"""Tests for the dependency-criticality placement planner."""

import math

import networkx as nx
import pytest

from yafs.application import create_applications_from_json
from yafs.core import Sim
from yafs.distribution import deterministic_distribution
from yafs.path_routing import DeviceSpeedAwareRouting
from yafs.placement_criticality import (
    ChainRequest,
    CriticalityPlacement,
    CriticalityPlanner,
    allocation_definition,
    composite_centrality,
    flow_edge_entropy,
    node_scores,
    pagerank,
)
from yafs.topology import Topology


def star(n_leaves=4):
    g = nx.star_graph(n_leaves)
    nx.set_edge_attributes(g, 1, "PR")
    nx.set_edge_attributes(g, 1, "BW")
    nx.set_node_attributes(g, 100, "IPT")
    return g


def path(n=5):
    g = nx.path_graph(n)
    nx.set_edge_attributes(g, 1, "PR")
    nx.set_edge_attributes(g, 1, "BW")
    nx.set_node_attributes(g, 100, "IPT")
    return g


def request(app="a", users=(), sla=math.inf, instance=0):
    return ChainRequest(app=app, modules=[f"{app}_m"], latency_requirement=sla,
                        user_nodes=list(users), instructions=0.0, instance=instance)


def test_pagerank_matches_networkx():
    pytest.importorskip("scipy")  # networkx's own pagerank needs scipy
    for g in (star(), path(7), nx.binomial_tree(5), nx.karate_club_graph()):
        ours, ref = pagerank(g), nx.pagerank(g, alpha=0.85, weight=None)
        assert all(abs(ours[n] - ref[n]) < 1e-5 for n in g)


def test_centrality_peaks_at_the_hub():
    c = composite_centrality(star())
    assert c[0] == pytest.approx(1.0)
    assert all(c[leaf] < 1.0 for leaf in range(1, 5))


def test_flow_entropy_is_high_where_traffic_splits():
    g = star()
    h = flow_edge_entropy(g, sources=[1, 2, 3, 4], targets=[1, 2, 3, 4])
    assert h[0] == pytest.approx(1.0)  # traffic spread evenly over all four links
    assert all(h[leaf] == 0.0 for leaf in range(1, 5))  # one link only


def test_role_term_marks_gateways():
    g = star()
    g.nodes[0]["node_role"] = "control-plane"
    s = node_scores(g, eligible=list(g), demand_nodes=[1, 2])
    assert s[0]["R"] == 1.0 and s[1]["R"] == 0.0
    assert s[0]["score"] > s[1]["score"]


def test_least_central_avoids_the_hub():
    planner = CriticalityPlanner(star(), eligible=list(range(5)), rule="least_central")
    host = planner.choose(request(users=[1]))
    assert host != 0


def test_latency_picks_the_users_node():
    planner = CriticalityPlanner(path(), eligible=list(range(5)), rule="latency")
    assert planner.choose(request(users=[4])) == 4


def test_sla_filter_and_fallback():
    g = path()
    planner = CriticalityPlanner(g, eligible=list(range(5)), rule="least_central")
    # Only nodes within 1 hop of the user at node 2 meet the SLA.
    assert planner.choose(request(users=[2], sla=1.0)) in {1, 2, 3}
    # No node meets an impossible SLA: fall back to the lowest-delay host.
    assert planner.choose(request(users=[2], sla=-1.0, instance=1)) == 2


def test_cei_keeps_chains_off_the_hub():
    planner = CriticalityPlanner(star(), eligible=list(range(5)), rule="cei", demand_nodes=[1, 2, 3, 4])
    hosts = [planner.choose(request(app=f"a{i}", users=[1 + i % 4])) for i in range(4)]
    assert 0 not in hosts


def test_cei_spreads_load():
    planner = CriticalityPlanner(path(), eligible=list(range(5)), rule="cei", lam=0.0)
    hosts = [planner.choose(request(app=f"a{i}")) for i in range(4)]
    assert len(set(hosts)) == 4


def test_spread_rotates_domains():
    domains = {0: "x", 1: "x", 2: "y", 3: "y", 4: "z"}
    planner = CriticalityPlanner(path(), eligible=list(range(5)), rule="spread", domains=domains)
    picked = [domains[planner.choose(request(app=f"a{i}"))] for i in range(3)]
    assert sorted(picked) == ["x", "y", "z"]


def test_random_rule_repeats_with_its_seed():
    a = CriticalityPlanner(path(), eligible=list(range(5)), rule="random", seed=3)
    b = CriticalityPlanner(path(), eligible=list(range(5)), rule="random", seed=3)
    reqs = [request(app=f"a{i}") for i in range(6)]
    assert a.plan(reqs) == b.plan(reqs)


def test_one_tier_per_node_is_least_central_order():
    g = path(7)
    tiers = CriticalityPlanner(g, eligible=list(g), rule="cei", tiers=len(g))
    least = CriticalityPlanner(g, eligible=list(g), rule="least_central")
    assert tiers.choose(request()) == least.choose(request())


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError):
        CriticalityPlanner(path(), eligible=[0], rule="nope")


def test_allocation_definition_lists_every_module():
    payload = allocation_definition("app", ["m1", "m2"], "n1")
    assert [i["module_name"] for i in payload["initialAllocation"]] == ["m1", "m2"]
    assert {i["id_resource"] for i in payload["initialAllocation"]} == {"n1"}


def _single_module_app(name):
    return {
        "id": name, "name": name, "HwReqs": 1, "MaxReqs": 200, "MaxLatency": 50,
        "transmission": [{"message_in": f"M.{name}", "module": f"{name}_01"}],
        "module": [{"id": 1, "name": f"{name}_01", "type": "MODULE", "RAM": 1}],
        "message": [{"id": 0, "name": f"M.{name}", "s": "None", "d": f"{name}_01",
                     "bytes": 20, "instructions": 30}],
    }


def test_core_layer_deploys_on_the_planned_host(tmp_path):
    t = Topology()
    t.G = path()
    apps = create_applications_from_json([_single_module_app(0)])
    planner = CriticalityPlanner(t.G, eligible=list(t.G), rule="latency")
    plan = planner.plan([request(app=0, users=[4])])
    sim = Sim(t, default_results_path=str(tmp_path / "trace"))
    sim.deploy_app(apps[0], CriticalityPlacement(plan, name="crit"), DeviceSpeedAwareRouting())
    msg = apps[0].get_message("M.0")
    sim.deploy_source(0, id_node=4, msg=msg, distribution=deterministic_distribution(10, name="d"))
    sim.run(100)
    hosts = {sim.alloc_DES[des] for des in sim.alloc_module[0]["0_01"]}
    assert hosts == {4}


def test_replicas_of_one_app_never_share_a_node():
    planner = CriticalityPlanner(path(), eligible=list(range(5)), rule="latency")
    plan = planner.plan([request(app="a", users=[4], instance=i) for i in range(3)])
    assert len(set(plan.values())) == 3
    assert plan[("a", 0)] == 4  # the first replica still goes next to the user
