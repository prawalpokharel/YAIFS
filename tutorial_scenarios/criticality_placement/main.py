"""
    Placement under node failures, on the topology of ``01_basicExample``.

    The same apps and users run four times with different placements: the
    tutorial's JSON allocation (everything on node 0), the lowest delay to each
    app's users, the least central node, and the dependency-criticality planner. Halfway through, the two most
    central nodes fail. The script prints how many requests still succeed.

    Run from the repository root:

        uv run tutorial_scenarios/criticality_placement/main.py
"""
import json
import logging
import shutil
import tempfile
from pathlib import Path

import networkx as nx
import pandas as pd

from yafs.application import create_applications_from_json
from yafs.core import Sim
from yafs.distribution import deterministic_distribution, deterministicDistributionStartPoint
from yafs.path_routing import DeviceSpeedAwareRouting
from yafs.placement import JSONPlacement
from yafs.placement_criticality import ChainRequest, CriticalityPlacement, CriticalityPlanner
from yafs.topology import Topology

DATA = Path(__file__).resolve().parents[1] / "01_basicExample" / "data"
STOP, T_FAIL = 20000, 10000
RULES = {"lowest delay": "latency", "least central": "least_central", "dependency-criticality": "cei"}


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def topology():
    g = nx.generators.binomial_tree(5)
    nx.set_edge_attributes(g, 1, "PR")
    nx.set_edge_attributes(g, 1, "BW")
    nx.set_node_attributes(g, 100, "IPT")
    return g


class FailNodes:
    """DES monitor that removes some nodes once."""

    def __init__(self, nodes):
        self.nodes, self.done = nodes, False

    def __call__(self, sim, routing):
        if not self.done:
            self.done = True
            for node in self.nodes:
                sim.remove_node(node)
            routing.invalid_cache_value = True


def run(placement_name, folder):
    g = topology()
    users = load("usersDefinition.json")["sources"]
    app_defs = load("appDefinition.json")
    if placement_name == "tutorial JSON":
        placement = JSONPlacement(name="Placement", json=load("allocDefinition.json"))
    else:
        rule = RULES[placement_name]
        planner = CriticalityPlanner(g, list(g), rule=rule, demand_nodes=[u["id_resource"] for u in users])
        requests = [
            ChainRequest(app=a["name"], modules=[m["name"] for m in a["module"]],
                         latency_requirement=a["MaxLatency"],
                         user_nodes=[u["id_resource"] for u in users if u["app"] == a["name"]],
                         instructions=sum(m["instructions"] for m in a["message"]))
            for a in app_defs
        ]
        placement = CriticalityPlacement(planner.plan(requests), name="Placement")

    t = Topology()
    t.G = g
    apps = create_applications_from_json(app_defs)
    routing = DeviceSpeedAwareRouting()
    s = Sim(t, default_results_path=folder + "sim_trace")
    for name in apps:
        s.deploy_app(apps[name], placement, routing)
    for user in users:
        msg = s.apps[user["app"]].get_message(user["message"])
        s.deploy_source(user["app"], id_node=user["id_resource"], msg=msg,
                        distribution=deterministic_distribution(100, name="Deterministic"))
    central = sorted(nx.betweenness_centrality(g).items(), key=lambda kv: -kv[1])[:2]
    s.deploy_monitor("Failure", FailNodes([n for n, _ in central]),
                     deterministicDistributionStartPoint(T_FAIL, 10 * STOP, name="Failure"),
                     **{"sim": s, "routing": routing})
    s.run(STOP)

    # A request counts as served when its module processed it (a COMP_M row with the
    # request id). Replies to users are traced only as link traffic, so a reply from a
    # module on the user's own node would leave no record of its own.
    events = pd.read_csv(folder + "sim_trace.csv")
    sent = events[(events["type"] == "SRC_M") & (events["time_emit"] >= T_FAIL)]
    served = events[events["type"] == "COMP_M"]
    ok = set(zip(served["app"].astype(str), served["id"]))
    success = sum((str(a), i) in ok for a, i in zip(sent["app"], sent["id"])) / max(len(sent), 1)
    return success


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    for name in ("tutorial JSON", *RULES):
        folder = tempfile.mkdtemp() + "/"
        try:
            print(f"{name:>24}: {100 * run(name, folder):5.1f}% of requests succeed after the failure")
        finally:
            shutil.rmtree(folder, ignore_errors=True)
