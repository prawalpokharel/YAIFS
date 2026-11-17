Placement under node failures
=============================

One way to run the code:

```
uv run tutorial_scenarios/criticality_placement/main.py
```

The script reuses the topology, applications and users of `01_basicExample`
(a binomial tree of 32 nodes, seven single-module apps and four users) and runs
the same simulation four times with a different placement each time:

- `tutorial JSON` uses `01_basicExample/data/allocDefinition.json`, which puts
  every module on node 0;
- `lowest delay` puts each app on the node closest to its users;
- `least central` puts each app on the eligible node with the lowest structural
  centrality;
- `dependency-criticality` uses `yafs.placement_criticality.CriticalityPlanner`
  with its default rule.

Halfway through (t = 10,000) the two nodes with the highest betweenness fail,
through `Sim.remove_node`. The script prints the share of requests sent after
the failure that the app's module still processes (`COMP_M` rows in the trace).

How the planner chooses a host
------------------------------

`CriticalityPlanner` places every module of an application instance on one node
(a chain moves as a unit), keeps only nodes that meet the app's latency
requirement, and never puts two replicas of one app on the same node while
another node is eligible. The rule then picks the host:

| Rule | Host choice |
|---|---|
| `random` | a random eligible node |
| `latency` | the lowest estimated delay to the app's users |
| `least_central` | the lowest composite centrality |
| `spread` | round-robin over failure domains (for example clusters) |
| `cei` | the best value of `(1 - lam) * (H(p)/log n - g * Phi) - lam * E` |

For `cei`, each node is scored with `alpha*C + beta*H + gamma*R`: structural
centrality, flow-edge entropy and a declared role (1 for control-plane nodes).
`p` is the load share of each node, `Phi` the score-weighted load share and `E`
the share of node power switched on. `tiers=k` restricts each choice to the
least central of `k` centrality tiers that can host the app.

Files
-----

- `main.py` builds the four runs and prints the comparison.
- The planner is in `src/yafs/placement_criticality.py` and its tests are in
  `tests/test_criticality_placement.py`.
