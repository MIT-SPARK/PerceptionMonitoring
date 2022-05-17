from functools import partial
import itertools
from diagnosability.hyperdiagnosable import Hyperdiagnosable
from rich import print
import numpy as np
import networkx as nx
from ortools.linear_solver import pywraplp
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def bound(h_est, confidence, D, F):
    delta = 1 - confidence
    return h_est + F * np.sqrt(np.log(2 / delta) / (2 * D))


def diagnostic(n, scopes, syndrome):
    solver = pywraplp.Solver.CreateSolver("SCIP")
    varname = [f"x{i}" for i in range(n)]
    x = [solver.IntVar(0, 1, varname[i]) for i in range(n)]
    for scope, o in zip(scopes, syndrome):
        vv = [x[i] for i in scope]
        if o:
            # At leat one active failure mode
            solver.Add(sum(vv) >= 1)
        else:
            # All Equal
            solver.Add((len(vv) - 1) * vv[0] == sum(vv[1:]))
    solver.Minimize(sum(x))
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        print("Could not find optimal solution")
        return []
    return [int(x[i].solution_value()) for i in range(n)]


def random_k_diagnosable(num_nodes, target_diagnosability):
    # assert (
    #     num_nodes > 2 * target_diagnosability
    # ), "Not enough nodes to achieve target diagnosability"
    kappa = -1
    assert num_nodes*target_diagnosability % 2 == 0, "N*k must be even"
    while kappa != target_diagnosability:
        graph = nx.random_regular_graph(n=num_nodes, d=target_diagnosability)
        scopes = [list(e) for e in graph.edges()]
        tests = [partial(Hyperdiagnosable.WeakOR, s) for s in scopes]
        dsys = Hyperdiagnosable(nr_variables=num_nodes, tests=tests, constraint=[])
        kappa = dsys.kappa(False)
    if kappa < target_diagnosability:
        print("WARNING, kappa is smaller than target diagnosability")
    return dsys, scopes, kappa

confidence = 1 - (1e-12)

num_nodes = 10
diagnosability = 4
hist = {k: [] for k in range(num_nodes + 1)}
for _ in range(10):
    cache = dict()
    dsys, test_scopes, kappa = random_k_diagnosable(num_nodes, diagnosability)
    assert dsys is not None, "Could not find a diagnosable system"
    print(f"Found a {kappa}-diagnosable system with {len(test_scopes)} tests")
    states = list(dsys.states())
    for x in tqdm(states, "States"):
        for syn in dsys.syndromes(x):
            if syn in cache:
                f = cache[syn]
            else:
                f = diagnostic(num_nodes, test_scopes, syn)
                cache[syn] = f
            h = hamming(x, f)
            hist[sum(x)].append(h)
with open('temporary/hist_10.pickle', 'wb') as handle:
    pickle.dump(hist, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('temporary/hist.pickle', "rb") as handle:
#     hist = pickle.load(handle)
# num_nodes = len(hist)-1

avg = {k: np.mean(v) for k, v in hist.items()}
print(avg)
hamming_distances = np.array(list(itertools.chain(*hist.values())))
h_est = np.mean(hamming_distances)
h_max = np.max(hamming_distances)
print(f"Estimated Hamming Distance: {h_est}")
print(f"Max Hamming Distance: {h_max}")
print(f"Num samples: {len(hamming_distances)}")
bnd = np.ceil(bound(h_est, confidence, sum(len(x) for x in hist.values()), num_nodes))
print(f"h <= {bnd}")

with plt.style.context(["science"]):
    # fig, (ham, ham_distr) = plt.subplots(1, 2, figsize=(10, 3))
    fig = plt.figure(figsize=(10, 3))
    ham = plt.gca()
    # plt.rcParams.update({'font.size': 12})
    sns.barplot(ax=ham, x=list(avg.keys()), y=list(avg.values()), color="xkcd:steel blue")
    ham.hlines(y=[bnd], xmin=-0.5, xmax=num_nodes+0.5, colors='r', linestyles='--', lw=2)
    ham.vlines(x=[4.5], ymin=0, ymax=bnd, colors='r', linestyles='--', lw=2)
    ham.set_xticks(range(len(avg)+1))
    ham.minorticks_off()
    ham.set(xlabel="Number of Active Failure Modes", ylabel="Average Hamming Distance")
    # y = [
    #     len([x for x in hamming_distances if x <= i]) / len(hamming_distances) * 100
    #     for i in range(num_nodes + 1)
    # ]
    # sns.barplot(ax=ham_distr, x=list(range(num_nodes + 1)), y=y, color="xkcd:steel blue")
    # ham_distr.vlines(x=[bnd+0.5], ymin=0, ymax=100, colors='r', linestyles='--', lw=2)
    # ham_distr.hlines(y=[99.9], xmin=0, xmax=num_nodes, colors='xkcd:steel grey', linestyles='--', lw=1.2)

    fig.savefig(f"temporary/dummy_example.pdf")

# states_probabilities = np.array([p**sum(s) for s in states])
# states_probabilities /= sum(states_probabilities)

# fault_identifications = []
# for i in range(len(states)):
#     s = states[i]
#     fi = list(
#         itertools.chain(
#             *[dsys.fault_identification(syn) for syn in dsys.syndromes(s)]
#         )
#     )
#     h = max([hamming(x,s) for x in fi])
#     fault_identifications.append(h)
# print(fault_identifications)
