import re
import numpy as np 
import pandas as pd
import math
import random
import copy
import os
import seaborn as sns
import json
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

def average(lst):
      return sum(lst)/len(lst)

def read_data(file_location):
    # outputs edges in order of timestamps if they exist and in random order otherwise
    with open(file_location, 'r') as file:
        vertices, edges = set(), set()
        edges_with_timestamps = []
        has_timestamp = False
        for line in file:
            line = line.strip()
            
            edge_pattern = re.compile(r'^\s*(\d+)[,\s]+(\d+)(?:[,\s]+([\d.]+))?(?:[,\s]+(\d+))?\s*$')
            match = re.search(edge_pattern, line)
            
            if match:
                u = int(match.group(1))
                v = int(match.group(2))
                vertices.add(u)
                vertices.add(v)

                if match.group(3) != None:
                    has_timestamp = True
                    t = float(match.group(3))
                    if (u,v) not in edges:
                        edges_with_timestamps.append([t,u,v])

                edges.add((u,v))

    if has_timestamp:
        edges_with_timestamps.sort()
        edge_list = [(u,v) for _,u,v in edges_with_timestamps]
    else:
        edge_list = list(edges)
        random.shuffle(edge_list)

    vertex_map = {old_id: new_id for new_id, old_id in enumerate(sorted(vertices))}
    new_edges = [(vertex_map[u], vertex_map[v]) for u, v in edge_list]  

    return len(vertices), new_edges
    
def perturb_list(lst, error_function, error_function_args):
    # for each list element, picks a random index ahead and swaps with the current element 
    m = len(lst)
    perturbed_lst = copy.deepcopy(lst)
    swapped = [False] * m
    for i in range(m):
        rand = math.floor(error_function(*error_function_args))
        p1, p2 = i, i + rand
        if p2 < 0:
            p2 = 0
        elif p2 >= m:
            p2 = m-1

        if swapped[p1] == False and swapped[p2] == False:
            perturbed_lst[p1], perturbed_lst[p2] = perturbed_lst[p2], perturbed_lst[p1]
            swapped[p1], swapped[p2] = True, True

    return [(u,v) for u,v in perturbed_lst]

def find_error(edges, prediction):
    # returns average and max error
    # for edges in the prediction but not in the edge sequence, error = len(edges)
    m = len(edges)
    real_positions = {(u,v):i for i,(u,v) in enumerate(edges)} 
    predicted_positions = {(u,v):i for i,(u,v) in enumerate(prediction)}

    errors = []
    for u,v in edges:
        if (u,v) not in predicted_positions.keys():
            errors.append(m)
        else:
            errors.append(abs(real_positions[(u,v)] - predicted_positions[(u,v)]))
    
    return average(errors), max(errors)

def save_predictions(predictions, filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(predictions)

    with open(filename, "w") as f:
        json.dump(data, f)

def plot_run_data(filename, x_kind):
    df = pd.read_csv(filename)
    label_map = {
    "incSCCPlus": r'$\text{IncSCC}^+$',
    "incSCCPlusEfficient": r'$\text{IncSCC}^+$ (optimized)',
    "learned_alg": "Learned IncSCC"}

    agg = (df.groupby(["algorithm", "eta"], as_index = False)
             .agg(mean_runtime=("runtime", "mean"),
                  std_runtime = ("runtime", "std"),
                  x_avg_eta   = ("average_eta", "mean"),
                  x_max_eta   = ("max_eta", "mean"),
                  n           = ("runtime", "size")))     

    x_col = "x_avg_eta" if x_kind == "average" else "x_max_eta"
    x_label = r'$\eta_{\text{avg}}$' if x_kind == "average" else r'$\eta_{\text{max}}$'                                          

    fig, ax = plt.subplots()
    for algo, sub in agg.groupby("algorithm"):
        sub = sub.sort_values(x_col)
        ax.plot(sub[x_col], sub["mean_runtime"], marker = "o",  label = label_map[algo])
        ax.fill_between(sub[x_col],
                        sub["mean_runtime"] - sub["std_runtime"],
                        sub["mean_runtime"] + sub["std_runtime"],
                        alpha = 0.2)
    fontsize = 16
    ax.set_xlabel(x_label, fontsize = fontsize)
    ax.set_ylabel("runtime (s)", fontsize = fontsize)
    ax.tick_params(axis = 'both', which = 'major', labelsize = fontsize)
    ax.legend(fontsize = 12)
    ax.grid(True, which = "both", linestyle = "--", alpha = 0.6)
    plt.tight_layout()

    suffix = f"_{x_kind}_eta.pdf"
    if filename.lower().endswith(".csv"):
        outname = filename[:-4] + suffix
    else:
        outname = filename + suffix
    plt.savefig(outname)
    plt.close(fig)
