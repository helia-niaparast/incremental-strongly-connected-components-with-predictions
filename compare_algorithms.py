from offline_alg import offline_incremental_scc
import learned_incSCC
import incSCCplus_optimized
import incSCCplus
import utils
import time
import random
import sys
import copy
import pandas as pd

sys.setrecursionlimit(5000) 

def run_offline_alg(n, edges):
      start = time.time()
      arr_edges = [[u,v] for (u,v) in edges]
      offline_incremental_scc(arr_edges, n)
      end = time.time()

      return end - start 

def run_incSCCplus(n, edges):
      start = time.time()
      adj_list = [[] for _ in range(n)]
      instance = incSCCplus.IncrementalSCC(adj_list)
      for u,v in edges:
            instance.add_new_edge(u,v) 
      end = time.time()
      
      return end - start 

def run_incSCCplus_optimized(n, edges):
      start = time.time()
      adj_list = [[] for _ in range(n)]
      instance = incSCCplus_optimized.IncrementalSCC(adj_list)
      for u,v in edges:
            instance.add_new_edge(u,v) 
      end = time.time()

      return end - start

def run_learned_incSCC(n, edges, perturb_function, params, reps, filename):
      times, average_etas, max_etas = [], [], []
      for _ in range(reps):
            predictions = utils.perturb_list(edges, perturb_function, params)
            utils.save_predictions(predictions, filename)
            average_eta, max_eta = utils.find_error(edges, predictions)
            average_etas.append(average_eta)
            max_etas.append(max_eta)

            start = time.time()
            instance = learned_incSCC.Online_Problem(n, edges, predictions)
            instance.run_algorithm()
            end = time.time()

            times.append(end - start)

      return times, average_etas, max_etas

datasets = ["sx-superuser-c2a", "sx-askubuntu", "Slashdot0811"]
dataset_name = datasets[0]
dataset = "Datasets/" + dataset_name + ".txt"

n, original_edges = utils.read_data(dataset)
m = len(original_edges)

with open("offline algorithm runtimes.txt", "a") as f:
    f.write(f"{dataset_name}: {run_offline_alg(n, original_edges)} \n")

reps = 10
df = pd.DataFrame(columns = ["algorithm", "runtime", "eta", "max_eta", "average_eta"])
df.to_csv(dataset_name + ", run data.csv", index = False)

incSCC_baseline = run_incSCCplus(n, copy.deepcopy(original_edges))
incSCC_optimized_baseline = run_incSCCplus_optimized(n, copy.deepcopy(original_edges))

learned_alg = {}
eta = 0
while True:
      edges = copy.deepcopy(original_edges)
      prediction_filename = dataset_name + ", predictions, eta = " + str(eta) + ".json"
      times, average_etas, max_etas = run_learned_incSCC(n, edges, random.gauss, (0,eta), reps, prediction_filename)
      average_time = utils.average(times)

      df = pd.DataFrame({
            "algorithm": ["incSCCPlus"] * reps + ["incSCCPlusEfficient"] * reps + ["learned_alg"] * reps,
            "runtime": [incSCC_baseline] * reps + [incSCC_optimized_baseline] * reps + times,
            "eta": [eta] * (3 * reps),
            "max_eta": max_etas + max_etas + max_etas,
            "average_eta": average_etas + average_etas + average_etas
      })
      df.to_csv(dataset_name + ", run data.csv", mode = 'a', header = False, index = False)

      if average_time > 2 * incSCC_optimized_baseline:
            break

      eta += 10

utils.plot_run_data(dataset_name + ", run data.csv", x_kind = "average")
utils.plot_run_data(dataset_name + ", run data.csv", x_kind = "max")