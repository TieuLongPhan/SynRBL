import json
import subprocess
import collections
import ast
import pandas as pd

src_file = "./Data/Validation_set/validation_set.csv"
dataset = pd.read_csv(src_file).to_dict("records")
db_splits = collections.defaultdict(lambda: [])

for entry in dataset:
    edb = ast.literal_eval(entry["datasets"])
    db_splits[edb[0]].append(entry)

db_splits = dict(db_splits)
dataset_names = list(db_splits.keys())
data_files = []

for k, v in db_splits.items():
    file = "valset_{}.csv".format(k)
    data_files.append(file)
    df = pd.DataFrame(v)
    df.to_csv(file, index=False)

for f in data_files:
    run_cmd = ["python3", "-m", "synrbl", "run"]
    run_cmd.extend(["--out-columns", "expected_reaction"])
    run_cmd.extend(["--cache"])
    synrbl_p = subprocess.Popen(run_cmd + [f])
    rcode = synrbl_p.wait()
    if rcode != 0:
        raise RuntimeError("SynRBL returned with exit code {}".format(rcode))

benchmark_files = []
for ds in dataset_names:
    cmd = ["python3", "-m", "synrbl", "benchmark"]
    cmd.extend(["--target-col", "expected_reaction"])
    cmd.extend(["--min-confidence", "0"])
    benchmark_file = "valset_{}_benchmark.json".format(ds)
    benchmark_files.append(benchmark_file)
    cmd.extend(["-o", benchmark_file])
    file = "valset_{}_out.csv".format(ds)
    synrbl_p = subprocess.Popen(cmd + [file])
    rcode = synrbl_p.wait()
    if rcode != 0:
        raise RuntimeError("SynRBL benchmark returned with exit code {}".format(rcode))

benchmark_results = {}
for file, ds in zip(benchmark_files, dataset_names):
    with open(file, "r") as f:
        b_data = json.load(f)
    benchmark_results[ds] = b_data
with open("benchmark_result.json", "w") as f:
    json.dump(benchmark_results, f, indent=4)
