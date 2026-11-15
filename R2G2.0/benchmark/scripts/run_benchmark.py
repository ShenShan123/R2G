#!/usr/bin/env python3
"""Generate or execute the frozen v3 benchmark; defaults to a dry run."""
from pathlib import Path
import argparse,json,sys,subprocess,os,csv,shlex
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data-root",type=Path,default=ROOT/"datasets/R2G_multiview_v3_dataset");p.add_argument("--output-dir",type=Path,default=ROOT/"runs");p.add_argument("--stage",choices=["placement","route"]);p.add_argument("--task",choices=["wirelength","congestion"]);p.add_argument("--model",choices=["gat","gine","resgatedgcn","han","hgt","rgcn"]);p.add_argument("--view",choices=list("BCDEF"));p.add_argument("--device",default="cpu");p.add_argument("--python",default=sys.executable);p.add_argument("--execute",action="store_true");p.add_argument("--include-unreported",action="store_true",help="Allow the 24 v3 placement congestion configurations without historical metrics");p.add_argument("--limit",type=int);a=p.parse_args()
 plan=json.loads((ROOT/"configs/benchmark_v3_L4H3_100_seed42.json").read_text());jobs=[j for j in plan["jobs"] if all(getattr(a,k) is None or getattr(a,k)==j["config"][k] for k in ["stage","task","model","view"])]
 if not a.include_unreported:jobs=[j for j in jobs if j["reference_result"]]
 if a.limit is not None:jobs=jobs[:a.limit]
 if not jobs:p.error("No supported recorded job matches. E/congestion is N/A; placement/congestion requires --include-unreported.")
 data=a.data_root.expanduser().resolve();output=a.output_dir.expanduser().resolve();seen=set()
 for j in jobs:
  cfg=j["config"];runtime=j["runtime"];project=j["project"]
  entry=ROOT/"models"/project/"main.py";root=data
  if j.get("historical_reference_only"):
   print("NOTE: unified six-model code reruns this historical placement recipe; exact reproduction of the removed frozen source is not claimed.",file=sys.stderr)
  cmd=[a.python,str(entry),"--data-root",str(root),"--output-dir",str(output),"--run-name",j["id"],"--device",a.device]
  for k,v in cfg.items():
   if v is None:continue
   flag="--"+k.replace("_","-")
   if isinstance(v,bool):
    if v:cmd.append(flag)
   else:cmd +=[flag,str(v)]
  assert j["id"] not in seen;seen.add(j["id"]);print(shlex.join(cmd),flush=True)
  if a.execute:
   if not (data/"manifest.csv").is_file():p.error("Pass --data-root pointing to the external dataset containing manifest.csv and samples/.")
   if (output/j["id"]).exists():p.error("Run directory already exists; use another output directory. Existing results are never overwritten.")
   env=os.environ.copy();env.setdefault("OMP_NUM_THREADS","4");env.setdefault("MKL_NUM_THREADS","4");env.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
   subprocess.run(cmd,env=env,cwd=ROOT,check=True)
 print(json.dumps({"jobs":len(jobs),"executed":a.execute}))
if __name__=="__main__":main()
