#!/usr/bin/env python3
"""Verify code hashes, baseline coverage and optionally extracted data hashes."""
from pathlib import Path
import argparse,json,hashlib,csv,collections
R=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--data-root",type=Path);a=p.parse_args()
 for r in json.loads((R/"provenance/source_files.json").read_text()):assert sha(R/r["destination"])==r["sha256"],r["destination"]
 rows=json.loads((R/"baselines/records.json").read_text());assert len(rows)==354
 for r in rows:
  path=R/r["raw_result"];assert sha(path)==r["raw_sha256"];d=json.loads(path.read_text());c=d["config"];assert d["status"]=="complete";assert c["stage"]==r["stage"] and c["seed"]==r["seed"];assert d["metrics"]["test"]["r2"]==r["r2"]
 slots=list(csv.DictReader((R/"baselines/v3_120_slots.csv").open(encoding="utf-8-sig")));assert len(slots)==120;assert collections.Counter(r["status"] for r in slots)=={"reported":84,"not_run":24,"not_applicable":12}
 if a.data_root:
  root=a.data_root.resolve();manifest=list(csv.DictReader((root/"manifest.csv").open()));assert len(manifest)==1200;assert len({r["design"] for r in manifest})==30
  for row in manifest:
   for v in "BCDEF":assert (root/row[v]).is_file()

 print(json.dumps({"source_hashes":"passed","baseline_results":len(rows),"v3_reported":84,"v3_missing":24,"not_applicable":12,"dataset_checked":bool(a.data_root),"dataset_full_hash":False}))
if __name__=="__main__":main()
