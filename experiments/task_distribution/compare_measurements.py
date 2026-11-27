"""Compare held-out measured allocation counts with simulated run proportions.

python compare_measurements.py --measured real_allocation_counts.csv

Required measured columns: run,pro_precise_fraction,policy,site2_requests,
site2_precise_requests. Run IDs must identify genuinely paired workloads.
Does not generate measurements or silently fit model parameters.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    base=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measured",type=Path,required=True)
    p.add_argument("--simulated",type=Path,default=base/"output"/"allocation_runs.csv")
    p.add_argument("--out",type=Path,default=base/"output"/"hardware_comparison.json")
    a=p.parse_args()
    with a.simulated.open(newline="",encoding="utf-8-sig") as f:sim=list(csv.DictReader(f))
    lookup={(int(r["run"]),float(r["pro_precise_fraction"]),r["policy"]):r for r in sim}
    rows=[]
    with a.measured.open(newline="",encoding="utf-8-sig") as f:
        measured=list(csv.DictReader(f))
    for r in measured:
        key=(int(r["run"]),float(r["pro_precise_fraction"]),r["policy"])
        if key not in lookup:raise ValueError(f"Unmatched measured workload {key}")
        n,p2=int(r["site2_requests"]),int(r["site2_precise_requests"])
        if not 0<=p2<=n or n==0:raise ValueError("Invalid measured denominator")
        obs=100*p2/n;pred=float(lookup[key]["precise_at_site2_pct"])
        rows.append(dict(run=key[0],pro_precise_fraction=key[1],policy=key[2],
                         observed_pct=obs,predicted_pct=pred,error_pp=pred-obs))
    if not rows:raise ValueError("Measured input is empty")
    errors=np.array([r["error_pp"] for r in rows])
    result=dict(comparison="held-out paired allocation composition",points=len(rows),
                mae_pp=float(np.abs(errors).mean()),rmse_pp=float(np.sqrt(np.mean(errors**2))),
                bias_pp=float(errors.mean()),details=rows,
                note="This comparison does not validate TTFT, decode, batching or E2E latency.")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
