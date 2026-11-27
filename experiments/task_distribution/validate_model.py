"""Meaningful deterministic queue checks and independent CSV re-aggregation.
Run after simulation: python validate_model.py
"""
from dataclasses import replace
import csv
import json
from pathlib import Path
import numpy as np
from simulate_allocation import CONFIG, Engine, make_arrivals, make_classes


def exact_trace(times):
    a=np.array(times,dtype=float)
    return dict(times=a,clients=np.ones(len(a),dtype=int),
                measured=np.ones(len(a),dtype=bool),ranks=np.arange(len(a)),uniform=np.zeros(len(a)))


def test_engine():
    c=replace(CONFIG,warmup_s=0.,measurement_s=30.,workstations_per_site=(1,1),
        per_workstation_prefill_tps=(400.,400.),input_bytes_per_token=0.,
        request_overhead_bytes=0,response_bytes=0,report_delay_s=.2)
    tr=exact_trace([0.,0.,0.])
    e=Engine(c,tr,np.ones(3,dtype=int),"L3GSA")
    r=e.run()
    assert np.allclose([x["service_start_s"] for x in r],[0,1,2])
    assert np.allclose([x["prefill_finish_s"] for x in r],[1,2,3])

    tr=exact_trace([0.]*20)
    e=Engine(c,tr,np.ones(20,dtype=int),"L3GSA");r=e.run()
    sample=next(x for x in e.snapshot_rows if x["sample_s"]==11 and x["site"]==1)
    assert abs(sample["outstanding_input_tokens"]-3600)<1e-7
    assert r[-1]["prefill_finish_s"]==20

    tr=exact_trace([0.]*8)
    e=Engine(replace(c,workstations_per_site=(4,4)),tr,np.ones(8,dtype=int),"L3GSA")
    r=e.run()
    assert sorted(x["prefill_finish_s"] for x in r)==[1]*4+[2]*4

    tr=exact_trace([0.])
    e=Engine(c,tr,np.ones(1,dtype=int),"L3GSA",
             [dict(time_s=.5,site=1,worker=1,prefill_tps=200)])
    assert abs(e.run()[0]["prefill_finish_s"]-1.5)<1e-8

    # Full-load throughput closes to the configured workstation token rate.
    tr=exact_trace([0.]*100)
    e=Engine(c,tr,np.ones(100,dtype=int),"L3GSA");r=e.run()
    assert abs(100*c.input_tokens/r[-1]["prefill_finish_s"]-400)<1e-8
    assert len(r)==100 and r[-1]["prefill_finish_s"]>c.measurement_s

    tr=exact_trace([.3,1.3])
    e=Engine(c,tr,np.ones(2,dtype=int),"L3GSA");r=e.run()
    assert np.allclose([x["route_report_age_s"] for x in r],[.3,.3])

    tr=make_arrivals(replace(c,offered_rps=3),1234)
    kinds=make_classes(c,tr,.2)
    r1=Engine(c,tr,kinds,"L3GSA").run()
    r2=Engine(c,tr,kinds,"L3GSA").run()
    assert json.dumps(r1,sort_keys=True)==json.dumps(r2,sort_keys=True)
    return ["exact single-server FIFO", "backlog persists beyond 10-second metric window",
            "four parallel workstations", "measured capacity step changes completion correctly",
            "saturated token-work rate and complete cohort drain",
            "report causality and delay", "seeded lifecycle determinism"]


def test_exported(base):
    with (base/"allocation_runs.csv").open() as f:
        runs=list(csv.DictReader(f))
    counts={}
    with (base/"requests.csv").open() as f:
        for r in csv.DictReader(f):
            if int(r["measurement"])!=1:continue
            key=(int(r["run"]),float(r["pro_precise_fraction"]),r["policy"])
            c=counts.setdefault(key,[0,0,0,0])
            c[0]+=1
            if int(r["assigned_site"])==2:
                c[1]+=1;c[2]+=int(r["kind"])==2
            c[3]+=int(r["semantic_match"])==0
            a,n,s,e=[float(r[k]) for k in ("arrival_s","network_arrival_s","service_start_s","prefill_finish_s")]
            assert a<=n<=s+1e-8 and s<e
            assert abs(float(r["queue_wait_s"])-(s-n))<1e-8
    for r in runs:
        key=(int(r["run"]),float(r["pro_precise_fraction"]),r["policy"])
        n,n2,p,wrong=counts[key]
        assert n==int(r["requests"]) and n2==int(r["site2_requests"]) and p==int(r["site2_precise_requests"])
        assert abs(100*p/n2-float(r["precise_at_site2_pct"]))<1e-9
        if r["policy"]!="Dyncast-task":assert wrong==0
    with (base/"plot_data.csv").open() as f:
        plot=list(csv.DictReader(f))
    for p in plot:
        x=float(p["precise_task_rate_rps"])
        for policy in ("L3GSA","Dyncast-task","CoAP-static"):
            values=[float(r["precise_at_site2_pct"]) for r in runs
                    if r["policy"]==policy and np.isclose(float(r["target_precise_rps"]),x)]
            for label,value in (("mean",np.mean(values)),("min",min(values)),("max",max(values))):
                assert abs(float(p[policy+"_"+label])-value)<1e-9
    # Same offered sequence and per-condition count across all paired policies.
    for rep in sorted({int(r["run"]) for r in runs}):
        assert len({r["requests"] for r in runs if int(r["run"])==rep})==1
    return ["independent request-log re-aggregation", "plot means/min-max match run ratios",
            "paired offered counts", "no semantic violations by constrained policies"]


def main():
    base=Path(__file__).resolve().parent
    checks=test_engine()+test_exported(base/"output")
    result=dict(status="passed",checks=checks,hardware_error_validation="unavailable: no independent measured per-request reference supplied")
    (base/"output"/"validation_report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
