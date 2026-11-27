"""Calibrated prefill-work queue simulation; not a GPU/vLLM emulator.

python simulate_allocation.py
python simulate_allocation.py --sensitivity
python simulate_allocation.py --arrivals measured_arrivals.csv --rates measured_rates.csv

Configuration is in CONFIG below. Plotting is a separate standalone file.
Requires Python >=3.10 and NumPy. All rates are aggregate PER WORKSTATION,
including its GPUs; GPU counts are NEVER an additional capacity multiplier.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass, replace
import csv
import hashlib
import heapq
import json
from pathlib import Path
import platform

import numpy as np

BASE = Path(__file__).resolve().parent
POLICIES = ("L3GSA", "Dyncast-task", "CoAP-static")
AUXILIARY_POLICIES = ("Static-demand-aware",)


@dataclass(frozen=True)
class Settings:
    input_tokens: int = 400
    output_tokens: int = 1  # Workload definition. Decode is NOT simulated.
    per_workstation_prefill_tps: tuple = (2823.0, 581.0)
    workstations_per_site: tuple = (4, 4)
    # Preserve old per-workstation offered load: 6 * 4 = 24 rps.
    # This is an explicit scenario choice, NOT a measurement.
    offered_rps: float = 24.0
    site1_precise_fraction: float = 0.10
    site2_precise_fractions: tuple = (0.0, 0.10, 0.15, 0.20)
    seeds: tuple = (2026112701, 2026112702, 2026112703)
    warmup_s: float = 60.0
    measurement_s: float = 300.0
    arrival_gamma_shape: float = 0.7  # Retained scenario assumption, not fitted.
    metric_window_s: float = 10.0
    report_period_s: float = 1.0
    report_delay_s: float = 0.2
    max_batch_requests: int = 1  # Effective FCFS token server, not real GPU batch size.
    routing_metric: str = "processed_work_plus_backlog"
    active_local_clients: int = 4  # Four clients in the tested region; other region idle.
    client_nic_mbps: float = 1000.0
    upload_mbps: tuple = (942.0, 944.0)
    download_mbps: tuple = (942.0, 944.0)
    input_bytes_per_token: float = 4.0  # Payload encoding assumption.
    request_overhead_bytes: int = 128
    response_bytes: int = 132
    # Unknown propagation and switch processing are EXCLUDED, not measured as zero.
    one_way_propagation_ms: float = 0.0


CONFIG = Settings()


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def make_arrivals(c, seed, file=None):
    ar, cl, st, client = np.random.SeedSequence(seed).spawn(4)
    rg = np.random.default_rng(ar)
    duration = c.warmup_s + c.measurement_s
    if file:
        with Path(file).open(newline="", encoding="utf-8-sig") as f:
            rr = list(csv.DictReader(f))
        times = np.array([float(r["arrival_s"]) for r in rr])
        if len(times) < 2 or np.any(np.diff(times) < 0) or times[0] < 0 or times[-1] >= duration:
            raise ValueError("arrival_s must be nondecreasing and inside the configured observation horizon")
        clients = np.array([int(r["client_id"]) for r in rr])
        if np.any(clients < 1) or np.any(clients > c.active_local_clients):
            raise ValueError("client_id outside local client range")
        observed_kinds = np.array([int(r["kind"]) for r in rr]) if rr and "kind" in rr[0] else None
        if observed_kinds is not None and np.any((observed_kinds<0)|(observed_kinds>2)):
            raise ValueError("Measured kind must be 0=fuzzy, 1=site1 precise, 2=site2 precise")
    else:
        arr, t = [], 0.0
        while True:
            t += rg.gamma(c.arrival_gamma_shape, 1/(c.arrival_gamma_shape*c.offered_rps))
            if t >= duration:
                break
            arr.append(t)
        times = np.array(arr)
        clients = np.random.default_rng(client).integers(1, c.active_local_clients+1, len(times))
    measured = times >= c.warmup_s
    ranks = np.zeros(len(times), dtype=int)
    cr = np.random.default_rng(cl)
    for mask in (~measured, measured):
        ranks[mask] = cr.permutation(int(mask.sum()))
    result = dict(times=times, clients=clients, measured=measured, ranks=ranks,
                  uniform=np.random.default_rng(st).random(len(times)))
    if file and observed_kinds is not None:
        result["observed_kinds"] = observed_kinds
    return result


def make_classes(c, trace, fraction):
    if "observed_kinds" in trace:
        n = int(trace["measured"].sum())
        if not n:
            raise ValueError("Replay trace has no measured arrivals")
        count = int(np.sum(trace["observed_kinds"][trace["measured"]] == 2))
        if abs(count/n-fraction) > .5/n+1e-10:
            raise ValueError("Measured class fraction does not match --fraction; use the recorded nominal condition")
        return trace["observed_kinds"].copy()
    kinds = np.zeros(len(trace["times"]), dtype=int)
    for mask in (~trace["measured"], trace["measured"]):
        n = int(mask.sum())
        n1, n2 = round(n*c.site1_precise_fraction), round(n*fraction)
        ranks = trace["ranks"]
        kinds[mask & (ranks < n1)] = 1
        kinds[mask & (ranks >= n1) & (ranks < n1+n2)] = 2
    return kinds


class Engine:
    """Per-workstation, work-conserving FIFO service with exact completion events.

    max_batch_requests>1 groups already waiting jobs without waiting to fill a
    batch. Total work is additive; this only tests a completion-granularity
    assumption. It does NOT model a GPU batch throughput curve.
    """
    def __init__(self, c, trace, kinds, policy, rate_rows=(), demand_fraction=None):
        self.c, self.trace, self.kinds, self.policy = c, trace, kinds, policy
        self.demand_fraction = demand_fraction
        self.now, self.sequence = 0.0, 0
        self.events = []
        self.workers = []
        for site, number in enumerate(c.workstations_per_site):
            for k in range(number):
                self.workers.append(dict(site=site, worker=k+1, rate=c.per_workstation_prefill_tps[site],
                                         remaining=0.0, active=[], queue=deque(), generation=0))
        self.served = np.zeros(2)
        self.accepted = np.zeros(2, dtype=int)
        self.inflight_tokens = np.zeros(2)
        self.worker_busy_s = np.zeros(len(self.workers))
        self.upload_free, self.download_free = np.zeros(2), np.zeros(2)
        self.client_free = np.zeros(c.active_local_clients)
        self.reports = [(0., 0., 0, self.capacity(i)) for i in range(2)]
        self.samples = {}
        self.assigned_samples = {}
        self.snapshot_rows = []
        self.requests = []
        self.conservation_error = 0.0
        for i, (t, kind, client) in enumerate(zip(trace["times"], kinds, trace["clients"])):
            self.requests.append(dict(request_id=i, arrival_s=float(t), client_id=int(client),
                                      kind=int(kind), measurement=int(trace["measured"][i]),
                                      assigned_site=0, worker=0, network_arrival_s=None,
                                      service_start_s=None, prefill_finish_s=None,
                                      surrogate_response_s=None, route_report_age_s=None))
            self.push(float(t), 4, "arrival", i)
        duration = c.warmup_s+c.measurement_s
        for k in range(int(np.floor(duration/c.report_period_s))+1):
            self.push(k*c.report_period_s, 2, "sample", k)
        for r in rate_rows:
            s, w = int(r["site"])-1, int(r["worker"])
            match = next((i for i,x in enumerate(self.workers) if x["site"]==s and x["worker"]==w), None)
            if match is None or float(r["prefill_tps"]) <= 0 or float(r["time_s"]) < 0:
                raise ValueError("Invalid measured rate row")
            self.push(float(r["time_s"]), 0, "rate", (match, float(r["prefill_tps"])))

    def capacity(self, site):
        return sum(w["rate"] for w in self.workers if w["site"]==site)

    def push(self, t, priority, event, payload):
        self.sequence += 1
        heapq.heappush(self.events, (t, priority, self.sequence, event, payload))

    def advance(self, t):
        dt = t-self.now
        if dt < -1e-10:
            raise AssertionError("Time reversed")
        for i,w in enumerate(self.workers):
            if w["active"]:
                work = min(w["remaining"], w["rate"]*dt)
                w["remaining"] = max(0., w["remaining"]-work)
                self.served[w["site"]] += work
                # Busy time clips the interval at the exact service boundary.
                self.worker_busy_s[i] += work/w["rate"]
        self.now = t

    def schedule_finish(self, wi):
        w = self.workers[wi]
        w["generation"] += 1
        if w["active"]:
            self.push(self.now+w["remaining"]/w["rate"], 1, "finish", (wi, w["generation"]))

    def start(self, wi):
        w = self.workers[wi]
        if w["active"] or not w["queue"]:
            return
        batch = [w["queue"].popleft() for _ in range(min(len(w["queue"]), self.c.max_batch_requests))]
        w["active"] = batch
        w["remaining"] = float(len(batch)*self.c.input_tokens)
        for ri in batch:
            self.requests[ri]["service_start_s"] = self.now
        self.schedule_finish(wi)

    def sample(self, k):
        c = self.c
        self.samples[k] = self.served.copy()
        self.assigned_samples[k] = self.accepted.copy()
        window_steps = round(c.metric_window_s/c.report_period_s)
        previous = self.samples.get(k-window_steps, np.zeros(2))
        backlog = np.maximum(0., self.accepted*c.input_tokens-self.served)
        payload = []
        for site in range(2):
            processed = self.served[site]-previous[site]
            base = processed+backlog[site]
            if c.routing_metric == "recent_assignments_only":
                old = self.assigned_samples.get(k-window_steps, np.zeros(2))[site]
                base = (self.accepted[site]-old)*c.input_tokens
            payload.append((self.now, base, int(self.accepted[site]), self.capacity(site)))
            queue_tokens = sum(w["remaining"]+len(w["queue"])*c.input_tokens
                               for w in self.workers if w["site"]==site)
            inflight = self.inflight_tokens[site]
            err = abs(backlog[site]-queue_tokens-inflight)
            self.conservation_error = max(self.conservation_error, err)
            if err > 1e-5:
                raise AssertionError(f"Work conservation failed: {err}")
            self.snapshot_rows.append(dict(sample_s=self.now, site=site+1,
                capacity_tps=self.capacity(site), outstanding_input_tokens=float(backlog[site]),
                processed_tokens_last_window=float(processed), accepted_requests=int(self.accepted[site]),
                waiting_requests=sum(len(w["queue"]) for w in self.workers if w["site"]==site),
                active_requests=sum(len(w["active"]) for w in self.workers if w["site"]==site)))
        self.push(self.now+c.report_delay_s, 3, "report", payload)

    def route(self, ri):
        c, req = self.c, self.requests[ri]
        kind = req["kind"]
        if self.policy != "Dyncast-task" and kind:
            site = kind-1
        elif self.policy in ("CoAP-static", "Static-demand-aware"):
            caps = np.array(c.per_workstation_prefill_tps)*np.array(c.workstations_per_site)
            probability = caps[1]/caps.sum()
            if self.policy == "Static-demand-aware":
                if self.demand_fraction is None:
                    raise ValueError("Auxiliary static control requires a declared forecast fraction")
                fuzzy_fraction = 1-c.site1_precise_fraction-self.demand_fraction
                probability = float(np.clip((probability-self.demand_fraction)/fuzzy_fraction,0,1)) if fuzzy_fraction>0 else 0.
            site = int(self.trace["uniform"][ri] < probability)
        else:
            costs = []
            for i, (_, base, count_at_sample, capacity) in enumerate(self.reports):
                # Locally known assignments since that sample are counted exactly.
                # Reported history/backlog is held until the next report arrives.
                unreported = (self.accepted[i]-count_at_sample)*c.input_tokens
                costs.append((base+unreported+c.input_tokens)/(capacity*c.metric_window_s))
            site = int(costs[1] < costs[0])  # Deterministic ties: site 1.
        req["assigned_site"] = site+1
        req["route_report_age_s"] = self.now-self.reports[site][0]
        self.accepted[site] += 1
        self.inflight_tokens[site] += c.input_tokens
        client = req["client_id"]-1
        begin = max(self.now, self.client_free[client], self.upload_free[site])
        size = c.input_tokens*c.input_bytes_per_token+c.request_overhead_bytes
        end = begin+8*size/(min(c.client_nic_mbps,c.upload_mbps[site])*1e6)
        self.client_free[client] = self.upload_free[site] = end
        req["network_arrival_s"] = end+c.one_way_propagation_ms/1000
        self.push(req["network_arrival_s"], 1, "enqueue", ri)

    def run(self):
        while self.events:
            t, _, _, event, payload = heapq.heappop(self.events)
            self.advance(t)
            if event == "arrival":
                self.route(payload)
            elif event == "sample":
                self.sample(payload)
            elif event == "report":
                self.reports = payload
            elif event == "rate":
                wi, rate = payload
                self.workers[wi]["rate"] = rate
                self.schedule_finish(wi)
            elif event == "enqueue":
                req = self.requests[payload]
                self.inflight_tokens[req["assigned_site"]-1] -= self.c.input_tokens
                candidates = [i for i,w in enumerate(self.workers) if w["site"]==req["assigned_site"]-1]
                wi = min(candidates, key=lambda i: (self.workers[i]["remaining"]+
                         len(self.workers[i]["queue"])*self.c.input_tokens)/self.workers[i]["rate"])
                req["worker"] = self.workers[wi]["worker"]
                self.workers[wi]["queue"].append(payload)
                self.start(wi)
            elif event == "finish":
                wi, generation = payload
                w = self.workers[wi]
                if generation != w["generation"]:
                    continue
                if w["remaining"] > 1e-5:
                    raise AssertionError("Premature completion")
                for ri in w["active"]:
                    req = self.requests[ri]
                    req["prefill_finish_s"] = self.now
                    site = req["assigned_site"]-1
                    begin = max(self.now,self.download_free[site])
                    end = begin+8*self.c.response_bytes/(min(self.c.client_nic_mbps,self.c.download_mbps[site])*1e6)
                    self.download_free[site] = end
                    req["surrogate_response_s"] = end+self.c.one_way_propagation_ms/1000
                w["active"] = []
                self.start(wi)
        for req in self.requests:
            if req["prefill_finish_s"] is None:
                raise AssertionError("Lost request")
            a, n, s, f = (req[k] for k in ("arrival_s", "network_arrival_s", "service_start_s", "prefill_finish_s"))
            if not (a <= n+1e-9 and n <= s+1e-9 and s < f):
                raise AssertionError("Invalid lifecycle")
            req["queue_wait_s"] = s-n
            req["effective_service_s"] = f-s
            req["prefill_surrogate_latency_s"] = req["surrogate_response_s"]-a
            req["semantic_match"] = int(req["kind"]==0 or req["kind"]==req["assigned_site"])
        if not np.allclose(self.served,self.accepted*self.c.input_tokens,atol=1e-5,rtol=0):
            raise AssertionError("Final input-token work not conserved")
        return self.requests


def summarise_run(c, requests, policy, rep, seed, fraction, engine, scenario):
    cohort = [r for r in requests if r["measurement"]]
    site2 = [r for r in cohort if r["assigned_site"]==2]
    precise = [r for r in site2 if r["kind"]==2]
    precise_all = [r for r in cohort if r["kind"]]
    wait = np.array([r["queue_wait_s"] for r in cohort])
    latency = np.array([r["prefill_surrogate_latency_s"] for r in cohort])
    end = c.warmup_s+c.measurement_s
    in_window = [r for r in requests if c.warmup_s <= r["prefill_finish_s"] < end]
    row = dict(scenario=scenario, run=rep, seed=seed, pro_precise_fraction=fraction,
               target_precise_rps=c.offered_rps*fraction, policy=policy, requests=len(cohort),
               actual_total_rps=len(cohort)/c.measurement_s,
               actual_precise_rps=sum(r["kind"]==2 for r in cohort)/c.measurement_s,
               site2_requests=len(site2), site2_precise_requests=len(precise),
               fuzzy_to_site2=sum(r["kind"]==0 for r in site2),
               wrong_site1_precise_to_site2=sum(r["kind"]==1 for r in site2),
               precise_at_site2_pct=100*len(precise)/len(site2) if site2 else 0.,
               precise_model_assignment_accuracy_pct=100*sum(r["semantic_match"] for r in precise_all)/len(precise_all),
               queue_wait_mean_s=float(wait.mean()), queue_wait_p95_s=float(np.quantile(wait,.95)),
               surrogate_latency_p50_s=float(np.quantile(latency,.5)),
               surrogate_latency_p95_s=float(np.quantile(latency,.95)),
               prefill_completions_in_window=len(in_window),
               matched_prefill_completions_in_window=sum(r["semantic_match"] for r in in_window),
               measured_arrivals_prefill_finished_after_window=sum(r["prefill_finish_s"] >= end for r in cohort),
               drained_measured_cohort=len(cohort),
               max_work_conservation_error_tokens=engine.conservation_error)
    if policy != "Dyncast-task":
        assert row["precise_model_assignment_accuracy_pct"]==100
    return row


def aggregate(rows, c):
    result, plot = [], []
    caps=np.array(c.per_workstation_prefill_tps)*np.array(c.workstations_per_site)
    weight=caps[1]/caps.sum()
    for fraction in c.site2_precise_fractions:
        p=dict(precise_task_rate_rps=c.offered_rps*fraction,
               ideal_proportion_pct=100*min(fraction/weight,1))
        for policy in POLICIES:
            rs=[r for r in rows if r["pro_precise_fraction"]==fraction and r["policy"]==policy]
            a=np.array([r["precise_at_site2_pct"] for r in rs])
            r=dict(pro_precise_fraction=fraction,target_precise_rps=c.offered_rps*fraction,
                   policy=policy,runs=len(rs),mean=float(a.mean()),min=float(a.min()),max=float(a.max()))
            for k in ("queue_wait_mean_s","queue_wait_p95_s","surrogate_latency_p95_s", "actual_total_rps"):
                r[k+"_mean"]=float(np.mean([x[k] for x in rs]))
            result.append(r)
            for k in ("mean","min","max"):
                p[f"{policy}_{k}"]=r[k]
        plot.append(p)
    return result,plot


def run_scenario(c, out, name, arrivals=None, rate_file=None, detailed=True):
    out.mkdir(parents=True, exist_ok=True)
    rate_rows=[]
    if rate_file:
        with Path(rate_file).open(newline="",encoding="utf-8-sig") as f:
            rate_rows=list(csv.DictReader(f))
    runs,inputs,auxiliary=[],[],[]
    request_path=out/"requests.csv"
    snapshots_path=out/"queue_snapshots.csv"
    with request_path.open("w",newline="",encoding="utf-8") as rf, snapshots_path.open("w",newline="",encoding="utf-8") as sf:
        rw=sw=None
        for rep,seed in enumerate(c.seeds,1):
            trace=make_arrivals(c,seed,arrivals)
            for ri,t in enumerate(trace["times"]):
                inputs.append(dict(run=rep,seed=seed,request_id=ri,arrival_s=float(t),client_id=int(trace["clients"][ri]),
                    measurement=int(trace["measured"][ri]),class_rank=int(trace["ranks"][ri]),
                    fixed_weight_uniform=float(trace["uniform"][ri])))
            for fraction in c.site2_precise_fractions:
                kinds=make_classes(c,trace,fraction)
                selected_policies=POLICIES+(AUXILIARY_POLICIES if name=="main" else ())
                for policy in selected_policies:
                    eng=Engine(c,trace,kinds,policy,rate_rows,demand_fraction=fraction)
                    req=eng.run()
                    runrow=summarise_run(c,req,policy,rep,seed,fraction,eng,name)
                    if policy in AUXILIARY_POLICIES:
                        auxiliary.append(runrow)
                        continue
                    runs.append(runrow)
                    if detailed:
                        tags=dict(run=rep,seed=seed,pro_precise_fraction=fraction,policy=policy)
                        for r in req:
                            row=dict(tags,**r)
                            if rw is None:
                                rw=csv.DictWriter(rf,fieldnames=list(row));rw.writeheader()
                            rw.writerow(row)
                        for r in eng.snapshot_rows:
                            row=dict(tags,**r)
                            if sw is None:
                                sw=csv.DictWriter(sf,fieldnames=list(row));sw.writeheader()
                            sw.writerow(row)
            print(f"{name}: run {rep}/{len(c.seeds)}, measured arrivals={int(trace['measured'].sum())}",flush=True)
    if not detailed:
        request_path.unlink();snapshots_path.unlink()
    summary,plot=aggregate(runs,c)
    write_csv(out/"allocation_runs.csv",runs)
    if auxiliary:
        write_csv(out/"auxiliary_control_runs.csv",auxiliary)
    write_csv(out/"allocation_summary.csv",summary)
    write_csv(out/"plot_data.csv",plot)
    if detailed:
        write_csv(out/"input_traces.csv",inputs)
    manifest=dict(configuration=asdict(c),scenario=name,
        result_type="capacity-calibrated prefill-work queue simulation; not new hardware measurements",
        routing_metric="(reported processed input tokens in last W + reported outstanding input tokens + locally known input work assigned since the sample + current request input tokens)/(reported aggregate prefill tps * W)",
        capacity_scope="2823/581 tokens/s are per-workstation log values; independent replication over 4 workstations/site is an explicit assumption",
        deployment_assumption="Two same-region model pools, one type-A pool and one type-B pool. The PDF specifies hardware types and 4 workstations/site but not this exact model-to-workstation placement.",
        sources=dict(topology="GSA (9).pdf, Section V-A and Table IV; Chinese draft not used",
            workload="User: 400 input tokens, 1 output token",
            rates="User: device prefill token/s logs, 2823 and 581; raw saturated-load logs not supplied"),
        rate_trace_file=str(rate_file) if rate_file else None,
        arrivals_file=str(arrivals) if arrivals else None,
        independent_arrival_repetitions=not bool(arrivals),
        limitations=["Logged prompt throughput may be achieved offered-load throughput rather than saturated service capacity.",
            "Four identical independent replicas per pool are assumed; scaling has not been measured.",
            "FCFS/additive token-work batching does not emulate vLLM continuous batching or KV-cache behavior.",
            "Decode, CPU/API delay and unmeasured propagation/switch delays are excluded; surrogate latency is not measured TTFT/E2E latency.",
            "Unbounded queues and no failures/timeouts; all requests drain, which is not proof of real service completion success.",
            "Main figure is allocation composition. Cohort is defined by arrival time and drained without boundary censoring.",
            "Three run proportions are averaged arithmetically; min-max bars are not confidence intervals.",
            "Sensitivity scenarios are not pooled as extra repetitions.",
            "No error against hardware latency/composition can be quantified without independent measured reference logs."],
        runtime=dict(python=platform.python_version(),numpy=np.__version__))
    for field,file in (("rate_trace_sha256",rate_file),("arrivals_sha256",arrivals)):
        if file:manifest[field]=hashlib.sha256(Path(file).read_bytes()).hexdigest()
    (out/"run_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    return runs,summary,plot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=BASE/"output")
    parser.add_argument("--arrivals",type=Path)
    parser.add_argument("--rates",type=Path)
    parser.add_argument("--fraction",type=float,help="Replay one declared site2 precise fraction")
    parser.add_argument("--sensitivity",action="store_true")
    args=parser.parse_args()
    c=CONFIG
    if args.fraction is not None:
        if not 0 <= args.fraction <= 1-c.site1_precise_fraction:
            raise ValueError("Invalid precise fraction")
        c=replace(c,site2_precise_fractions=(args.fraction,))
    if not np.isclose(c.metric_window_s/c.report_period_s,round(c.metric_window_s/c.report_period_s)):
        raise ValueError("metric_window_s must be an integer multiple of report_period_s")
    if args.arrivals:
        # Replaying one measured trace does not produce three independent arrival traces.
        c=replace(c,seeds=(c.seeds[0],))
    _,summary,plot=run_scenario(c,args.out,"main",args.arrivals,args.rates)
    for r in plot:
        print(f"rps={r['precise_task_rate_rps']:g} " + " | ".join(f"{p}: {r[p+'_mean']:.3f}% [{r[p+'_min']:.3f},{r[p+'_max']:.3f}]" for p in POLICIES))
    if args.sensitivity:
        scenarios=[("poisson_arrivals",replace(c,arrival_gamma_shape=1.)),
                   ("faster_reporting",replace(c,report_period_s=.1,report_delay_s=0.)),
                   ("recent_assignments_only",replace(c,routing_metric="recent_assignments_only"))]
        all_rows=[]
        main_lookup={(r["pro_precise_fraction"],r["policy"]):r["mean"] for r in summary}
        for name,setting in scenarios:
            _,ss,_=run_scenario(setting,args.out/"sensitivity"/name,name,args.arrivals,args.rates,False)
            for r in ss:
                all_rows.append(dict(scenario=name,**r,delta_from_main_pp=r["mean"]-main_lookup[(r["pro_precise_fraction"],r["policy"])]))
        write_csv(args.out/"sensitivity_summary.csv",all_rows)


if __name__=="__main__":
    main()
