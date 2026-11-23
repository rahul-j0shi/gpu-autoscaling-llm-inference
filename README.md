# Paying Only for Tokens Served: Queue-Driven, Spot-Aware GPU Autoscaling for LLM Inference

Nov 23, 2026 · @Rahul

**Abstract.** This project builds an event-driven autoscaling system for LLM inference on Kubernetes. Instead of CPU or memory utilisation, which misrepresent the load on a GPU-bound model server, it scales workers on queue depth: the number of requests or tokens waiting. It uses KEDA to scale worker pods, including to zero, and a node provisioner to add and remove GPU nodes. It handles spot-instance interruptions by draining in-flight requests within the reclaim notice. A cost model and a replayed bursty workload compare five scaling policies on cost per million tokens, idle GPU-hours, tail time-to-first-token and requests lost to preemption.

**Project 6 of 6** in the inference-systems series. The full system runs on a local multi-node Kubernetes cluster (k3d) on a CPU-only Ubuntu workstation (16 GB RAM); GPU workers and GPU nodes are simulated with faithful timing and metrics.

## 1. Problem and motivation

GPUs dominate the cost of serving language models, and a GPU bills by the hour whether or not it is generating tokens. Real traffic is bursty: peaks during working hours, troughs at night, sudden spikes from batch jobs or agents. A fixed fleet sized for the peak wastes most of its hours; one sized for the average misses its latency targets at peak.

Kubernetes' default Horizontal Pod Autoscaler scales on CPU and memory utilisation, and both mislead for LLM servers. CPU stays low because the GPU does the work, and GPU memory looks constant because engines like vLLM reserve their KV-cache memory at startup. Requests can pile up while the pod appears idle ([analysis](https://dev.to/soniarotglam/why-vllm-autoscaling-on-kubernetes-breaks-and-what-to-use-instead-1231)). The signal that tracks real load is the **queue**: how many requests or tokens are waiting. The vLLM Production Stack's reference configuration scales on `vllm:num_requests_waiting` through KEDA ([vLLM Production Stack](https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/autoscaling-keda.html)).

Cost can fall further with **spot** (preemptible) GPUs, sold cheaply but reclaimable with about two minutes' notice. Using them safely requires draining in-flight work before the node disappears. This remains a gap even in mature systems: NVIDIA Dynamo's planner documentation states that scale-down does not drain in-flight requests ([Dynamo Planner](https://docs.nvidia.com/dynamo/latest/components/planner)).

## 2. Background: Kubernetes autoscaling, spot capacity, unit economics

### 2.1 Kubernetes building blocks

A **pod** is one or more containers scheduled together; a **deployment** keeps a desired number of identical pods running; a **node** is a machine that runs pods. A **service** gives pods a stable network address. The **scheduler** places pods on nodes with enough resources; pods that do not fit stay **Pending**.

### 2.2 Two layers of autoscaling

- **Pod autoscaling** changes how many replicas run. The **HPA** adjusts replicas toward a target metric. **KEDA** (Kubernetes Event-Driven Autoscaling, a CNCF graduated project since August 2023) feeds the HPA from external **scalers**, such as a Prometheus query or a Redis list length, and adds **scale-to-zero** ([CNCF](https://www.cncf.io/announcements/2023/08/22/cloud-native-computing-foundation-announces-graduation-of-kubernetes-autoscaler-keda/)).
- **Node autoscaling** changes how many machines exist. The Cluster Autoscaler and **Karpenter** add nodes when pods are Pending and remove nodes that are empty or underused.

The two interact: KEDA creates pods, pods go Pending, the node provisioner adds a GPU node, the model loads, and only then is capacity live. That chain, often minutes for a GPU node plus model load, is the **cold-start** cost every policy must account for.

### 2.3 Queue-based scaling

The basic rule sets replicas proportional to backlog:

```latex
\text{desired replicas} = \left\lceil \frac{\text{queue depth}}{\text{target per replica}} \right\rceil
```

The vLLM reference uses a target of 5 waiting requests, a 15 s polling interval and a 360 s cooldown before scaling down; scale-to-zero needs a second trigger on incoming request rate so a new request wakes the system. Variants weight the queue by expected tokens rather than requests, since one long request can cost as much as a hundred short ones.

### 2.4 Spot interruption and graceful drain

When a cloud reclaims a spot node, it sends a notice, typically two minutes ahead. A correct response: **cordon** the node so nothing new lands there; tell each worker to stop accepting requests; let in-flight requests finish or return them to the queue; then terminate. Kubernetes provides a **preStop hook** for the shutdown logic and `terminationGracePeriodSeconds` to bound it; a **PodDisruptionBudget** limits how many replicas may be down at once. Karpenter automates this sequence, launching a replacement first and then tainting, draining and terminating ([Karpenter](https://karpenter.sh/v1.0/concepts/disruption/)).

### 2.5 Unit economics

The figure that matters to a business is cost per unit of output:

```latex
\text{cost per 1M tokens} = \frac{\sum_{\text{nodes}} \text{GPU-hours} \times \text{hourly price}}{\text{tokens served}} \times 10^{6}
```

It must be read alongside a **service-level objective (SLO)**, for example p95 time-to-first-token under 2 s. **Goodput** counts only requests that met the SLO. A policy that is cheap but breaks the SLO is not a valid result.

## 3. System design

Seven components run on one local Kubernetes cluster.

1. **Gateway and queue.** A small HTTP gateway accepts OpenAI-style requests and pushes them to a Redis queue with an estimated token count; it records arrival time for latency measurement.
2. **Inference worker.** Pulls from the queue and serves requests. Version A is a faithful simulator: configurable model-load delay, prefill time proportional to prompt length, per-token decode time and a concurrency limit. Version B runs `llama-server` with a small quantised model. Both export Prometheus metrics named like vLLM's (`num_requests_waiting`, `num_requests_running`, time to first token).
3. **Metrics.** A lightweight Prometheus scrapes workers and the queue; Grafana dashboards show queue depth, replicas, nodes and latency over time.
4. **Pod autoscaler.** A KEDA `ScaledObject` on queue depth, with an activation trigger on request rate for scale-to-zero, and tuned polling, cooldown and stabilisation windows.
5. **Node provisioner.** A small controller (Python with `kopf`, or Go) that watches Pending worker pods and adds simulated "GPU nodes" (k3d nodes labelled and tainted as GPU) after a configurable boot delay, removing nodes that stay empty beyond a timeout. Each node carries a capacity type, on-demand or spot, and an hourly price.
6. **Spot interruption simulator.** A chaos controller that selects spot nodes at a configurable rate, posts a two-minute interruption notice, then deletes the node.
7. **Drain logic and cost ledger.** Workers handle the notice through a preStop hook: stop pulling, finish requests that can complete in time, return the rest to the queue. A ledger records node-seconds by capacity type and tokens served, producing cost per million tokens.

## 4. Implementation plan

| # | Milestone | Deliverable | Done when |
| --- | --- | --- | --- |
| M1 | Cluster foundations | k3d cluster (1 server, 3 agents), kubectl and Helm workflow | A sample deployment scales manually and survives a node restart |
| M2 | Worker and gateway | Simulated worker, Redis queue, gateway, Prometheus metrics | Requests flow end to end; metrics visible in Prometheus |
| M3 | Queue-based pod scaling | KEDA ScaledObject on queue depth with scale-to-zero | Replicas follow a step load up and back to zero |
| M4 | Node provisioning | Controller adding and removing simulated GPU nodes with boot delay | Pending pods trigger nodes; empty nodes are removed |
| M5 | Spot interruption and drain | Chaos controller; preStop drain; requeue logic | Zero lost requests across 50 interruptions |
| M6 | Cost ledger and dashboards | Per-node billing by capacity type; Grafana dashboards | Cost per 1M tokens computed for any run |
| M7 | Policy comparison | Five policies on the same replayed trace | Section 5 results table and plots |
| M8 | Realistic worker | Replace the simulator with `llama-server` on a small quantised model | Same experiments rerun; differences explained |

Resource notes. A k3d cluster with KEDA, Redis, a slim Prometheus (not the full kube-prometheus stack) and several simulated workers is estimated to need 4–6 GB of RAM, which fits alongside development on a 16 GB machine. No public free tier offers GPUs inside Kubernetes, so GPU nodes are simulated; the control logic, timing and cost accounting are the subject of the project and behave identically. Use current published on-demand and spot GPU prices from one cloud provider in the cost model, and cite the date. Expected effort: three to four weeks part-time.

## 5. Evaluation

**Workload.** A 24-hour synthetic trace compressed to about one hour of wall-clock time, with a daily cycle, two sharp spikes and an idle period. Request lengths follow a long-tailed distribution.

**Policies compared.**

1. Static fleet sized for peak.
2. CPU-based HPA (the Kubernetes default).
3. Queue-depth KEDA scaling with a minimum of one replica.
4. Queue-depth scaling with scale-to-zero.
5. Queue-depth scaling on a spot/on-demand mix with graceful drain.

| Metric | Definition | Why it matters |
| --- | --- | --- |
| Cost per 1M tokens | Ledger cost ÷ tokens served × 10⁶ | The headline economic result |
| Idle GPU-hours | Node-hours with no request in progress | Waste the policy failed to remove |
| p95 / p99 TTFT | Time to first token at the tail | User-facing latency |
| SLO attainment, goodput | % of requests within target; requests/s within target | Whether savings broke service |
| Scale-up latency | Queue spike to first new token served | Cold-start cost |
| Lost requests | Requests failed during interruptions | Correctness of drain |
| Scaling oscillation | Replica changes per hour | Stability of the controller |

**Additional experiments.** Sensitivity to polling interval and cooldown; queue measured in requests vs estimated tokens; interruption rate from 0 to high on the spot-mix policy; cold-start delay sweep to find when scale-to-zero stops paying off.

## 6. What you will learn and what it demonstrates

**What you will learn:**

- **Kubernetes in depth.** Pods, deployments, services, scheduling, taints and tolerations, labels, Helm, and writing a custom controller.
- **Autoscaling control.** Pod vs node autoscaling, metric selection, polling, cooldown and stabilisation, and why badly tuned controllers oscillate.
- **Why LLM serving breaks standard assumptions.** GPU-bound work, preallocated memory, long cold starts and highly variable request costs.
- **Reliability engineering.** Graceful shutdown, draining, requeueing, disruption budgets and chaos testing.
- **Observability.** Prometheus metrics design, PromQL queries, Grafana dashboards and latency percentiles.
- **FinOps.** Building a cost model, unit economics per million tokens, spot vs on-demand trade-offs, and reading cost together with SLOs.

**What it demonstrates to a reviewer:**

- You can run inference as a production service, not only as a model.
- You make infrastructure decisions with a cost model and an SLO, the way platform teams are measured.
- You can build and test failure handling deliberately, with a measurable result: zero lost requests under interruption.

## 7. Extensions

- **Prefix-aware routing.** Route each request to the worker most likely to hold its prompt prefix in cache (connects to Project 3), as llm-d and the Kubernetes Gateway API Inference Extension do. Compare hit rate and TTFT against round-robin.
- **Disaggregated prefill and decode.** Separate worker pools for prompt processing and token generation, each scaled on its own signal.
- **SLO-driven scaling.** Scale on predicted TTFT rather than raw queue length, in the spirit of Dynamo Planner's SLA mode.
- **Predictive scaling.** Forecast load from the daily cycle and pre-warm nodes before spikes to hide cold starts.
- **Real inference engine.** Swap the worker for vLLM on a single cloud GPU for one short validation run, confirming metric names and drain behaviour against the real engine.

## 8. Glossary and references

| Term | Meaning |
| --- | --- |
| Pod, deployment, node, service | Running container group; replica manager; machine; stable network endpoint |
| Pending pod | A pod that cannot be placed for lack of resources |
| Taint, toleration | Node marking that repels pods; pod permission to ignore it |
| HPA | Kubernetes' Horizontal Pod Autoscaler |
| KEDA, ScaledObject, scaler | Event-driven autoscaler; its configuration; its metric source |
| Scale-to-zero, activation | Removing all replicas when idle; the trigger that wakes them |
| Cluster Autoscaler, Karpenter | Node-level autoscalers |
| Cold start | Delay from scale-up decision to serving capacity |
| Queue depth, backlog | Requests or tokens waiting |
| Polling interval, cooldown, stabilisation window | How often to check; how long to wait before scaling down; smoothing |
| Spot / preemptible instance | Discounted capacity the provider can reclaim |
| Interruption notice | Advance warning before reclaim |
| Cordon, drain | Block new pods on a node; move existing work off it |
| preStop hook, termination grace period | Shutdown handler; its time limit |
| PodDisruptionBudget | Minimum available replicas during disruption |
| SLO, SLA | Internal latency target; contractual commitment |
| p95, p99 | Latency below which 95% or 99% of requests fall |
| Goodput | Throughput counting only SLO-compliant requests |
| FinOps, unit economics | Cloud cost management; cost per unit of output |
| Prometheus, PromQL, Grafana | Metrics store; its query language; dashboards |
| Prefix-aware routing | Sending requests to workers with matching cached prefixes |
| Disaggregated serving | Prefill and decode on separate worker pools |

**References**

1. vLLM Production Stack, *Autoscaling with KEDA*. [docs.vllm.ai](https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/autoscaling-keda.html)
2. KEDA graduation announcement, CNCF, Aug 2023. [cncf.io](https://www.cncf.io/announcements/2023/08/22/cloud-native-computing-foundation-announces-graduation-of-kubernetes-autoscaler-keda/)
3. Karpenter, *Disruption*. [karpenter.sh](https://karpenter.sh/v1.0/concepts/disruption/)
4. NVIDIA Dynamo, *Planner*. [docs.nvidia.com](https://docs.nvidia.com/dynamo/latest/components/planner)
5. KServe, *LLMInferenceService autoscaling* (llm-d Workload Variant Autoscaler). [kserve.github.io](https://kserve.github.io/website/docs/next/model-serving/generative-inference/llmisvc/autoscaling/llmisvc-autoscaling)
6. AIBrix release. [vllm.ai](https://vllm.ai/blog/aibrix-release)
7. *Why vLLM autoscaling on Kubernetes breaks*. [dev.to](https://dev.to/soniarotglam/why-vllm-autoscaling-on-kubernetes-breaks-and-what-to-use-instead-1231)
