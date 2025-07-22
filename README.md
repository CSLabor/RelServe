# RelServe

We implement the dynamic priority updating mechanism and the adaptive batch arranger inside a [customized version of vLLM](https://github.com/CSLabor/vllm_relserve), please install the customized vLLM first according to guidelines in the repo.

Then use the `entrypoint.sh` to reproduce the experimental results:

```
# Basic Usage: bash entrypoint.sh <sched_name> <model_name> <dataset_name> <req_rate> <starvation> <GPU_id> <nrun>

# vLLM
bash entrypoint.sh fcfs opt_13b amazon 1.0 100 1 0

# vLLM-SP
bash entrypoint.sh priority opt_13b amazon 1.0 100 1 0

# RelServe
bash entrypoint.sh priority_ada_pabs opt_13b amazon 1.0 100 1 0

# RelServe-DP
bash entrypoint.sh priority_islt_pabs opt_13b amazon 1.0 100 1 0

# RelServe-PP
bash entrypoint.sh priority_ovlp_pabs opt_13b amazon 1.0 100 1 0
```

