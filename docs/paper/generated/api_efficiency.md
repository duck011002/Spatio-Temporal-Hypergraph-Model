# API 成本与延迟

ledger 按 attempt 最终状态计一次；CA 包含 validation+test，TKY/NYC 当前目录主要是 test；不可直接横比总量。

| Dataset | Run | Successes | Retried requests | Input tokens | Estimated USD | Latency p50/p95 seconds |
| --- | --- | --- | --- | --- | --- | --- |
| NYC | nyc_a5_jev_force_test | 1347 | 2 | 1713133 | 0.07208061 | [0.4045352414250374, 1.3761836208403113] |
| CA | ca_a5_jev | 6309 | 1 | 8238989 | 0.34610205000000005 | [0.3854604959487915, 0.4556398957967758] |
| TKY | tky_a5_jev_v1 | 7038 | 1 | 7807452 | 0.32797749600000004 | [0.3744620718061924, 0.4720665737986564] |
