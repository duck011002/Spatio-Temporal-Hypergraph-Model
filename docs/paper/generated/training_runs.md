# 已完成 A4 训练记录

计时为 summary 的开始至结束，含运行内验证/测试；不是纯 GPU kernel 时间。

| Dataset | Run | Seconds | Parameters | Seed | Neighbors | LR warmup | Validate interval | Cooldown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NYC | 20260723_155322_nyc_sthgcn_seed80786525 | 1778 | 28639395 | 80786525 | 300-500 | 8000 | 500 | 1.5 |
| CA | 20260921_001455_ca_sthgcn_seed27486607 | 8134 | 32629768 | 27486607 | 300-600 | 14000 | 500 | 1.4 |
| TKY | 20260921_023057_tky_sthgcn_seed54607333 | 20426 | 30986567 | 54607333 | 400-240 | 48000 | 4000 | 1.5 |
