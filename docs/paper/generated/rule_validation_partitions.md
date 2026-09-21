# 规则教师的验证分区

selection选择后原样报告calibration/audit；不是按test选参。

| Dataset | Partition | N | Rescued | Lost | ΔR1 | ΔMRR |
| --- | --- | --- | --- | --- | --- | --- |
| nyc | selection | 796 | 42 | 29 | 0.016332 | 0.011394 |
| nyc | calibration | 342 | 19 | 13 | 0.017544 | 0.007350 |
| nyc | audit | 262 | 12 | 17 | -0.019084 | -0.005071 |
| ca | selection | 2043 | 48 | 37 | 0.005384 | 0.005117 |
| ca | calibration | 755 | 13 | 11 | 0.002649 | 0.004300 |
| ca | audit | 731 | 13 | 13 | 0.000000 | 0.006694 |
| tky | selection | 4006 | 100 | 53 | 0.011732 | 0.007720 |
| tky | calibration | 1448 | 25 | 22 | 0.002072 | 0.001470 |
| tky | audit | 1414 | 25 | 17 | 0.005658 | 0.004126 |
