# E1 aggregate (pre-registered rules)

Windows: [0, 1, 2, 3]; cells done: w0=19, w1=19, w2=37, w3=37

## Rule (a): precision gain of C2-in over C1 at matched hit ratio

| budget | window | mean gap (points) | overlap hit ratio (%) |
|---|---|---|---|
| 1000 | 0 | -1.24 | 6-36 |
| 1000 | 1 | -1.04 | 6-37 |
| 1000 | 2 | -1.07 | 7-36 |
| 1000 | 3 | -1.01 | 7-37 |
| 10000 | 0 | +1.19 | 6-39 |
| 10000 | 1 | +1.69 | 6-39 |
| 10000 | 2 | +1.39 | 7-39 |
| 10000 | 3 | +1.51 | 7-39 |

| budget | windows | mean gap | 95% t-interval | decision (>= 2 points, interval above 0) |
|---|---|---|---|---|
| 1000 | 4 | -1.09 | [-1.25, -0.92] | not favourable |
| 10000 | 4 | +1.44 | [+1.11, +1.78] | not favourable |

## Rule (b) and realised precision at audit-chosen thresholds

| target | budget | system | feasible/total | realised precision (%) | hit ratio (%) |
|---|---|---|---|---|---|
| 0.90 | 250 | C1-cal | 4/12 | 83.2 | 6.7 |
| 0.90 | 250 | C2-in | 0/12 | nan | nan |
| 0.90 | 250 | C2-qqp-cal | 2/12 | 83.6 | 5.6 |
| 0.90 | 500 | C1-cal | 4/12 | 85.2 | 4.7 |
| 0.90 | 500 | C2-in | 0/12 | nan | nan |
| 0.90 | 500 | C2-qqp-cal | 0/12 | nan | nan |
| 0.90 | 1000 | C1-cal | 8/12 | 85.9 | 4.1 |
| 0.90 | 1000 | C2-in | 4/12 | 81.1 | 7.8 |
| 0.90 | 1000 | C2-qqp-cal | 8/12 | 84.8 | 4.5 |
| 0.90 | 2000 | C1-cal | 12/12 | 87.2 | 3.3 |
| 0.90 | 2000 | C2-in | 4/12 | 87.3 | 3.6 |
| 0.90 | 2000 | C2-qqp-cal | 12/12 | 86.9 | 2.6 |
| 0.90 | 5000 | C1-cal | 12/12 | 89.6 | 1.8 |
| 0.90 | 5000 | C2-in | 12/12 | 83.9 | 6.9 |
| 0.90 | 5000 | C2-qqp-cal | 10/12 | 87.5 | 2.1 |
| 0.90 | 10000 | C1-cal | 12/12 | 88.2 | 2.3 |
| 0.90 | 10000 | C2-in | 12/12 | 89.0 | 4.0 |
| 0.90 | 10000 | C2-qqp-cal | 10/12 | 88.1 | 1.6 |
| 0.95 | 250 | C1-cal | 0/12 | nan | nan |
| 0.95 | 250 | C2-in | 0/12 | nan | nan |
| 0.95 | 250 | C2-qqp-cal | 0/12 | nan | nan |
| 0.95 | 500 | C1-cal | 0/12 | nan | nan |
| 0.95 | 500 | C2-in | 0/12 | nan | nan |
| 0.95 | 500 | C2-qqp-cal | 0/12 | nan | nan |
| 0.95 | 1000 | C1-cal | 4/12 | 90.3 | 1.4 |
| 0.95 | 1000 | C2-in | 0/12 | nan | nan |
| 0.95 | 1000 | C2-qqp-cal | 6/12 | 86.6 | 2.7 |
| 0.95 | 2000 | C1-cal | 4/12 | 89.1 | 1.8 |
| 0.95 | 2000 | C2-in | 0/12 | nan | nan |
| 0.95 | 2000 | C2-qqp-cal | 2/12 | 90.3 | 1.0 |
| 0.95 | 5000 | C1-cal | 12/12 | 91.8 | 0.5 |
| 0.95 | 5000 | C2-in | 8/12 | 89.0 | 3.5 |
| 0.95 | 5000 | C2-qqp-cal | 4/12 | 90.2 | 0.5 |
| 0.95 | 10000 | C1-cal | 4/12 | 91.9 | 0.3 |
| 0.95 | 10000 | C2-in | 12/12 | 91.4 | 2.1 |
| 0.95 | 10000 | C2-qqp-cal | 2/12 | 89.6 | 0.3 |
| 0.70 (amendment) | 250 | C1-cal | 6/6 | 71.7 | 16.0 |
| 0.70 (amendment) | 250 | C2-in | 0/6 | nan | nan |
| 0.70 (amendment) | 250 | C2-qqp-cal | 6/6 | 64.5 | 20.2 |
| 0.70 (amendment) | 500 | C1-cal | 6/6 | 62.1 | 24.1 |
| 0.70 (amendment) | 500 | C2-in | 4/6 | 62.0 | 23.5 |
| 0.70 (amendment) | 500 | C2-qqp-cal | 6/6 | 60.3 | 24.1 |
| 0.70 (amendment) | 1000 | C1-cal | 6/6 | 64.3 | 22.2 |
| 0.70 (amendment) | 1000 | C2-in | 6/6 | 69.2 | 17.8 |
| 0.70 (amendment) | 1000 | C2-qqp-cal | 6/6 | 62.0 | 22.6 |
| 0.70 (amendment) | 2000 | C1-cal | 6/6 | 65.9 | 20.7 |
| 0.70 (amendment) | 2000 | C2-in | 6/6 | 63.8 | 23.1 |
| 0.70 (amendment) | 2000 | C2-qqp-cal | 6/6 | 64.2 | 20.5 |
| 0.70 (amendment) | 5000 | C1-cal | 6/6 | 66.2 | 20.4 |
| 0.70 (amendment) | 5000 | C2-in | 6/6 | 63.4 | 23.7 |
| 0.70 (amendment) | 5000 | C2-qqp-cal | 6/6 | 64.8 | 20.1 |
| 0.70 (amendment) | 10000 | C1-cal | 6/6 | 65.3 | 21.4 |
| 0.70 (amendment) | 10000 | C2-in | 6/6 | 66.1 | 22.2 |
| 0.70 (amendment) | 10000 | C2-qqp-cal | 6/6 | 63.5 | 21.4 |
| 0.80 (amendment) | 250 | C1-cal | 4/6 | 81.2 | 8.5 |
| 0.80 (amendment) | 250 | C2-in | 0/6 | nan | nan |
| 0.80 (amendment) | 250 | C2-qqp-cal | 4/6 | 77.6 | 9.4 |
| 0.80 (amendment) | 500 | C1-cal | 6/6 | 77.0 | 11.8 |
| 0.80 (amendment) | 500 | C2-in | 2/6 | 73.0 | 14.7 |
| 0.80 (amendment) | 500 | C2-qqp-cal | 6/6 | 74.4 | 11.6 |
| 0.80 (amendment) | 1000 | C1-cal | 6/6 | 79.3 | 10.1 |
| 0.80 (amendment) | 1000 | C2-in | 4/6 | 75.1 | 13.4 |
| 0.80 (amendment) | 1000 | C2-qqp-cal | 6/6 | 72.0 | 13.0 |
| 0.80 (amendment) | 2000 | C1-cal | 6/6 | 79.0 | 10.2 |
| 0.80 (amendment) | 2000 | C2-in | 6/6 | 78.5 | 10.4 |
| 0.80 (amendment) | 2000 | C2-qqp-cal | 6/6 | 77.4 | 9.5 |
| 0.80 (amendment) | 5000 | C1-cal | 6/6 | 78.6 | 10.4 |
| 0.80 (amendment) | 5000 | C2-in | 6/6 | 74.6 | 14.6 |
| 0.80 (amendment) | 5000 | C2-qqp-cal | 6/6 | 77.1 | 9.7 |
| 0.80 (amendment) | 10000 | C1-cal | 6/6 | 77.9 | 10.9 |
| 0.80 (amendment) | 10000 | C2-in | 6/6 | 78.3 | 12.6 |
| 0.80 (amendment) | 10000 | C2-qqp-cal | 6/6 | 75.9 | 10.3 |

### Paired absolute target error, C2-in vs C1-cal (cells where both are feasible)

| target | cells | mean |err| C1-cal | mean |err| C2-in | difference (C1 - C2) | 95% bootstrap CI | decision |
|---|---|---|---|---|---|---|
| 0.90 | 32 | 2.43 | 4.24 | -1.81 | [-2.81, -0.83] | not favourable |
| 0.95 | 12 | 3.18 | 6.04 | -2.87 | [-4.94, -0.65] | not favourable |
| 0.70 (amendment) | 28 | 5.30 | 6.01 | -0.71 | [-2.14, +0.73] | not favourable |
| 0.80 (amendment) | 24 | 2.36 | 4.69 | -2.33 | [-3.83, -1.04] | not favourable |
