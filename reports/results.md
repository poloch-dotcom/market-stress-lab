# Results (synthetic market, seed 42)

Book priced at spot 7,659, implied vol 25.5%. Short puts struck at 7,276.

## 1. One-day risk measures (EUR)

| Method (1 day) | VaR 99% | ES 97.5% |
|---|---|---|
| Historical (full reval) | 830,642 | 906,516 |
| Normal on P&L | 624,745 | 627,833 |
| Cornish-Fisher | 1,558,410 | - |
| Delta-normal (linearized) | 628,873 | - |
| Monte Carlo, Gaussian | 635,747 | 643,229 |
| Monte Carlo, Student-t (5 dof) | 706,351 | 739,971 |
| Filtered historical (GARCH) | 661,997 | 706,491 |

GARCH on the P&L: alpha = 0.130, beta = 0.828, persistence = 0.958.

## 2. Backtest of the 99% one-day VaR

2500 out-of-sample days, 500-day rolling window, GARCH refitted every 20 days.

| VaR model | exceptions | expected | kupiec_p | christoffersen_cc_p | last_250d | basel_zone | avg_var |
|---|---|---|---|---|---|---|---|
| historical | 33 | 25.0 | 0.125 | 0.065 | 1 | green | 886988 |
| normal | 59 | 25.0 | 0.0 | 0.0 | 3 | green | 633800 |
| fhs | 28 | 25.0 | 0.554 | 0.514 | 3 | green | 828944 |

## 3. Stress tests (10-day horizon)

### Hypothetical scenarios, P&L by position (EUR)

| Scenario (10 days) | equity | bond | credit | fx | short_puts | total |
|---|---|---|---|---|---|---|
| Equity crash, flight to quality | -2,211,992 | 861,600 | -900,000 | 256,355 | -3,098,827 | -5,092,863 |
| Stagflation, rates up | -951,626 | -1,965,000 | -480,000 | 152,273 | -1,130,592 | -4,374,945 |
| Credit event | -1,392,920 | 425,400 | -1,500,000 | 101,007 | -1,782,550 | -4,149,064 |
| Volatility spike only | -295,545 | 70,150 | -90,000 | 0 | -891,623 | -1,207,018 |

### Worst historical windows replayed on today's book

| Historical window (10 days) | EQ | IR | CS | FX | VOL | pnl |
|---|---|---|---|---|---|---|
| window ending 2022-03-01 | -0.252 | -44.012 | 65.183 | -0.026 | 42.589 | -6,102,835 |
| window ending 2016-08-02 | -0.157 | 27.249 | 13.007 | -0.073 | 24.73 | -4,662,181 |
| window ending 2023-12-01 | -0.122 | 14 | 33.729 | -0.018 | 16.68 | -3,358,938 |

### Reverse stress test (stressed covariance: 2016-01-18 to 2016-12-30)

Most plausible 10-day move producing each loss. EQ and FX in log-return, IR and CS in bp, VOL in vol points.

| loss target | EQ | IR | CS | FX | VOL | distance (sigmas) | tail prob. |
|---|---|---|---|---|---|---|---|
| 2.00 M€ | -0.09 | -7.992 | 16.162 | -0.009 | 9.377 | 1.78 | 67.65% |
| 4.00 M€ | -0.178 | -15.794 | 32.134 | -0.017 | 18.15 | 3.49 | 3.27% |
| 6.00 M€ | -0.269 | -22.905 | 49.086 | -0.026 | 27.126 | 5.27 | 0.00% |

## 4. Shock calibration: RMSE on test-period tail moves

Core shocks given: EQ and CS. Predicted: IR (bp), FX (log-return), VOL (vol pts). Lower is better.

| Horizon, model | IR | FX | VOL |
|---|---|---|---|
| 1-day, zero | 7.210 | 0.008 | 2.733 |
| 1-day, gaussian | 6.530 | 0.008 | 1.427 |
| 1-day, gbm | 6.819 | 0.008 | 1.350 |
| 1-day, hybrid | 6.552 | 0.008 | 1.373 |
| 10-day, zero | 23.538 | 0.023 | 9.615 |
| 10-day, gaussian | 20.042 | 0.023 | 5.272 |
| 10-day, gbm | 20.705 | 0.025 | 5.842 |
| 10-day, hybrid | 19.848 | 0.024 | 5.201 |

Example: the narrative 'equities -25%, spreads +150 bp' completed by the hybrid model (10-day): EQ = -0.25, CS = +150, IR = -26.3, FX = +0.0268, VOL = +27.
