# Delphi Scaling Analysis Report

## Llama 3

Data: 133 points, 10 budgets
Flop factor mode: learned per-budget (OLS)

### Approach 2 (Parabolic IsoFLOP Fits)

  a (N* exponent)   = 0.4632
  b (D* exponent)   = 0.5368
  a + b             = 1.0000
  a intercept       = -0.2543
  b intercept       = -0.5238

### Approach 3 (VPNLS Grid, resolution=0.001)

  E     = 0.6048
  A     = 59.14
  B     = 156.22
  alpha = 0.3100
  beta  = 0.3140
  a     = 0.5032
  b     = 0.4968
  RSS   = 0.001480

### Minima Power Law (L = x1 * D*^x2 + x0)

Fit to Approach 2 parabola vertices (D*, L_min).

  x0 (offset)     = 0.649869
  x1 (coefficient) = 3335.410170
  x2 (exponent)   = -0.428627
  RSS             = 0.00015275

### D* Comparison (Approach 3 vs Approach 2)

Positive delta means Approach 3 infers more tokens than Approach 2.

      Budget              k        D* (A2)        D* (A3)      Delta     Delta%
      ------------------------------------------------------------------------
      6.0e18          5.802       4.41e+09       4.31e+09  -9.80e+07      -2.2%
      1.0e19          5.878       5.14e+09       5.52e+09  +3.85e+08      +7.5%
      3.0e19          5.912       7.56e+09       9.50e+09  +1.94e+09     +25.7%
      6.0e19          6.217       1.15e+10       1.31e+10  +1.54e+09     +13.4%
      1.0e20          6.641       1.53e+10       1.63e+10  +9.86e+08      +6.4%
      3.0e20          6.003       2.53e+10       2.96e+10  +4.29e+09     +16.9%
      6.0e20          6.147       4.10e+10       4.13e+10  +3.04e+08      +0.7%
      1.0e21          6.239       5.46e+10       5.28e+10  -1.80e+09      -3.3%
      3.0e21          4.137       9.82e+10       1.12e+11  +1.36e+10     +13.9%
      1.0e22          5.536       2.38e+11       1.76e+11  -6.23e+10     -26.2%

  Mean delta%     = +5.3%
  Mean |delta%|   = 11.6%
  Median delta%   = +7.0%
  Median |delta%| = 10.4%

## Delphi

Data: 78 points, 7 budgets
Flop factor mode: learned per-budget (OLS)

### Approach 2 (Parabolic IsoFLOP Fits)

  a (N* exponent)   = 0.3907
  b (D* exponent)   = 0.5744
  a + b             = 0.9651
  a intercept       = 1.3622
  b intercept       = -1.4342

### Approach 3 (VPNLS Grid, resolution=0.001)

  E     = 2.5786
  A     = 302423.82
  B     = 3928.62
  alpha = 0.6830
  beta  = 0.4130
  a     = 0.3768
  b     = 0.6232
  RSS   = 0.010100

### Minima Power Law (L = x1 * D*^x2 + x0)

Fit to Approach 2 parabola vertices (D*, L_min).

  x0 (offset)     = 1.784029
  x1 (coefficient) = 88.078909
  x2 (exponent)   = -0.184121
  RSS             = 0.00013615

### D* Comparison (Approach 3 vs Approach 2)

Positive delta means Approach 3 infers more tokens than Approach 2.

      Budget              k        D* (A2)        D* (A3)      Delta     Delta%
      ------------------------------------------------------------------------
      2.9e18          3.969       1.48e+09       1.63e+09  +1.46e+08      +9.9%
      9.1e18          5.708       2.90e+09       2.64e+09  -2.55e+08      -8.8%
      1.8e19          6.246       4.20e+09       3.79e+09  -4.15e+08      -9.9%
      3.1e19          6.946       5.70e+09       5.06e+09  -6.35e+08     -11.1%
      8.9e19          6.635       1.02e+10       1.00e+10  -2.34e+08      -2.3%
      1.7e20          6.134       1.59e+10       1.59e+10  +4.52e+07      +0.3%
      3.1e20          5.703       2.19e+10       2.38e+10  +1.96e+09      +8.9%

  Mean delta%     = -1.9%
  Mean |delta%|   = 7.3%
  Median delta%   = -2.3%
  Median |delta%| = 8.9%
