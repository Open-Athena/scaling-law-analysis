# Approach 2 Extrapolation Demo

Create a self-contained Python script comparing Chinchilla Approach 2 token predictions against the analytical ground truth.

All details below reference Hoffmann et al. 2022, "Training Compute-Optimal Large Language Models" (https://arxiv.org/abs/2203.15556).

The script should:

- Define the Chinchilla loss surface: `L(N, D) = E + A/N^α + B/D^β` with parameters `α=0.34, β=0.28, A=406.4, B=410.7, E=1.69` (Appendix D)
- Use the compute constraint `C = 6 * N * D` (FLOPs)
- Derive the analytical ground truth (Section 3): minimizing L subject to the compute constraint yields `Dₒₚₜ(C) = (1/G) * (C/6)^b` where `b = α/(α+β)` and `G = (α*A / (β*B))^(1/(α+β))`
- Implement Approach 2 (Section 3):
  - For each compute budget in `[1e17, 1e18, 1e19, 1e20, 1e21]`, sample 15 points along the IsoFLOP curve spanning ±16x around the true optimum (log₁₀ range = 1.2)
  - Fit a parabola to `L vs log₁₀(D)` for each budget and extract the vertex as the inferred `Dₒₚₜ`
- Fit a power law `log₁₀(Dₒₚₜ) = b * log₁₀(C) + b₀` to the inferred optima via linear regression
- Extrapolate to `C = 1e24` FLOPs: use the fitted coefficients (`b`, `b₀`) to predict `Dₒₚₜ` at the new budget, and compare against the analytical ground truth at the same budget
- Print the true `Dₒₚₜ`, predicted `Dₒₚₜ`, and relative error (%)

Use only numpy and scipy. Keep the code minimal and readable.
