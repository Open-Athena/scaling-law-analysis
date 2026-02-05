# Derivations Spec

Mathematical derivations for scaling law inference errors.

## Derivation 1: Approach 2 Intercept Error

**Goal**: Derive a closed-form expression for errors in Approach 2 N* inference (both exponent and intercept) as a function of loss surface parameters and grid specification.

**Inputs**: α, β (loss surface exponents), W (grid half-width in log₁₀ space), n (number of sample points).

**Approach**:

1. Work in log-space coordinates centered on the true optimum: w = log₁₀(N/N*), so N = N*·10^w
   - w measures how many orders of magnitude a sample point is from the true optimum: w=0 at the optimum, w=1 means 10× larger, w=-1 means 10× smaller
2. Substitute into the IsoFLOP loss L(N;C) = E + A·N^(-α) + B·(6N/C)^β to get L(w)
   - This gives L(w) = E + P·10^(-αw) + R·10^(βw) where P and R are functions of N*, C, and loss surface parameters
   - Note: L(w) is not a parabola in w — it contains exponential terms
3. Use the first-order optimality condition (dL/dN = 0 at N*) to relate P and R, which may allow simplification
   - At the true optimum, the loss is minimized, so derivatives balance. This constrains the ratio of P to R, potentially letting you eliminate one in favor of the other
4. Derive parabola coefficients from least-squares fitting:
   - Fitting L̂ = a₀ + a₁w + a₂w² to points (wᵢ, Lᵢ) minimizes Σ(Lᵢ - a₀ - a₁wᵢ - a₂wᵢ²)²
   - Taking derivatives w.r.t. a₀, a₁, a₂ and setting to zero gives normal equations
   - The solution involves sums: Σwᵢ, Σwᵢ², Σwᵢ³, Σwᵢ⁴, ΣLᵢ, ΣwᵢLᵢ, Σwᵢ²Lᵢ
   - For equally-spaced points in [-W, W], the grid is symmetric about 0: for each +wⱼ there is a -wⱼ
   - Odd-power sums vanish by cancellation: wⱼ + (-wⱼ) = 0, so Σwᵢ = 0 and Σwᵢ³ = 0
   - This decouples some of the normal equations, yielding simpler formulas for a₁ and a₂
   - The parabola vertex is at w = -a₁/(2a₂)
5. Express a₁ and a₂ in terms of sums involving L(wᵢ), then substitute the loss function L(w) = E + P·10^(-αw) + R·10^(βw)
   - The sums ΣLᵢ, ΣwᵢLᵢ, Σwᵢ²Lᵢ each become sums of exponential terms evaluated at the grid points
   - Use the P/R relationship from step 3 to simplify; look for terms that cancel or factor out
   - The goal is an expression for a₁ and a₂ (and hence the vertex shift) that depends only on α, β, W, n
6. Connect the vertex shift to inference errors
   - If the fitted parabola's minimum is at w = δw instead of w = 0, then the inferred optimum is N̂* = N*·10^δw (off by a multiplicative factor)
   - Approach 2 repeats this at multiple compute budgets, then fits log(N̂*) vs log(C) to get exponent and intercept
   - Write log(N̂*) in terms of log(N*) and δw, then see how δw affects the slope (exponent) and intercept of that fit
7. Check whether the vertex shift δw depends on C
   - From step 6, log(N̂*) = log(N*) + δw·log(10). If δw varies with C, it changes the slope of log(N̂*) vs log(C), corrupting the exponent
   - If δw is constant across C, it only adds a constant offset, affecting the intercept but not the exponent
   - Look back at your expression for δw from step 5: does it contain C, or only α, β, W, n?

**Validation**: 
- Compare derived expressions against numerical Approach 2 results across multiple surface configurations (symmetric, chinchilla, high imbalance)
- Target machine precision agreement (1e-10)
- Sanity check: symmetric surfaces (α = β) should produce zero error

## Deliverables

- `results/derivations/scaling_exponent_errors.html` — derivation document
- `src/scaling_law_analysis/derivations/exponent_errors.py` — validation code
- `results/derivations/validation_intercept_errors.png` — comparison figure
