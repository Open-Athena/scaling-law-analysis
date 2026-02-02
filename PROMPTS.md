AGENTS.md

Use uv for env management

---

Create a simulation to do the following:

- Generate data from a chinchilla loss function using published params with no noise 
- Fit that data to attempt to recover parameters 
- Summarize the accuracy of parameter recovery

Do this all in a single file.

---

Revise the data generation logic to sample along isoflop curves instead.  Revise the fitting procedure to use Chinchilla approach 2.  Fit parabolas to results from each compute budget and then infer parameters from a log-linear line through the inferred minimal points.

---

Good now see what happens when alpha is .9 and beta .1

---

Now revise this simulation to generate data for multiple step sizes in the isoflop data.  For each step size, sample around the same center point.  I want to show how parameter recovery changes as the step size changes.

----


Attempt to find an analytical form for the difference between best-fit parabolas and chinchilla isoflop curves.  Use this to directly quantify differences between inferred alpha/beta parameters and true values as a function of compute C.  Prove that this analytical form is correct with empirical tests.  Create a worktree to do this in.
