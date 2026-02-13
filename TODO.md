# TODO

On article:

- Links to share from simple demo example:
  - https://gemini.google.com/share/6b5b3e9b3e0b
  - https://chatgpt.com/share/69879ab5-957c-800e-a37f-038b10d79f1e
- Reduce decimal precision in figures where possible
- Expand on this:
  - > For surfaces with asymmetric exponents, wider sampling grids amplify the parabola-fitting mismatch, increasing the constant vertex shift and thus the intercept bias.
  - Mention that it affects extrapolation as well
- Rename "High Imbalance" to "Asymmetric"
- Add note on how using log loss intead of loss biases everything all the time?
- Replace references to scaling grid sizes with code blocks e.g. `Small` instead of "Small"
- Revisit this, which is not really correct:
  - > Asymmetric curve shapes: The IsoFLOP curves are visibly steeper on one side of the minimum than the other, consistent with alpha != beta. This is the condition under which the parabolic approximation introduces systematic intercept bias.
- High priority:
  - Make it clear that approach 2 only fits two params of the surface, and that just avoiding the others is an important part of its adoption
  - Add "Why it doesn't matter" section on loss floor at high compute budgets
  - Make connection to downstream performance per FLOP/$
  - Reference OpenAI OpenWebText2 (a=.73, b=.27) and Evo 
  - Justify legitimacy of Chinchilla formula
    - https://arxiv.org/pdf/2509.23963
      - Reference this paper for sure as it also does perturbations to test robustness of the fit (in different interpretations of model parameters b/c paper is ambiguous)
      - They focus on tokens/params as the metric to analyze robustness for
    - Balance with https://arxiv.org/pdf/2502.18969 and Kaplan form for surface (see 3.2) which is different
      - This also explains many extensions for other things (epochs, num experts, sparsity, data modalities/mixtures, etc.)
  - Make sure to mention that we're using standard chinchilla params for simulations despite critiques of that fit from Epoch and Toronto/Stanford paper
- Prior to final review:
  - Review figures.py for ways to use existing code utilities and then regen
- SDD refactor 
  - Refactor AGENTS.md to explain specs 
  - Move build.md to specs/ and add sync.md
  - Add commands for build, sync, review, deploy (on top of global commit and push)
  - copy review as subagent for orthogonality
