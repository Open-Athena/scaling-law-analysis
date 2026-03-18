# TODO

On post:

- Substantiate this more:
  - That data, when modeled at a single base pair level, has very low information content per token, and this easily skews scaling behavior well away from what's common in text. More specifically, the (controversial) assumption that parameters and tokens ever scale equally in some text datasets is rarely true with DNA.

On article:

- Primary:
  - Add limitations:
    - Add limitation on the possibility of sampling grid errors canceling out
    - Add limitation on not exploring data efficiency (yet)
    - Mention hardware discretization in extrapolations, re: https://openathena.slack.com/archives/C0884476QSC/p1773257160211219?thread_ts=1773062864.434009&cid=C0884476QSC
      - This is relevant for the simulations too in continuous param space
    - Mention C=6ND assumption as a limitation
    - Discuss Olmo hybrid assumption of constrained scaling exponents for architecture comparisons
    - Bootstrap for within-budget resampling on Approach 2 only
  - Add https://arxiv.org/abs/2603.03276 as reference on multimodal asymmetry
  - Make a reference implementation
  - Mention FLOP factor correction and WLS weighting for approach 2 as possible improvements
    - Or at least mention importance of reliance on C=6ND assumption
- Secondary:
  - Add WLS analysis
    - Cite https://arxiv.org/pdf/2406.19146 when discussing WLS adjustments based on noise at different budgets
      - See 2.3 Data analysis
  - Add exp6 validation for proof to appendix
  - Consider https://arxiv.org/abs/2603.06603 as another citation for methods that "extend individual terms in isolation (e.g. token scaling terms alone)"
  - Cite [Gemstones: A Model Suite for Multi-Faceted Scaling Laws](https://arxiv.org/pdf/2502.06857) on how C=6ND breaks down w/ model shape
  - Cite [Scaling Laws for Native Multimodal Models](https://arxiv.org/pdf/2504.07951) on PlantCAD issue for empirical C ~ D^b method (see C. Scaling Laws)
  - Mention the demo prompt examples for making your own simulator; examples:
    - https://gemini.google.com/share/6b5b3e9b3e0b
    - https://chatgpt.com/share/69879ab5-957c-800e-a37f-038b10d79f1e
  - Add note advising against using logloss given bias in simulations and ml-scalefit reproduction
  - Improve intercept error proof formatting
  - Copy intercept-error proof into paper appendix
  - Add citations from "Configuration-to-Performance Scaling Law with Neural Ansatz" on other adaptations of functional forms for Chinchilla scaling laws
  - Review figures.py for ways to use existing code utilities and then regen (or push back into experiments code)