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
  - Fix DeepSeek 2.7 reference (comes from Kaplan)
  - Finish reproductions (experiment 0)
  - On abstract:
    - Emphasize data efficiency findings
    - Include error extrapolations on Olmo
    - Add mention of JS execution times and brevity
  - Add MoE scaling reference: Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models (Jan 2025)
    - Cite on use of LSE: "we find that applying log transformations improves the fit of the resulting IsoFLOP surface."
  - On data scaling faster than params (also in the post) cite SODA:
    - "This asymmetry (data scaling faster than model size) is consistent with the lower information density of
discrete audio tokens compared to text."
    - Ask Potsawee about "Neither the compute-optimal extrapolation nor the parametric fit very accurately predicts validation loss for our final SODA runs"
  - Add exp6 validation for proof to appendix
  - Cite https://arxiv.org/pdf/2406.19146 when discussing WLS adjustments based on noise at different budgets
    - See 2.3 Data analysis
  - Mention the demo prompt examples for making your own simulator; examples:
    - https://gemini.google.com/share/6b5b3e9b3e0b
    - https://chatgpt.com/share/69879ab5-957c-800e-a37f-038b10d79f1e
  - Cite [Scaling Laws for Native Multimodal Models](https://arxiv.org/pdf/2504.07951) on PlantCAD issue for empirical C ~ D^b method (see C. Scaling Laws)
  - Copy intercept-error proof into paper appendix
  - Make a reference implementation for other users
- Secondary:
  - Validate the intercept error proof in greater detail manually
  - Pad out list of Approach 2 uses in first paragraph
  - Add citations from "Configuration-to-Performance Scaling Law with Neural Ansatz" on other adaptations of functional forms for Chinchilla scaling laws
- Prior to final review:
  - Review figures.py for ways to use existing code utilities and then regen (or push back into experiments code)