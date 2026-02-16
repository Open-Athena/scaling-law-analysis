# TODO

On article:

- Primary:
  - Add more appendix figures
  - Mention the demo prompt examples for making your own simulator; examples:
    - https://gemini.google.com/share/6b5b3e9b3e0b
    - https://chatgpt.com/share/69879ab5-957c-800e-a37f-038b10d79f1e
  - Add limitation on the possibility of sampling grid errors canceling out
- Secondary:
  - Validate the intercept error proof in greater detail manually
  - Update editorial guidelines to match longer length now
  - Pad out list of Approach 2 uses in first paragraph
  - Justify legitimacy of Chinchilla formula
    - https://arxiv.org/pdf/2509.23963
      - Reference this paper for sure as it also does perturbations to test robustness of the fit (in different interpretations of model parameters b/c paper is ambiguous)
      - They focus on tokens/params as the metric to analyze robustness for
    - Balance with https://arxiv.org/pdf/2502.18969 and Kaplan form for surface (see 3.2) which is different
      - This also explains many extensions for other things (epochs, num experts, sparsity, data modalities/mixtures, etc.)
    - Make sure to mention that we're using standard chinchilla params for simulations despite critiques of that fit from Epoch and Toronto/Stanford paper
    - On Compounding Errors section: add a configuration where the bias sources reinforce rather than offset each other (e.g. an offset direction that pushes in the same direction as asymmetry error), to demonstrate the compounding case directly
  - Reduce decimal precision in figures where possible
- Prior to final review:
  - Review figures.py for ways to use existing code utilities and then regen
