# TODO

On article:

- Primary:
  - Code:
    - Check LossSurface.loss (add vectorized version)
      - return self.E + self.A / np.power(N, self.alpha) + self.B / np.power(D, self.beta)
    - Check bases for logs
    - Standardize the 2d/5d grid search 
  - Mention the demo prompt examples for making your own simulator; examples:
    - https://gemini.google.com/share/6b5b3e9b3e0b
    - https://chatgpt.com/share/69879ab5-957c-800e-a37f-038b10d79f1e
  - Add limitation on the possibility of sampling grid errors canceling out
  - Add limitation on not exploring data efficiency (yet)
  - Add limitation on how 2D grid search + Nelder-Mead probably won't scale well to many more parameters
  - Add appendix table on published scaling law exponents:
    - See "Published Fit Parameters" in Scaling Laws notes
  - Improve this after adding appendix table w/ other reported numbers:
    - > "An exponent ratio of 3.0 is comparable to what has been observed in practice, e.g. DeepSeek[3] reports compute-optimal allocation exponents"
  - On scaling laws in other modalities:
    - See [ProGen3](https://www.biorxiv.org/content/10.1101/2025.04.15.649055v2.full.pdf):
      - "We fit a variant of the equation proposed by Kaplan et al"
      - It looks like they used Kaplan formula 1.5 exactly
    - SODA from Marin: https://arxiv.org/abs/2602.16687
      
- Secondary:
  - Validate the intercept error proof in greater detail manually
  - Pad out list of Approach 2 uses in first paragraph
  - Add citations from "Configuration-to-Performance Scaling Law with Neural Ansatz" on other adaptations of functional forms for Chinchilla scaling laws
- Prior to final review:
  - Review figures.py for ways to use existing code utilities and then regen