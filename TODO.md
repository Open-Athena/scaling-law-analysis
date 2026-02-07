# TODO

On article:

- Discuss practical relevance of sampling grid width
  - Note that many experiments (like Llama3 and Marin) use token grids spanning 1-2 decades (OOMs base 10)
  - This contrasts with the 16x sampling grid in the current write-up, which spans ~2.4 decades
  - Make sure to show errors for more common grid widths (e.g. +/-10x)
  - Add section on what is a "normal" grid width
- Links to share from simple demo example:
  - https://gemini.google.com/share/707eb7ef86ee
  - https://chatgpt.com/share/69879ab5-957c-800e-a37f-038b10d79f1e
- Reduce decimal precision in figures where possible
- Expand on this:
  - > For surfaces with asymmetric exponents, wider sampling grids amplify the parabola-fitting mismatch, increasing the constant vertex shift and thus the intercept bias.
  - Mention that it affects extrapolation as well
- Rename "High Imbalance" to "Asymmetric"
- Replace references to scaling grid sizes with code blocks e.g. `Small` instead of "Small"
- Change y-range of first extrapolation error plot (bars are too small)
- Update "The Happy Path — Symmetric Surfaces" section title
- Create combined figure of real isoflop curve plots from Llama, DeepSeek, etc. showing how these sampling biases are real
- Convert to github.io 
- Add CI
- Prior to final review:
  - Review figures.py for ways to use existing code utilities and then regen
