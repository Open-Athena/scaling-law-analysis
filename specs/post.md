
# Post Spec: Problems with Chinchilla Approach 2

> **Editorial Guidelines**
> - Length: target a ~10 minute read
> - Audience: ML practitioners familiar with scaling laws but not Approach 2/3 nuances
> - **Docx export:** convert to `results/post/post.docx` by running `uv run python3 -c "import pypandoc; ..."` (requires `pypandoc` and `pypandoc_binary`). Skip the Editorial Guidelines blockquote and "New references" sections.

## Introduction

This post is about neural scaling laws, so I'll start off by saying that I have no idea how to define exactly what those are now. An abridged history for them starts with early observations in computer vision that informed more canonical formulations in NLP (Kaplan/Hoffman laws), which were then succeeded by a zoo of adaptations to various training regimes, data modalities, and downstream applications. Scaling laws for transfer learning, precision, overfitting, overtraining, infinite compute, repetition (decidedly un-infinite compute), MoE sparsity, data mixtures, generalization performance, heterogenous eval subsets, and so on all exist now with varying degrees of fidelity and adoption. These examples are all largely specific to text too -- similar scaling laws for vision, vision-language models, audio, DNA, proteins, etc. also exist and many include their own idiosyncratic extensions to standard Kaplan or Hoffman (i.e. Chinchilla) forms.

Adherence to the Chinchilla flavor of functional form fit on empirical data is a consistent undercurrent in all of the methods above, but the line between "law" and "ansatz" within them was lost on me a long time ago. It's probably not worth trying to find either as laws are stretched more and more to a meta learning problem, solved in a familiar empirical risk minimization framework (e.g. minimizing extrapolation error) with trained neural networks or agentic discovery processes.

My point is that the word "law" in "neural scaling law" is more of an eponym at this point, which makes the space hard to characterize as someone that is 1) somewhat new to the field and 2) accustomed to statistical laws that you don't usually argue with. Either way, I'll stop talking through my hat about what scaling laws are now and instead tip it to Zhang et al. in [Configuration-to-Performance Scaling Law with Neural Ansatz](https://www.arxiv.org/abs/2602.10300) (disclaimer: Kaiyue is a collaborator of ours) for framing scaling laws as a learned ansatz to predict downstream performance directly from a comprehensive training configuration. That's my favorite definition so far.

We're hardly alone at Open Athena in wanting a tool for this "input=configuration, output=performance" problem.  Our potential configurations span a much wider space than most groups face though. We work on a number of problems in materials science, climate and weather, genomics, proteomics and, well, language via [Marin](https://marin.community/).  Scaling in a number of ways is a common goal as is finding a unified framework to do it within. To that end, our first attempt at developing scaling laws in this multimodal scientific landscape began with DNA. There is at least some, but still very little, scaling science to build on in the world of genomic language modeling, so that on top of resembling text much more than say electron cloud densities or near-earth observations makes it more of a natural starting point. That effort didn't make it very far though before hitting some fundamental issues with even the most definitively law-like of all the scaling laws in Chinchilla.

## Criticisms

The most commonly applied method for fitting Chinchilla laws is "Approach 2" from the paper, which involves fitting parabolas to IsoFLOP curves, finding the minima along those curves and then regressing the minima against FLOP counts. Again as someone who hasn't been working in neural scaling for long, this struck me as either a peculiar bit of Jugaad or something with deep theoretical foundations that got left out of the paper (or any paper I've seen since that used the same method). I was happy to assume the latter for a while until I stared at enough IsoFLOP curves with wonky parabola fits and big discrepancies with fully parametric estimations to wonder if there was something more problematic with this method when applied to data like DNA. That data, when modeled at a single base pair level, has very low information content per token, and this easily skews scaling behavior well away from what's common in text. More specifically, the (controversial) assumption that parameters and tokens ever scale equally in some text datasets is rarely true with DNA. This means that compute-optimal scaling depends on the target scale rather than permitting convenient guidelines like "target 20 tokens per parameter at any scale". It also means that scaling laws need to be estimated from a loss surface with more extreme parameterizations, and that's at least one of the ways in which the approximations inherent to Approach 2 break down.

The math of Approach 2 works by first assuming parabolic approximations to the isocontours of the Chinchilla loss surface defined by equal amounts of compute. Compute isn't a parameter of the loss surface itself, but it is used to constrain the relationship between parameter count ($N$) and token count ($D$) through the approximation $C = 6ND$. This is a problematic assumption in DNA models with tiny vocabularies, which I'll conveniently ignore for now (next blog post maybe). This constraint then gives the formula for an IsoFLOP curve as a univariate function of $D$ (or $N$) alone:

- The Chinchilla loss surface is $L(N, D) = E + A / N^\alpha + B / D^\beta$
- The compute constraint $C = 6ND$ implies $N = C / (6D)$
- Substituting gives the isoFLOP curve as a function of $D$ alone:

$$L(D; C) = E + \frac{A \cdot 6^\alpha}{C^\alpha} D^\alpha + \frac{B}{D^\beta} \tag{1}$$

Aside from some impractical cases like $\alpha=2, \beta=-1$, this is not a parabola and parabolas are only as accurate at estimating this function near critical points as a 2-nd order Taylor approximation. So that immediately raises a number of questions like:

1. How close to the true IsoFLOP curve minima do I need to sample loss values for the Taylor approximation to be accurate?
    - Taylor approximations rely on proximity for fidelity, so what does it mean if I design my IsoFLOP sweep with token counts spanning many orders of magnitude?
2. How sensitive is this approximation to big disparities between token and model scaling? 
    - The underlying IsoFLOP formula isn't symmetric like parabolas are when $\alpha \neq \beta$, so how does approximation accuracy vary as a function of the difference between those two exponents?
3. What happens if the true minimum isn't centered within the token count grid I chose for each FLOP budget?
    - If I train far more models on one side of the minima than another at a given compute budget, or outright miss it entirely, what does that mean for the inferred parabola vertex? And what happens if this lopsidedness varies across compute budgets?
4. Does the accuracy of the approximation even matter?
    - The minima implied by the parabola vertices might be wrong, but if they're wrong in a consistent way then are the results from Approach 2 still valid?  E.g. inferred scaling exponents may still be correct (spoiler: they are in *some* cases) so what does that mean for downstream applications like extrapolation to greater compute budgets (spoiler: nothing good)?
5. Do these kinds of potential problems come up in real scaling law studies?
    - Even though publishing raw IsoFLOP data is not common, making it difficult to analyze directly, can we still observe or infer that some of these potential problems above come up in practice?

Answering these questions and proposing some alternatives is what we tried to do in our paper [Problems with Chinchilla Approach 2](https://open-athena.github.io/scaling-law-analysis/). To give some flavor for it, here's an example showing how inferred parameters and extrapolations from Approach 2 break in virtually ideal conditions (no statistical noise, symmetric loss surface, no concerns for embedding vs non-embedding model params, $C=6ND$ is exact, etc.) aside from token and parameter count grids that aren't perfectly centered on the true minima:

![Approach 2 Error Example](results/article/figures/off_center_drifting_bias/off_center_drifting_bias.png)

This simulated, non-constant "drift" in the difference between known and inferred minima makes every single estimate from Approach 2 wrong. It's also not hard to find clear signs of this in published results, e.g. the next figure in the paper is:

![Published IsoFLOP Examples](results/article/static/isoflop_curve_examples.png)

If anything, it's much harder to find published results that don't show these kinds of potential problems. I'll also emphasize that these handsome parabolas with their clean fits and convincingly captured minima are not enough to be accurate. That wasn't intuitive to me at first, but the size of the token and model parameter grid used, how well centered that grid is, and how asymmetric the underlying loss surface is all have some impact on Approach 2 accuracy. They can cancel or compound in various ways as well, so it's hard to break down biases in real experimental results without both a known ground truth and access to the raw data.

I'll say one more time that I have not been working in the world of neural scaling for long. I love it, and I very much want to see it become an integral part of what we do at Open Athena. I am thoroughly confused though as to how such a seemingly integral building block in the space became so popular, especially after having spent a lot of time trying to simulate scenarios to advantage it (and failing). Its progenitors and power-users at frontier AI labs have undoubtedly applied it, measured its success empirically, and found margins of error to be acceptable. They also undoubtedly have much smarter people than me looking at this problem, so I'll do my best to cover some blind spots with this work:

1. Approach 2 is more data efficient and requires fewer models to be trained
    - This approach is only estimating 2 of the 5 Chinchilla parameters, so it should hypothetically require less training data to fit it
    - I don't think this argument is so simple at all. Approach 2 estimates 6 parameters for every compute budget and 4 parameters across them to give inference on both scaling exponents. Only 2 of those parameters matter in the end (the scaling exponents), but you still need the others to be accurate for the two that matter to be accurate. There is also a contentious relationship between the efficiency of the parabola parameter estimates increasing as sampling grids get wider and the fact that wider grids mean worse Taylor approximation accuracy. I don't know how to model this all analytically so I'll instead point out that none of our simulations demonstrate this argument to be true -- we actually find that Approach 2 is even less efficient (in the statistical sense) than fully parametric methods.
2. Accuracy of scaling laws is not actually that important for many applications
    - They may be used as more of a dead reckoning tool for YOLO runs, loose guidelines, less critical dataset prioritization/mixing tasks (i.e. order matters, not magnitude) or for informing smaller scale experiments as intermediate steps to hero runs, and none of these require high precision.
    - This argument makes a lot of sense to me -- they're not use cases I'm excited about though in new scientific modalities.
3. Estimating scaling laws at sufficiently high compute budgets requires little precision
    - At some point, enough tokens and parameters in any reasonable proportion get you close to the irreducible loss floor. This means that tradeoffs in token and parameter allocations start to become irrelevant, so you don't need to be very precise about them.
    - This argument almost makes sense to me, but why would you need a scaling law in the first place in that scenario rather than just running a single sweep at a compute budget near the floor?  Or choose the allocation based on other reasons?
4. Extrapolation from scaling laws for real training runs isn't that important anymore
    - Modern LLMs are far from compute-optimal, so Chinchilla extrapolations aren't relevant
    - This argument makes complete sense to me, for final models at least. Speedruns are a different story, e.g. Karpathy's nanochat [results](https://x.com/karpathy/status/2009037707918626874) using Approach 2, as are new data modalities/domains where even basic knowledge of compute-optimal scaling is missing.
5. Errors from Approach 2 in real experiments cancel out
    - It might be possible that the systematic biases within a compute budget arise from a random process across budgets, and potentially give close to unbiased estimates in the end.
    - There might be something to this argument. It's easy to design an IsoFLOP experiment where it's definitely not true. That doesn't mean I can rule out that big scaling law experiments might accumulate errors like a Pachinko board and net out at a zero mean gaussian. If there is a reason that I am somehow gilding a lily with this work, I wouldn't be surprised if the CLT was behind it. I've looked at enough published IsoFLOP curves to feel like this is highly unlikely though.


## Optimisms

Maybe all my griping has made some progress in a case against Approach 2, so what about Chinchilla Approach 3? That approach fits scaling laws without any tricks -- it's just a fairly typical 5D nonlinear optimization problem. It is well known for being sensitive to initialization, generally unstable, and prone to finding local optima. While I can't see how Approach 2 solves the local optima problem given that it's typically biased away from the correct answers in the first place, I can say that I think it's a good idea to treat estimating scaling exponents differently from estimating everything else. That's something we explored in the paper as well. The 5D Chinchilla formula is linearly separable so we separate the exponents out from the other coefficients (by Variable Projection), grid search over them in a 2D space before refining them further via optimization, again all in a 2D space (we call that method VPNLS). This definitely elides issues with initialization sensitivity and conditions the optimization far better than the highly anisotropic 5D space.

Ultimately, Approach 3 with a grid search for initialization and the other method we proposed are both viable, better alternatives to Approach 2 in all of the simulations we ran. VPNLS is apparently better at avoiding rare suboptimal fits from Approach 3, and it does so in a way that is very persistent across conditions. That was a fun surprise to be honest -- I was mostly using that method at first to make grid search initializations faster. Regardless, both methods are easy to implement, relatively fast to execute, don't actually require more training data for any reason we're aware of, and most importantly: they're **unbiased**. I would argue any minor hassle from them, e.g. judging convergence per scipy and tuning tolerances down, is very frequently worth it to avoid the footgun of an arguably simpler method that can't guarantee unbiased inference. I'd also argue they're a much better foundation to build on for intentionally introducing bias with priors, regularization, huber loss, or anything else needed to combat outliers or other practical concerns I haven't touched on (non-embedding params, loss spikes, overfitting, etc.).

## Conclusion

- Discuss how diagnostics for Approach 2 could make sense, but you might as well just use a different method
- Discuss how this relates to all the Chinchilla extension methods listed in the beginning
- Share links on how to run your own simulations via one-shot prompts?


### New references

- [Neural Neural Scaling Laws](https://arxiv.org/abs/2601.19831)
- [Can Language Models Discover Scaling Laws?](https://arxiv.org/abs/2507.21184)
- [EvoSLD: Automated Neural Scaling Law Discovery With Large Language Models](https://arxiv.org/abs/2507.21184v1)
- [Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models](https://arxiv.org/abs/2507.17702)