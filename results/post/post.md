# Problems with Chinchilla Approach 2

*Eric Czech · March 24, 2026 (Final venue TBD)*

## Motivation

I’ll start off by saying that I have no idea how to define exactly what neural scaling laws are now, which is probably not a great start for a post that is critical of a well-established method for finding them. I can at least say that a good definition sits somewhere between where they appear to be going: “economic outputs as a function of economic inputs” and where they started out: “validation loss as a function of dataset size” (i.e. learning curves).

At that first extreme, you may agree with me that “economic outputs” (e.g. human labor equivalents) are unlikely to be so predictable from “economic inputs” (e.g. hero runs at fractional GDP cost) that the relationship between them would be described well as a “law”. At the other extreme, it’s much easier to believe that more direct relationships between model performance (e.g. validation loss) and resources (e.g. model size, FLOPs, time, etc.) might be measurable with high accuracy and simple methods.

My point is that the evolution of neural scaling laws is complicated and now includes a lot of statistical modeling challenges that didn’t exist 4+ years ago. That’s about how long ago the original Kaplan and Hoffmann scaling laws extended earlier, more informal observations (mostly in computer vision) to formulations that hold up well at scale. A lot has changed since then – adaptations now span from low-level training details, e.g. precision, sparsity, depth/width tradeoffs, optimizers, overfitting, epochs, etc., to downstream task performance, agentic systems and more abstract measures of intelligence. The implications of scaling laws have expanded along with difficulties in quantifying them, and both of these motivate a lot of my personal interest in this space.

Doing some of that “quantifying” is where I spend a lot of my time at [Open Athena](https://openathena.ai/). We work in scientific domains like materials science, climate modeling, and genomics/proteomics on top of building competitive open language models (via [Marin](https://marin.community/)). Bridging language models with scientific foundation models is one goal we have, as is better understanding how to scale thoughtfully. To that end, DNA provides a good starting point: it is about as similar to text as any high-throughput scientific data modality and at least one scaling law study already exists for it ([Evo 1](https://www.science.org/doi/10.1126/science.ado9336)). We hit some problems when trying to do this ourselves though, and a few of them made their way into this project: [Problems with Chinchilla Approach 2](https://openathena.ai/scaling-law-analysis/). The rest of this post summarizes the issues, implications and potential alternatives described there.

## Problems

Chinchilla Approach 2 is a commonly applied method for fitting scaling laws from Hoffmann et al. 2022. The method involves fitting parabolas to IsoFLOP curves, finding the minima along those curves, and then regressing the inferred minima against FLOP counts. This gives a way to indirectly estimate the 2 exponential parameters (\alpha, \beta) of the 5 parameter loss surface proposed in the paper:

$$L(N,D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

The math of Approach 2 works by first assuming parabolic approximations to isocontours of the loss surface defined by equal amounts of compute. Compute isn’t a parameter of the loss surface itself, but it is used to constrain the relationship between parameter count (N) and token count (D) through the approximation C=6ND. Like I mentioned above, our initial application of this approach outside of text was in DNA modeling and the tiny vocabularies in those models end up breaking this assumption – I’ll just ignore that for now and keep going. This constraint then gives the formula for an IsoFLOP curve as a univariate function of D (or N) alone:

$$L(D;\,C) = E + A\!\left(\frac{6D}{C}\right)^{\!\alpha} + \frac{B}{D^\beta}$$

Aside from some impractical cases like \=2,=−1, this is not a parabola and parabolas are only as accurate at estimating this function near critical points as a 2nd order Taylor approximation. So that immediately raises a number of questions like:

1. How close to the true curve minima do I need to sample loss values in my IsoFLOP sweep grid for the Taylor approximation to be accurate?

2. How sensitive is this approximation to big disparities between token and model scaling? The underlying IsoFLOP formula isn’t symmetric like parabolas are when . So how does approximation accuracy vary as a function of the difference between those two exponents?

3. What happens if the true minimum isn’t centered within the token count grid I chose for each FLOP budget? If I train far more models on one side of the minimum than the other at a given compute budget, or outright miss it entirely, what does that mean for the inferred parabola vertex?

4. Does the accuracy of the approximation even matter? The minima implied by the parabola vertices might be wrong, but if they’re wrong in a consistent way then are the results from Approach 2 still valid?

5. Do these kinds of potential problems come up in real scaling law studies? 

The paper tries to answer these questions, and shows that it’s easy to come up with ways that Approach 2 breaks down in controlled simulations. It does this by moving around the middle of IsoFLOP grids, making those grids wider or skinnier, changing the token/param scaling ratios, and either increasing sampling noise or decreasing that noise to zero. All of these result in some kind of measurable Approach 2 error. 

As a very concrete and potentially surprising example of one of these issues, here is an [Interactive Approach 2 Bias Application](https://claude.ai/public/artifacts/ead15f94-b2b3-4dd8-b7a2-daf1d113e214) showing how plugging in the published Chinchilla loss surface parameters and fitting data drawn from it with Approach 2 results in extrapolation errors. Personally, I would have never guessed that coming into this work. The errors in this case result from the Taylor approximation I mentioned, and you can see how that gets worse in the app by increasing the sampling grid width.

So even if finding these errors is not hard, do they actually matter? Real experiments include a mixture of them and they don’t vary in the convenient, predictable ways that our simulations assume. Most of the rest of the paper is about characterizing and measuring those implications.

## Implications

The Approach 2 biases we found can cancel or compound across compute budgets in real IsoFLOP experiments. Their net effect might even come close to zero in some cases, but this would only be true in expectation if the underlying misconfigurations in IsoFLOP experiment design are random rather than systematic. 

One way to get a sense for how often these issues occur in IsoFLOP experiments is to just look at a bunch of them. We catalogued \~15 [examples](https://docs.google.com/document/d/1Ab3cYJJYu7t_99Idj9gpHv2icZmRTPBJ_brJxsgnVA4/edit?usp=sharing) of published IsoFLOP curves to show, at least qualitatively, that some sources of systematic error are easy to spot (namely uncentered and/or shifting parabola vertices). Here are a few examples:

![Published IsoFLOP Examples][image1]

A better way to measure the implications of Approach 2 biases is to get real IsoFLOP data, fit it with Approach 2, fit it with some other unbiased (or at least less biased) method, and then compare the results. Here is what that looks like for Llama 3:

![Misallocation Errors][image2]

This shows how compute-optimal allocation forecasts differ between Approach 2 and “Approach 3”, a standard, parametric method for fitting scaling laws that also comes from the Chinchilla paper. The difference is expressed as a percentage of the overall pretraining compute budget, and that works out to roughly 6.5% (“DCL %” in the figure). That’s a significant error at this scale, worth \>$1M at standard GPU-hour valuation, and it’s particularly notable given how forgiving symmetric loss surfaces are to Approach 2 biases.  The “b/a” column shows how scaling asymmetry is minimal within the Llama 3 pretraining data, where 1 indicates perfect symmetry. This suggests that most of the error likely comes from the way the IsoFLOP sweep was configured.

We further substantiated that claim by coming up with an IsoFLOP data quality control pipeline designed to compensate for some Approach 2 biases. Applying this pipeline quickly leads to convergent estimates across fitting methods:

![Approach 2 Progressive Filtering][image3]

In the end, there is no perfect way to estimate scaling laws, and using a method like Approach 3 as a ground-truth for comparison is not bulletproof evidence of the extent to which Approach 2 can be wrong. In other words, Approach 3 does not necessarily provide unbiased inference either, but the evidence presented over simulated data, real data, and in residual analysis, strongly suggests that it’s much closer to correct in practice.

## Alternatives 

The obvious alternative to Approach 2 is Approach 3\. Like I mentioned above, this method has the advantage of providing something far closer to unbiased inference. Some less obvious things about it are:

1. It is neither unstable, sensitive to initialization, nor prone to local minima with the right reparameterization.  
2. You actually need **less** data than Approach 2 to fit it, and our paper covers that in an argument on statistical efficiency.  
3. You can apply it to the same experimental data as Approach 2, unlike Chinchilla Approach 1\.  
4. You don’t necessarily have to run IsoFLOP experiments for Approach 3\.  
5. Approach 3 involves fitting far fewer parameters, which may not be an obvious difference to anyone that hasn’t thought much about the methods (and to say nothing about *effective* parameters fit).  
6. It can be just as simple to implement as Approach 2\.

That first point is likely the most heterodox, so I’ll do my best to justify it. The 5 parameter Chinchilla loss surface paired with any typical objective function (Huber or MSE usually) gives a non-convex optimization. This means that local minima are hard to avoid unless you can search the optimization space extensively. This is why most applications of Approach 3 in practice frequently run optimizations multiple times from different starting points, or use some kind of grid search to find more optimal starting points.

Our paper proposes a reparameterization based on Variable Projection that flattens the 5D optimization into two dimensions. This doesn’t magically solve any problems related to non-convexity, but it makes super dense grid search very practical. You could even exhaustively search over the two parameters in the optimization, the (,) exponents, down to some desired level of decimal precision. This gets impractical beyond 4-5 sigfigs, so you could also do this to slightly lower precision and seed an optimization from there.  Or you could do this in [pure javascript](https://gist.github.com/eric-czech/ebd9a80d58c7b5e9c40ba390ff884617) with no linear algebra or optimization libraries in a few seconds and \~70 lines of code (to 1e-3 precision). I’m not sure why you’d want to do it that way – it makes my other point about this not necessarily being complicated to implement though. Better implementations can be found in our repo for the project at [https://github.com/Open-Athena/vpnls](https://github.com/Open-Athena/vpnls).

## Conclusion  

I’m not entirely sure it was worth doing all this work to identify weaknesses in Approach 2. I was originally using it to fit a basic scaling law on some DNA data, and staring at enough wonky parabola fits ultimately got me thinking harder about how that method really works. If I had known that I’d eventually end up with an alternative that has many advantages on top of simply being more standard, I probably would have switched methods and called it a day. It just didn’t work out that way – the advantages were never clear at the start and they rolled out in a very nonlinear fashion.

Hopefully, there is something in between these simulations, proofs, the IsoFLOP data we gathered and curated, the reparameterization we proposed, and the literature we synthesized that will help nudge the space of scaling law modeling towards better standardization. In our paper, we gave more than 10 examples of analytical extensions to Chinchilla laws that have already been published. Many of these are very recent too, so I think it’s fair to say that this is a rapidly evolving space and that, much like the original Kaplan and Hoffmann scaling laws, its potential for long-term impact is high. I think it's also fair to say that the space would benefit a lot from more efficient, cohesive strategies for fitting experimental data.


[image1]: <data:image/png;base64,...>
[image2]: <data:image/png;base64,...>
[image3]: <data:image/png;base64,...>