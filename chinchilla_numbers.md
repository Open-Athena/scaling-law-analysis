#### Chinchilla Published Numbers

Olmo Hybrid: From Theory to Practice
- https://allenai.org/papers/olmo-hybrid
- pretraining data: Dolma 3
- Scaling law numbers in Figure 8 (method=Approach 3):
- Transformer: α = 0.252, β = 0.213
  - a=0.213/(0.213+0.252)=0.4580645161
  - b=0.252/(0.213+0.252)=0.5419354839
- Hybrid: α = 0.226, β = 0.219
  - a=0.219/(0.219 + 0.226)=0.4921348315
  - b=0.226/(0.219 + 0.226)=0.5078651685
- Pure GDN: α = 0.183, β = 0.227
  - a=0.227/(0.183 + 0.227)=0.5536585366
  - b=0.183/(0.183 + 0.227)=0.4463414634

* [Scaling Unlocks Broader Generation and Deeper Functional Understanding of Proteins](https://www.biorxiv.org/content/10.1101/2025.04.15.649055v2) (Oct 2025\)  
  * https://www.biorxiv.org/content/10.1101/2025.04.15.649055v2  
  * ProGen3  
  * Trains on 1.5T amino acids from Profluent Protein Atlas v1  
  * Results (method=Kaplan)  
    * They only report beta/alpha = 1.479
* [Scaling Open Discrete Audio Foundation Models with Interleaved Semantic, Acoustic, and Text Tokens](https://arxiv.org/abs/2602.16687)  
  * https://arxiv.org/abs/2602.16687  
  * Affiliations:  
    * Orgs: OpenAthena, SCB 10X  
    * Univ: Stanford, University of Southern California, University of Cambridge  
  * Scaling laws on Audio and text data (mixed)  
  * Figure 4 (method=approach 2):  
    * a=0.367, b=0.579  
* [Scaling Laws For Dense Retrieval](https://arxiv.org/pdf/2403.18684)  
  * https://arxiv.org/abs/2403.18684  
  * Xiaohongshu Inc, Tsinghua University  
  * Scaling laws for Transformers on query-document pairs  
  * Results from 4.4 Model-Data Joint Laws (method=Kaplan)  
    * alpha=0.56, beta=1.31 --> beta/alpha = 2.3393
* [Sequence modeling and design from molecular to genome scale with Evo](https://www.biorxiv.org/content/10.1101/2024.02.27.582234v2)  
  * https://www.biorxiv.org/content/10.1101/2024.02.27.582234v2  
  * Figure S5 data extracted in [https://github.com/eric-czech/evo-scaling-law-extraction](https://github.com/eric-czech/evo-scaling-law-extraction)  
  * Results (method=approach 2):  
    * Transformer++: a=0.552, b=0.551  
    * Mamba: a=0.388, b=0.487  
    * Hyena: a=0.504, b=0.499  
    * StripedHyena: a=0.483, b=0.539  
* [Training Compute-Optimal Protein Language Models](https://www.biorxiv.org/content/10.1101/2024.06.06.597716v1)  
  * https://www.biorxiv.org/content/10.1101/2024.06.06.597716v1  
  * BioMap Research, Tsinghua, MBZUAI  
  * Uses 194B tokens of protein sequence from UniRef and ColabFold  
  * Results (method=approach 2):  
    * CLM: a=0.578, b=0.422  
    * MLM: a=0.776, b=0.230  
* See [Scaling Laws for Native Multimodal Models](https://arxiv.org/abs/2504.07951) Table 2  
  * https://arxiv.org/abs/2504.07951  
  * Apple, Sorbonne University  
  * This is notable for using approach 3 for early fusion and a different approach in late-fusion models.  They do the scaling fits both empirically (similar to what I did with PlantCAD) and derive new closed form solutions too:  
  * ![][image1]  
  * From Table 2 (method=approach 3\)  
    * NMM (early fusion) average: a=0.5262, b=0.473  
    * NMM (late fusion): a=0.6358, b=0.4619  
    * Sparse NMM (early-fusion): a=.361, b=0.656  
* See [DeepSeek LLM Scaling Open-Source Language Models with Longtermism](https://arxiv.org/pdf/2401.02954)   
  * https://arxiv.org/abs/2401.02954  
  * Table 4 (method=approach 2\)  
    * OpenAI (OpenWebText2) a=0.73, b=0.27  
    * Chinchilla (MassiveText) a=0.49, b=0.51  
    * Ours (Early Data) a=0.450, b=0.550  
    * Ours (Current Data) a=0.524, b=0.476  
    * Ours (OpenWebText2) a=0.578, b=0.422  
* [Exploring Scaling Laws for EHR Foundation Models](https://arxiv.org/pdf/2505.22964)  
  * https://arxiv.org/abs/2505.22964  
  * Microsoft Research, University of Southern California  
  * Data is tokens from EHR records  
  * Figure 1 (method=approach 2): a=.58, b=.44  
* [Scaling Laws for Imitation Learning in Single-Agent Games](https://arxiv.org/pdf/2307.09423) (method=approach 2\)  
  * https://arxiv.org/abs/2307.09423  
  * Amazon, Princeton, Harvard, UPenn  
  * Reinforcement learning trained on single-agent game states for Atari games  
    * Action spaces consist of a mixture of images and ASCII grid representations  
  * Trains CNNs and Transformer models  
  * Exponents by Atari game (method=approach 2):  
    * NetHack: a=.61, b=.39  
    * Battle Zone: a=.58, b=.51  
    * Breakout: a=.74, b=.46  
* [xLSTM Scaling Laws: Competitive Performance with Linear Time-Complexity](https://arxiv.org/abs/2510.02228)  
  * https://arxiv.org/abs/2510.02228  
  * NXAI  
  * Evaluates on DCLM  
  * Exponents by architecture (method=approach 2):  
    * Transformer: a=.575, b=.424  
    * xLSTM: a=.547, b=.417  
* [Scaling Behavior of Discrete Diffusion Language Models](https://arxiv.org/abs/2512.10858)  
  * https://arxiv.org/abs/2512.10858  
  * ETH Zurich  
  * Uses Nemotron-CC dataset  
  * From Table 1 (their best model) (method=approach 2):  
    * a=.589, b=.411  
* [Scaling Laws For Diffusion Transformers](https://arxiv.org/abs/2410.08184)  
  * https://arxiv.org/abs/2410.08184  
  * ByteDance, Shanghai Artificial Intelligence Laboratory  
  * Evaluates Diffusion Transformers on image-text pairs from [Laion-Aesthetic](https://laion.ai/blog/laion-aesthetics/) dataset  
  * “we adopt a vanilla Transformer design (Vaswani, 2017), using a concatenation of image, text, and time tokens as input to the models.“  
  * Equation 5 and 6 (method=approach 2):  
    * a=.5681, b=.4319

