
## Contents

- [Abstract](index.md)
- [Project Motivation](motivation.md)
- [Biological & Theoretical Background](background.html)
- [Model Structure](structure.md)
- [Usage](usage.md)
- [First Steps: Pyro](pyro.html)
- [Model Reconstruction](model.html)
- [Performance Comparison](performance.html)
- [Conclusions](conclusions.md)

# Biological & Theoretical Background

## A Biological Primer

A core concept of molecular biology is that a biomolecule's function is largely governed by its sequence of residues. The amino acids of a protein, or nucleotides of a ribonucleic acid (RNA), determine the molecule's structure, reactivity and specificity. As such, a major challenge that remains in biomedical research is understanding how mutations affect biomolecule function to the point that one can model and predict the effect of any particular mutation to any given protein or RNA.

For fundamental biological research, this would help elucidate the biochemical rules and constraints that govern biomolecular function. For medicine, this could provide vital information for understanding which genetic variants may be associated with disease. For biotechnology, this could provide vital aid to both the development of modified proteins with beneficial features and the construction of large molecular libraries enriched with functional sequences.

There is a clear need to be able to rapidly assess whether a given mutation will affect a biomolecule's function. However, assessing this experimentally is difficult and expensive. Technological advances have facilitated what is currently the best experimental option - high-throughput experimental assays known as [deep mutational scans](https://www.nature.com/articles/nmeth.3027), which can assess the effect of thousands of mutations on molecular, cellular or organismal fitness under selection pressure. However, this approach is very resource-intensive and still only manages to explore a fraction of a biomolecule's exponentially large sequence space, meaning that systematic exploration of all mutation combinations is not possible.

Instead, numerous computational models have been developed to tackle this problem, mostly by leveraging the fact that for millions of years, evolution has been applying a process of massively parallel mutagenesis and selection that can be thought of as a deep mutational scan. By analysing the natural sequence variation of biomolecule families, [one can infer functional and structural information](https://www.nature.com/articles/nbt.2419) for the biomolecule. 

This can also be applied to mutation effect prediction. The majority of previous models overcome the vast combinatorial complexity of sequence interactions by considering residues independently or only in the context of pairwise interactions. In the years preceding this project, the Marks lab took a different approach; instead of explicitly modelling interactions through unique free parameters, they tried implicitly capturing these interactions with latent variables; more specifically, capturing the problem's complexity with approximate inference using a bayesian deep latent-variable model, which is discussed below.

* * *

## Statistical Model Overview

The original paper can be found [here](https://www.nature.com/articles/s41592-018-0138-4), whilst the pre-print version can be found [here](https://arxiv.org/abs/1712.06527). For related work by the Marks lab, see [here](https://marks.hms.harvard.edu/publications.html).

### Summary

The model constructed, in its full form, is a Bayesian Variational Autoencoder (B-VAE) that models both the latent dimension (the encoder's output) and the decoder's parameters as variational parameters sampled from a gaussian distribution. Augmenting this variational approximation architecture, the final layer of the decoder is subject to a width-one convolution, a structured sparsity prior, and a final global temperature parameter. The non-Bayesian, standard VAE version of the model is also subject to dropout and $l2$ regularisation.

The input to the model is a dataset of sequence alignments from one protein family; a focus sequence from one species, and a set of sufficiently similar, evolutionarily related sequences from other species (see model structure for more information).

The model works by learning a variational approximation of a sequence family's latent variable distribution, $z$ given $x$. From this approximation, it can generate an estimate of the conditional distribution of $x$ given $z$. By learning both, the model can use an ELBO approximation of $p(x|\theta)$, defined as the probability of x given the parameters, $\theta$, of x's generative process from $z$. One can define $p(x|\theta)$ as a heuristic for evaluating the probability of observing a sequence given the family's generative process parameters, $\theta$ - and as such, one can take the log-ratio of such probabilities (or, more accurately, the ELBO approximations) between a mutant sequence and wild type sequence as a heuristic for the fitness of that mutant sequence.

### Heuristic

Probabilistic latent-variable models such as variational autoencoders reveal structure hidden in data by first positing that the data is drawn from a hidden generative process before then using inference to learn the parameters of this process's distribution. Here, the data, $x$, is the set of sequences alignments for a particular biomolecule family. The latent variable model posits that this data is generated from some hidden distribution, $z$, (of dimension $D$) and the evolutionary process of generating $x$ from $z$ is parameterized by $\theta$, such that:

$$ z \sim \mathcal{N}(0,I_D) $$

$$ x \sim p(x|z,\theta) $$

In our case, we assume this latent distribution $z$ from which $x$ is generated is a multivariate standard normal, and what we are interested in finding is $\theta$, the parameters of the hidden generative process that maps $z$ to $x$. Once we learn the $\theta$ that fits $p(x|\theta)$ to the observed data, we can consider the following ratio:

$$\log\frac{p(x^{\text{Mutant}}|\theta)}{p(x^{\text{Wild-type}}|\theta)}$$

As a heuristic for determining the relative favorability of the mutant sequence $x^{\text{Mutant}}$ against the wild-type sequence $x^{\text{Wild-type}}$. This can be alternatively expressed as: given the evolutionary parameters, $\theta$, that generate the observed sequences, $x$, what is the likelihood of observing this mutated sequence compared to observing the wild type sequence. The assumption behind this model's power is that a mutated sequence that is unlikely to be generated from the evolutionary process that generates the sequence family in question is also unlikely to be as functional or fit as the wild-type sequence. In a sense, the generative process defined by $\theta$ governs the structure and constraints of what defines this sequence family - mutated sequences that violate this process are unlikely to be viable sequences.

### Basic Model


Since $z$ is hidden (we do not know a priori which $z$ variables are responsible for which $x$ variables), we must model $p(x|\theta)$ through a marginal likelihood:

$$ p(x|\theta) = \int p(x|z,\theta)p(z)dz $$

Direct computation of this is intractable. The trick common to all variational autoencoders is to instead deal with this integral's lower bound, the Evidence Lower Bound (ELBO), $\mathcal{L}(\phi)$ given as:

$$\mathcal{L}(\phi)=\mathbb{E}_q[\log p(x|z,\theta)]-D_{KL}[q(z|x,\phi)||p(z)]$$

Where $q(z|x,\phi)$ is the variational approximation of hidden variables given the observed variables. This technique is integral to all VAEs. In general, maximising the ELBO can intuitively be thought of as minimizing  the reconstruction loss of the autoencoder, whilst also minimising the Kullback-Leibler distance between the variational approximation of the distribution of $z$ given $x$ with the prior distribution of $z$.

Using neural networks, we can model both $q(z|x,\phi)$ and $p(x|z,\theta)$ - the model's encoder takes in x and learns $q(z|x,\phi)$, the model's decoder samples from $q(z|x,\phi)$ and learns $p(x|z,\theta)$ through the following calculation:

$$p(x_i = a|z) = \frac{e^{f(z_a^i)}}{\sum_be^{f(z_b^i)}} \qquad i = 1, ..., L$$

Where $x_i$ is the $i$th element of sequence $x$ of length $L$, and $f(z)$ represents the non-linear neural network function approximation of the function parameterized by $\theta$. It is important to note that each element of the sequence, $x_i$, is conditionally independent of all other element given $z$; in other words, correlations between elements are mediated only by the latent variables.

Since the encoder learns $q(z|x,\phi)$ and the decoder learns $p(x|z,\theta)$, we can produce the ELBO for log-ratio approximations.

### Doubly-Variational Model

An important additional feature of this model is that variational approximations are made for both $z$ and $\theta$. In this sense, the integral we hope to evaluate:

$$\log p(X) = \log\iint p(X|,Z,\theta)p(Z)p(\theta)dZd\theta$$

A fuller derivation from this integral to the ELBO can be found in the [original paper describing the model](https://www-nature-com.ezp-prod1.hul.harvard.edu/articles/s41592-018-0138-4.pdf). With this addition, the ELBO is instead given as:

$$\mathcal{L}(\phi) = N\mathbb{E}_{x\in X}\big[\mathbb{E}_q[\log p(x|z,\theta)]-D_{KL}[q(z|x,\phi)||p(z)]\big] - \sum_i D_{KL}(q(\theta_i)||p(\theta_i)) $$

Where N is the effective number of sequences (see model structure for information on the data-reweighting scheme), $q(\theta)$ is the variational approximation and $p(\theta)$ is the prior, which is again the standard normal distribution.

In simple terms, the KL divergence of the $\theta$ parameters is included alongside the original KL divergence term. However, because the original ELBO is defined per-sequence and these parameters are defined across-sequences, the original ELBO term must be weighted by the effective number of sequences.

### Additional Parameterisation

Biologically-motivated structures are built into the final layer of the model's decoder. First, a group sparsity prior is applied to favor sparsely interacting subsystems. This sparsity prior is a priori logit-normally distributed set of parameters that can implement sparsity through a number of different routes, all based on techniques used for sparse Bayesian learning:

- Using a continuous relaxation of a spike and slab prior with a logit normit scale distribution (this is the default option used in all performance evaluation).
- Using a moment-matched Gaussian approximation to the log-space Hyperbolic Secant hyperprior of the Horseshoe.
- Using a Horseshoe prior approach
- Using Laplace sparsity with exponential hyperprior
- Using Automatic Relevance Determination sparsity with an inverse-gamma hyperprior.

Each technique essentially works by encouraging sparse interactions (i.e. allowing parameters to equal or almost equal zero). In doing so, they discourage overly complex models and, it is hoped, allow the model to better reflect reality. Interactions should not only be sparse but also local; the key idea behind achieving this is that the sparsity parameters learnt through these techniques are tiled over the final layer a number of times so as to encourage local interactions within tilings rather than between tilings. The final output is also subject to a width-1 convolution, which it is hoped captures correlations in amino acid usage. Additionally, there is a global temperature parameter, $\lambda$, applied to the model's output as follows:

$$x_{\text{after}}=log(1+e^\lambda)x_{\text{before}}$$

The log-exponential formulation coerces this parameter to be positive. This temperature parameter serves to capture the strength of the above constraints, the hope being that this added feature will capture global, sequence-wide correlations in the selective strength.

In the Bayesian model, these parameters are all also subject to variational approximation. For the model structure without variational approximation, dropout and l2-regularisation are also used.
