# DDPM: the noise-prediction loss and the loop that trains on it

**Source:** Jonathan Ho, Ajay Jain, Pieter Abbeel,
[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239v2),
v2 (16 Dec 2020), §3.4, equation (14), and Algorithm 1 (printed at the top of
page 4, inside §3.2). Other locations used: §1, §2 equations (1)–(7), §3.1,
§3.2 equations (10)–(12), §3.3 equation (13), §4, §4.1 Table 1, §4.2 Table 2,
§4.3, §5,
Algorithm 2, Appendices B and C.
Official code: [hojonathanho/diffusion at commit `1e0dceb`](https://github.com/hojonathanho/diffusion/tree/1e0dceb3b3495bbe19116a5e1b3596cd0706c543)
(2020-06-22 UTC). The default branch, `master`, has two commits and the
repository has no tags; its other branch, `gh-pages`, holds only the project
web page. So `1e0dceb` is the latest code commit before v2.

## What it computes and what it changes

One training step takes a clean training image, picks a random noise level
$t$, adds exactly that much Gaussian (normal, bell-curve) noise in one shot,
and asks the network to output the noise it added. The loss is the squared
error between the true noise and the predicted noise. Equation (14) names the
average of that error $L_\mathrm{simple}$, and Algorithm 1 is the loop that
minimizes it by gradient descent. A network that predicts the noise at every
level can then be run backwards from pure noise to an image. Algorithm 2
(not otherwise covered here) calls the network $T$ times, for
$t=T,\dots,1$, where $T$ is the number of noise levels (1000 in the paper).
Each call removes a scaled copy of the predicted noise from the current noisy
image $\mathbf x_t$, rescales the result and, at every step except the last,
adds fresh Gaussian noise (lines 2–4).

The paper's first loss, eq. (3), is an upper bound on the negative
log-likelihood (NLL) $-\log p_\theta(\mathbf x_0)$ of the data, called the
_variational bound_. It treats $\mathbf x_1,\dots,\mathbf x_T$ as hidden
variables ("latents", §2):
$p_\theta(\mathbf x_0):=\int p_\theta(\mathbf x_{0:T})\,d\mathbf x_{1:T}$ (§2)
is the model's probability of the image with them integrated out. The bound
averages over them as the noising chain $q(\mathbf x_{1:T}\vert\mathbf x_0)$
draws them. A distribution of the hidden variables given the data is called a
_posterior_. A variational autoencoder (VAE) learns an approximation to the
model's own posterior $p_\theta(\mathbf z\vert\mathbf x)$ (hidden $\mathbf z$,
data $\mathbf x$), a "recognition model $q_\phi(\mathbf z\vert\mathbf x)$",
also called a "probabilistic encoder", whose parameters $\phi$ are learned
"jointly with the generative model parameters $\theta$" (Kingma and Welling,
[arXiv 1312.6114v10](https://arxiv.org/abs/1312.6114v10), §2.1). Here the
noising chain does that job, and the paper calls it "the approximate
posterior" (§2). Unlike the encoder it has no learnable parameters (§3.1). The
VAE paper calls its $p_\theta(\mathbf x)$, the counterpart of
$p_\theta(\mathbf x_0)$, "the marginal likelihood" (1312.6114v10, §2.1). So
eq. (3), "the usual variational bound on negative log likelihood" (§2), is the
VAE's "(variational) lower bound on the marginal likelihood" (1312.6114v10,
§2.2), also called the evidence lower bound (ELBO), with its sign flipped (my
comparison). Eq. (5) splits it into one term per timestep.

The terms depend on the noise schedule. Noising step $t$ adds Gaussian noise
of variance $\beta_t$ and scales the image by $\sqrt{\alpha_t}$, where
$\alpha_t=1-\beta_t$, and $\bar\alpha_t=\alpha_1\alpha_2\cdots\alpha_t$ is
the product of the first $t$ of them (§2, eq. (2); see the Symbols table).
For $t>1$, eq. (12) (§3.2) writes each term, up to a constant that does not
involve $\theta$, as the same squared error multiplied by a weight
$\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}$ that depends on $t$.
Here $\sigma_t^2$ is the variance of the denoising step
$p_\theta(\mathbf x_{t-1}\vert\mathbf x_t)$, set to "untrained time dependent
constants" (§3.2). §3.2 states its choices "for $1<t\le T$" and tries
$\sigma_t^2=\beta_t$ and
$\sigma_t^2=\tilde\beta_t:=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$
(eq. (7)). The paper gives no value for $\sigma_1^2$; §3.4 says only that the
$t=1$ case of (14) ignores it (I searched every $\sigma$ in the PDF text and
TeX source, appendices included). At $t=1$ the second formula would give
$\tilde\beta_1=0$, because $\bar\alpha_0$, an empty product, is 1 (check 7,
below); the code's $t=1$ value is under "Reverse-process variance
$\sigma_t^2$". For $t>1$ the same $\sigma_t$ scales the fresh noise that
Algorithm 2 adds, $\sigma_t\mathbf z$, where $\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I)$
is new noise with mean 0 and variance 1 in every coordinate (lines 3–4).
With $\sigma_t^2=\beta_t$ the variance of $\sigma_t\mathbf z$ is at most
$\beta_T=0.02$, against variance 1 for the pure noise $\mathbf x_T\sim\mathcal N(\mathbf 0,\mathbf I)$
that sampling starts from (line 1). Eq. (14) drops the weight of eq. (12), so
every $t>1$ counts equally; the network, its input and its target stay the
same. The baseline in Table 2
has the network predict $\tilde{\boldsymbol\mu}_t$ instead of the noise: the
mean of $\mathbf x_{t-1}$ given $\mathbf x_t$ and $\mathbf x_0$, "the forward
process posterior mean" (§3.2; eqs. (6)–(7)).

At $t=1$ the bound's term is
$L_0=-\log p_\theta(\mathbf x_0\vert\mathbf x_1)$, the _decoder_ of eq. (13):
the last denoising step, which gives each pixel the Gaussian probability of
its bin $[x-\frac1{255},x+\frac1{255}]$, with the bins at $\pm1$ running out
to $\pm\infty$ (§3.3). Eq. (14) uses the same squared error there too, which
matches $L_0$ only approximately; "The $t=1$ approximation", after check 6
below, explains how.

## The equation

$$
L_\mathrm{simple}(\theta) := \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}}\Big[\big\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_\theta\big(\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon},\,t\big)\big\|^2\Big] \tag{14}
$$

Algorithm 1 (Training), as printed:

```text
1: repeat
2:   x_0 ∼ q(x_0)
3:   t ∼ Uniform({1, …, T})
4:   ε ∼ N(0, I)
5:   Take gradient descent step on
         ∇_θ ‖ε − ε_θ(√ᾱ_t x_0 + √(1 − ᾱ_t) ε, t)‖²
6: until converged
```

## Symbols

| Symbol | Shape or type | Meaning | Role | Defined at |
| ------ | ------------- | ------- | ---- | ---------- |
| $L_\mathrm{simple}(\theta)$ | one number | the training loss | computed intermediate (the objective) | Eq. (14) itself, introduced with "we found it beneficial to sample quality (and simpler to implement) to train on the following variant of the variational bound" (§3.4) |
| $\mathbf x_0$ | vector of $D$ numbers in $[-1,1]$; for CIFAR10 a $32\times32\times3$ image, $D=3072$ (check 1) | one clean training image | input data | "$\mathbf x_1,\dots,\mathbf x_T$ are latents of the same dimensionality as the data $\mathbf x_0\sim q(\mathbf x_0)$" (§2); "We assume that image data consists of integers in $\{0, 1, \dots, 255\}$ scaled linearly to $[-1, 1]$" (§3.3); "where $D$ is the data dimensionality" (§3.3, after eq. (13)); "the $32\times32\times3$ or $256\times256\times3$ images in our experiments" (§4.3) |
| $q(\mathbf x_0)$ | distribution over images | the training data | input data | the §2 sentence quoted for $\mathbf x_0$; Algorithm 1, line 2 |
| $t$ | integer in $\{1,\dots,T\}$ | which noise level this sample uses; larger $t$ means more noise | random sample | "where $t$ is uniform between 1 and $T$" (§3.4); Algorithm 1, line 3 |
| $T$ | integer, 1000 | number of noise levels (diffusion steps) | fixed setting | "We set $T = 1000$ for all experiments" (§4) |
| $\boldsymbol\epsilon$ | $D$-vector, same shape as $\mathbf x_0$ | Gaussian noise. The same draw is mixed into the network's input and used as the target | random sample | "reparameterizing Eq. (4) as $\mathbf x_t(\mathbf x_0,\boldsymbol\epsilon)=\sqrt{\bar\alpha_t}\mathbf x_0+\sqrt{1-\bar\alpha_t}\boldsymbol\epsilon$ for $\boldsymbol\epsilon\sim\mathcal N(\mathbf 0,\mathbf I)$" (§3.2) |
| $\sqrt{\bar\alpha_t}\mathbf x_0+\sqrt{1-\bar\alpha_t}\boldsymbol\epsilon$, called $\mathbf x_t$ | $D$-vector | the noisy image at level $t$, made in one step | computed intermediate | the §3.2 sentence quoted for $\boldsymbol\epsilon$. It is a draw from eq. (4), "$q(\mathbf x_t\vert\mathbf x_0)=\mathcal N(\mathbf x_t;\sqrt{\bar\alpha_t}\mathbf x_0,(1-\bar\alpha_t)\mathbf I)$" (§2) |
| $\bar\alpha_t$ | number in $(0,1)$; a table of $T$ values | the share of the clean image's variance that is left after $t$ noising steps. The bar marks a running product, not a mean | fixed setting (precomputed) | "using the notation $\alpha_t := 1-\beta_t$ and $\bar\alpha_t := \prod_{s=1}^t\alpha_s$" (§2, before eq. (4)) |
| $\beta_t$ (inside $\bar\alpha_t$) | number; $T$ values | variance of the noise that step $t$ adds | fixed setting | "according to a variance schedule $\beta_1,\dots,\beta_T$" (§2); "We ignore the fact that the forward process variances $\beta_t$ are learnable by reparameterization and instead fix them to constants" (§3.1); "We set the forward process variances to constants increasing linearly from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$" (§4) |
| $\boldsymbol\epsilon_\theta(\cdot,t)$ | function: ($D$-vector, integer $t$) → $D$-vector | the network that predicts the noise. The subscript $\theta$ is a label (this function has the weights $\theta$), not an index. Eq. (11)'s middle expression and §4.3 write $\boldsymbol\epsilon_\theta(\mathbf x_t)$ without $t$; eq. (14) and Algorithm 1 pass $t$ | the network whose weights are trained | "where $\boldsymbol\epsilon_\theta$ is a function approximator intended to predict $\boldsymbol\epsilon$ from $\mathbf x_t$" (§3.2, after eq. (11)); "we use a U-Net backbone similar to an unmasked PixelCNN++" and "Parameters are shared across time, which is specified to the network using the Transformer sinusoidal position embedding" (§4) |
| subscript $t,\mathbf x_0,\boldsymbol\epsilon$ under $\mathbb E$ | names three random draws | the average runs over $t$, $\mathbf x_0$ and $\boldsymbol\epsilon$, each drawn independently | random sample | $t$: the §3.4 quote above. $\mathbf x_0$ and $\boldsymbol\epsilon$: Algorithm 1, lines 2 and 4, "$\mathbf x_0\sim q(\mathbf x_0)$" and "$\boldsymbol\epsilon\sim\mathcal N(\mathbf 0,\mathbf I)$" |

**Not defined in the paper:**

- $\theta$. No sentence says what $\theta$ is. The paper only uses it: "is
  clearly differentiable with respect to $\theta$" (§3.4), "where $C$ is a
  constant that does not depend on $\theta$" (§3.2), and $\nabla_\theta$ in
  Algorithm 1. Appendix B gives parameter counts ("Our CIFAR10 model has 35.7
  million parameters"). This reading takes $\theta$ to be the U-Net's
  weights. That is an inference from these uses. I looked in §2, §3.1–§3.4,
  §4 and Appendix B.
- $\Vert\cdot\Vert^2$. Not defined. The reading takes it as the sum of the
  squared entries over the $D$ coordinates, because §4.3 computes "the root
  mean squared error $\sqrt{\Vert\mathbf x_0-\hat{\mathbf x}_0\Vert^2/D}$". Dividing
  by $D$ gives a mean only if $\Vert\cdot\Vert^2$ is a sum over the $D$
  coordinates. §4.2 calls a loss "akin to Eq. (14)" an "unweighted mean
  squared error"; that is the paper's name for the loss and does not settle
  sum against mean, and a mean only rescales the loss by $1/D$. The official
  code takes a mean instead of a sum (see "What
  the code must decide").
- $\mathcal N(\mathbf 0,\mathbf I)$, $\mathbf 0$ and $\mathbf I$. Not defined.
  The reading takes them in their standard sense: a Gaussian with mean zero and
  identity covariance, which means every coordinate is an independent normal
  number with mean 0 and variance 1. The paper's own use agrees with this:
  §3.2 says $\sigma_t^2=\beta_t$ "is optimal for $\mathbf x_0\sim\mathcal N(\mathbf 0,\mathbf I)$"
  and connects the two choices to "data with coordinatewise unit variance".
- $\sim$ (Algorithm 1, lines 2–4). Not defined. The reading takes $a\sim p$
  as "$a$ is drawn from the distribution $p$". Every use I found in the PDF
  text has a value on the left and a distribution on the right, as in
  "$\mathbf x_0\sim q(\mathbf x_0)$" (§2); none means "roughly".
- "converged" (Algorithm 1, line 6). There is no stopping rule. Appendix B
  gives a fixed number of steps instead ("10.6 hours to train to completion at
  800k steps" for CIFAR10), and reports scores from a checkpoint chosen
  during training (see "What the paper claims"). I searched the PDF text for
  "converge": apart from this line, it appears only in reference titles.

## Read from the inside out

1. $\bar\alpha_t$: a table lookup. Set $\alpha_s=1-\beta_s$ and multiply
   $\alpha_1\alpha_2\cdots\alpha_t$. The product's index $s$ runs from 1 to
   $t$. The result is one number. With the §4 schedule, $\bar\alpha_1=0.9999$
   and $\bar\alpha_{1000}\approx4.0\times10^{-5}$ (check output below), so
   almost no signal is left at $t=T$. In NumPy this is
   `abar = np.cumprod(1 - betas)`.
2. $\sqrt{\bar\alpha_t}\,\mathbf x_0$ and $\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon$:
   each multiplies a $D$-vector by a number and gives a $D$-vector.
3. Their sum is $\mathbf x_t$, a $D$-vector with the shape of the image. Eq.
   (2) adds noise one step at a time,
   $\mathbf x_t=\sqrt{1-\beta_t}\,\mathbf x_{t-1}+\sqrt{\beta_t}\cdot$(new
   noise). After $t$ such steps, the clean image is scaled by
   $\sqrt{\alpha_1}\cdots\sqrt{\alpha_t}=\sqrt{\bar\alpha_t}$. Each step also
   multiplies the variance of the noise gathered so far by $\alpha_t$ and
   adds new noise of variance $\beta_t$, drawn independently of it. Variances
   of independent noises add, and a sum of independent Gaussians is Gaussian,
   so the noise variance follows $v_t=\alpha_tv_{t-1}+\beta_t$ with $v_0=0$,
   which gives $v_t=1-\bar\alpha_t$, since
   $\alpha_t(1-\bar\alpha_{t-1})+(1-\alpha_t)=1-\bar\alpha_t$ (my derivation).
   That is eq. (4), so Algorithm 1 can jump straight to step $t$ without running the
   chain. Check 4 below confirms this for all 1000 steps. The two squared
   coefficients add to 1, $\bar\alpha_t+(1-\bar\alpha_t)=1$, so if each
   coordinate of $\mathbf x_0$ has variance 1 and is independent of
   $\boldsymbol\epsilon$, each coordinate of $\mathbf x_t$ has variance 1 at
   every $t$ (my derivation; check 4). Appendix C, item 2, gives this as the
   purpose of the $\sqrt{1-\beta_t}$ factor: "so that variance does not grow
   when adding noise".
4. $\boldsymbol\epsilon_\theta(\mathbf x_t,t)$: the network takes the noisy
   image and the integer $t$ and returns a $D$-vector, the same shape as
   $\boldsymbol\epsilon$. Appendix B: "Diffusion time $t$ is specified by
   adding the Transformer sinusoidal position embedding [60] into each residual
   block".
5. $\boldsymbol\epsilon-\boldsymbol\epsilon_\theta(\dots)$: subtract two
   $D$-vectors entry by entry; the result is a $D$-vector.
6. $\Vert\cdot\Vert^2$: the superscript 2 is a power, applied to the norm.
   Square each of the $D$ entries and add them up. The result is one number,
   at least 0, for each sample.
7. $\mathbb E_{t,\mathbf x_0,\boldsymbol\epsilon}[\cdot]$: the average over
   the three draws. $t$ is uniform on $\{1,\dots,T\}$ (each value has
   probability $1/T$), $\mathbf x_0$ is a training image, and
   $\boldsymbol\epsilon$ is a new standard-normal $D$-vector. The result is
   one number that depends only on $\theta$. Because $t$ is uniform, this
   equals $\frac1T\sum_{t=1}^T\mathbb E_{\mathbf x_0,\boldsymbol\epsilon}[\cdot]$:
   the bound's sum over timesteps, with every term given the same weight (my
   rewriting).
8. $:=$ defines the name $L_\mathrm{simple}(\theta)$. The line contains no
   inequality.

Algorithm 1 adds the order of operations and the update:

- Lines 2–4 draw one $\mathbf x_0$, one $t$ and one $\boldsymbol\epsilon$.
  That gives one random sample of the average in (14), standing in for the
  average itself (a _Monte Carlo_ estimate). The code averages a batch of
  such samples instead; see "Batch and one $t$ per example".
- Line 5: $\nabla_\theta$ is the vector of partial derivatives of the
  per-sample loss with respect to every network weight, so it has the same
  shape as $\theta$. Only $\boldsymbol\epsilon_\theta$ depends on $\theta$.
  For this step, $\mathbf x_0$, $t$, $\boldsymbol\epsilon$ and $\bar\alpha_t$
  are constants. A gradient descent step is
  $\theta\leftarrow\theta-\eta\nabla_\theta(\cdot)$. Algorithm 1 gives no step
  size $\eta$; see "Optimizer and step".
- Lines 1 and 6 repeat this "until converged". See the undefined list above.

In PyTorch, with a zero-based table `abar = torch.cumprod(1 - betas, 0)` and
a batch `x0` of `B` images, lines 2–4 and the loss of line 5 are one line:
`t = torch.randint(0, T, (B,)); eps = torch.randn_like(x0); a = abar[t].view(B, 1, 1, 1); loss = ((eps - model(a.sqrt() * x0 + (1 - a).sqrt() * eps, t)) ** 2).mean()`.
`torch.randint` excludes its upper bound, so `t` runs over
$\{0,\dots,T-1\}$, the paper's $t$ minus 1, as in the official code (see
"Timestep indexing"). `torch.randn_like` draws new standard-normal noise of
`x0`'s shape at every call, as line 4 does. `.mean()` with no dimension
averages over all elements, pixels and batch; see "Sum or mean over the $D$
pixels". These behaviours are from the PyTorch 2.15 documentation for
[`torch.cumprod`](https://docs.pytorch.org/docs/2.15/generated/torch.cumprod.html),
[`torch.randint`](https://docs.pytorch.org/docs/2.15/generated/torch.randint.html),
[`torch.randn_like`](https://docs.pytorch.org/docs/2.15/generated/torch.randn_like.html)
and [`torch.mean`](https://docs.pytorch.org/docs/2.15/generated/torch.mean.html).

**Kind of statement:** eq. (14) names (step 8) an objective, a loss to
minimize: Algorithm 1 takes gradient _descent_ steps on it. It averages over a uniform $t$, the training data and new Gaussian noise.
Algorithm 1 is an update rule. Each step changes only $\theta$; $\beta_t$,
$\bar\alpha_t$, $T$ and the dataset stay fixed.

## Where the samples come from

- $\mathbf x_0$: a fixed dataset collected before training (CIFAR10,
  CelebA-HQ, LSUN; Appendix B). Appendix B adds an
  augmentation: "We used random horizontal flips during training for
  CIFAR10", and also for every other dataset except LSUN Bedroom.
- $t$ and $\boldsymbol\epsilon$: drawn new at every step by the training loop,
  from fixed distributions. The model does not produce them.
- No copy of the model generates training data. Nothing is sampled from
  $p_\theta$ in the training loop, and the sampler (Algorithm 2) does not run
  there.
- Detached: nothing needs to be detached. The target $\boldsymbol\epsilon$ and
  the input $\mathbf x_t$ come from data and noise, not from $\theta$, so no
  gradient flows into them. The official `training_losses`
  (`diffusion_utils_2.py` lines 274–304, linked under "What the code must
  decide") has no stop-gradient.
  An exponential moving average of the weights ("We used EMA on model
  parameters with a decay factor of 0.9999", Appendix B) is kept as a second
  copy for evaluation: the code's separate "evaluation loop for use during
  training" (`run_cifar.py` line 113) builds the sampler a second time under
  the comment "EMA versions of the above" (`tpu_utils.py` lines 294–308), runs
  that "ema pass" on each saved checkpoint (lines 488–504), and skips the
  "non-ema pass" because `skip_non_ema_pass=True` (`run_cifar.py` line 129;
  `tpu_utils.py` lines 506–512). The average does not enter the loss.

## Check with small numbers

I used $D=2$ pixels and $T=2$, with $\bar\alpha_1=0.64$ and
$\bar\alpha_2=0.36$ so that every square root is exact (0.8 and 0.6). There are
two images and two noise draws, which cover both values of $t$. The stand-in
network is $\boldsymbol\epsilon_\theta(\mathbf x,t)=\theta\cdot t\cdot\mathbf x$,
with one weight $\theta=0.5$ shared across $t$, as the U-Net's weights are.
The checks cover:

- every term of (14), the batch mean, and $D=3072$ for CIFAR10;
- the gradient in Algorithm 1, line 5, and one update step;
- the average over all draws, exact and by sampling;
- the one-step noising (eq. (2) gives eq. (4));
- both cases the paper names, $t>1$ (eq. (12)) and $t=1$ (decoder, eq. (13),
  both its density × bin width approximation and the exact integral);
- the per-$t$ weights that (14) drops, for the paper's schedule.

Eq. (14) has no min, max, clip or indicator, so the only branch is $t=1$
versus $t>1$.

By hand: for sample 1 ($t=1$), $\mathbf x_1=0.8(1,0)+0.6(1,-1)=(1.4,-0.6)$,
$\boldsymbol\epsilon_\theta=0.5\cdot1\cdot\mathbf x_1=(0.7,-0.3)$, and the
error is $(0.3,-0.7)$, with squared norm $0.09+0.49=0.58$. For sample 2
($t=2$), $\mathbf x_2=0.6(0,-1)+0.8(0.5,0.5)=(0.4,-0.2)$,
$\boldsymbol\epsilon_\theta=0.5\cdot2\cdot\mathbf x_2=(0.4,-0.2)$, and the
error is $(0.1,0.7)$, with squared norm $0.01+0.49=0.50$. The asserts below
check both values.

```python
"""Small-number checks for DDPM (arXiv 2006.11239v2) eq. (14) and Algorithm 1."""

from math import erf

import numpy as np

# Toy schedule, T = 2, chosen so every square root is exact:
# abar_1 = 0.64 -> sqrt 0.8, sqrt(1 - abar) 0.6; abar_2 = 0.36 -> 0.6, 0.8.
abar = {0: 1.0, 1: 0.64, 2: 0.36}  # abar_0 = 1 is the empty product (used for t = 1)
alpha = {t: abar[t] / abar[t - 1] for t in (1, 2)}  # 0.64, 0.5625
beta = {t: 1 - alpha[t] for t in (1, 2)}  # 0.36, 0.4375
D = 2  # two "pixels"


def eps_model(x, t, theta):
    """Stand-in for the U-Net: one shared weight, told the timestep t."""
    return theta * t * x


def x_t(x0, eps, t):
    return np.sqrt(abar[t]) * x0 + np.sqrt(1 - abar[t]) * eps


def per_sample_loss(x0, eps, t, theta):
    """The term inside E[...] of eq. (14): squared norm summed over D."""
    return float(np.sum((eps - eps_model(x_t(x0, eps, t), t, theta)) ** 2))


theta = 0.5
batch = [  # (x0, eps, t): two images, two noise draws, both timesteps
    (np.array([1.0, 0.0]), np.array([1.0, -1.0]), 1),
    (np.array([0.0, -1.0]), np.array([0.5, 0.5]), 2),
]

# 1. Eq. (14) for each sample, then the batch mean.
assert np.allclose(x_t(*batch[0]), [1.4, -0.6]) and np.allclose(x_t(*batch[1]), [0.4, -0.2])
losses = [per_sample_loss(x0, e, t, theta) for x0, e, t in batch]
assert np.allclose(losses, [0.58, 0.50])
print("per-sample ||eps - eps_theta||^2:", np.round(losses, 6), " batch mean:", np.mean(losses))
# The official code averages over pixels instead of summing (nn.meanflat): 1/D of the above.
print("same with mean over the D pixels:", np.mean(losses) / D)
assert 32 * 32 * 3 == 3072  # D for a 32 x 32 x 3 CIFAR10 image (Section 4.3)

# 2. Algorithm 1, line 5: gradient with respect to theta only; x_t and eps are constants.
def grad(x0, eps, t, theta):
    xt = x_t(x0, eps, t)
    return float(-2 * t * np.sum(xt * (eps - theta * t * xt)))


g = [grad(x0, e, t, theta) for x0, e, t in batch]
h = 1e-6
fd = [(per_sample_loss(x0, e, t, theta + h) - per_sample_loss(x0, e, t, theta - h)) / (2 * h)
      for x0, e, t in batch]
assert np.allclose(g, [-1.68, 0.4]) and np.allclose(g, fd, atol=1e-6)
lr = 0.1
theta_new = theta - lr * np.mean(g)
assert np.isclose(theta_new, 0.564)
new_loss = np.mean([per_sample_loss(x0, e, t, theta_new) for x0, e, t in batch])
assert new_loss < np.mean(losses)
print("gradients:", np.round(g, 6), " theta after one step:", round(theta_new, 6),
      " batch loss after:", round(new_loss, 6))

# 3. The expectation E_{t, x0, eps}: closed form for the toy vs a Monte Carlo batch mean.
data = [np.array([1.0, 0.0]), np.array([0.0, -1.0])]  # q(x0): uniform over two images
exact = np.mean([D * (1 - theta * t * np.sqrt(1 - abar[t])) ** 2
                 + (theta * t * np.sqrt(abar[t])) ** 2 * np.sum(x0 ** 2)
                 for t in (1, 2) for x0 in data])
rng = np.random.default_rng(0)
N = 1_000_000
ts = rng.integers(1, 3, size=N)  # t ~ Uniform({1, 2})
x0s = np.stack(data)[rng.integers(0, 2, size=N)]  # x0 ~ q(x0)
es = rng.standard_normal((N, D))  # eps ~ N(0, I)
a = np.array([abar[1], abar[2]])[ts - 1][:, None]
xts = np.sqrt(a) * x0s + np.sqrt(1 - a) * es
mc = np.mean(np.sum((es - theta * ts[:, None] * xts) ** 2, axis=1))
assert abs(mc - exact) < 0.01
print("E[...] exact:", round(exact, 6), " Monte Carlo (1e6 draws):", round(mc, 6))

# 4. Why one-shot noising works: t single steps of eq. (2) give eq. (4).
betas = np.linspace(1e-4, 0.02, 1000)  # the paper's schedule, Section 4
abars = np.cumprod(1 - betas)
coef, var = 1.0, 0.0  # x_t = coef * x0 + noise with variance var
for t in range(1000):
    coef, var = np.sqrt(1 - betas[t]) * coef, (1 - betas[t]) * var + betas[t]
    assert np.isclose(coef, np.sqrt(abars[t])) and np.isclose(var, 1 - abars[t])
    assert np.isclose(coef ** 2 + var, 1.0)  # Var(x_t) = coef^2 Var(x0) + var = 1 if Var(x0) = 1
print("abar_1 =", abars[0], " abar_1000 =", f"{abars[-1]:.3e}",
      " Var(x_t) at t=1000 for unit-variance x0:", round(coef ** 2 + var, 12))

# 5. Branch t > 1: eq. (12) is eq. (10) with eq. (11) plugged in (t = 2, sigma^2 = beta).
x0, e, t = batch[1]
xt = x_t(x0, e, t)
mu_tilde = (np.sqrt(abar[t - 1]) * beta[t] / (1 - abar[t]) * x0
            + np.sqrt(alpha[t]) * (1 - abar[t - 1]) / (1 - abar[t]) * xt)  # eq. (7)
mu_theta = (xt - beta[t] / np.sqrt(1 - abar[t]) * eps_model(xt, t, theta)) / np.sqrt(alpha[t])  # eq. (11)
s2 = beta[t]
lhs = np.sum((mu_tilde - mu_theta) ** 2) / (2 * s2)
w = beta[t] ** 2 / (2 * s2 * alpha[t] * (1 - abar[t]))  # weight in eq. (12)
assert np.isclose(lhs, w * per_sample_loss(x0, e, t, theta))
print("t=2: eq.(10) term", round(lhs, 6), "= weight", round(w, 6), "x", losses[1])

# 6. Branch t = 1: the paper's approximation of the decoder (13), -log(density x bin width),
#    is the same squared error with weight beta_1 / (2 sigma_1^2 alpha_1), plus a constant
#    that has no theta in it. The exact (13) integrates the Gaussian over each pixel's bin.
x0, e, t = batch[0]
xt = x_t(x0, e, t)
s2 = beta[1]  # sigma_1^2: the paper gives none; this toy uses beta_1 (my choice)
# mu_theta(x_1, 1): eq. (11) at t = 1, as Algorithm 2 line 4 computes it when t = 1
mu1 = (xt - beta[1] / np.sqrt(1 - abar[1]) * eps_model(xt, 1, theta)) / np.sqrt(alpha[1])
bin_width = 2 / 255  # bin [x - 1/255, x + 1/255] of eq. (13)
approx = -np.log(np.exp(-(x0 - mu1) ** 2 / (2 * s2)) / np.sqrt(2 * np.pi * s2) * bin_width)
const = D * (0.5 * np.log(2 * np.pi * s2) - np.log(bin_width))
w1 = beta[1] / (2 * s2 * alpha[1])
assert np.isclose(approx.sum() - const, w1 * per_sample_loss(x0, e, 1, theta))
Phi = lambda z: 0.5 * (1 + erf(z / np.sqrt(2)))  # standard normal CDF
lo = np.where(x0 == -1, -np.inf, x0 - 1 / 255)  # delta_-: edge bin at -1 runs to -inf
hi = np.where(x0 == 1, np.inf, x0 + 1 / 255)  # delta_+: edge bin at +1 runs to +inf
# x0[1] = 0 is off the 256-level grid; the bins of (13) are still defined there.
exact = np.array([-np.log(Phi((h - m) / np.sqrt(s2)) - Phi((l - m) / np.sqrt(s2)))
                  for l, h, m in zip(lo, hi, mu1)])
assert abs(exact[1] - approx[1]) < 1e-3  # interior pixel: the approximation is close
assert approx[0] - exact[0] > 4  # edge pixel x0[0] = 1: it is far off
print("t=1: approx. decoder NLL (density x bin width)", round(approx.sum(), 6), "= const",
      round(const, 6), "+ weight", round(w1, 6), "x", losses[0])
print("     per pixel: approx.", np.round(approx, 6), " exact eq.(13)", np.round(exact, 6),
      " exact total", round(exact.sum(), 6))

# 7. Which t the unweighted loss down-weights, with the paper's schedule (Section 4).
T = 1000
al = 1 - betas
abar_prev = np.append(1.0, abars[:-1])
w_large = betas ** 2 / (2 * betas * al * (1 - abars))  # sigma_t^2 = beta_t (Section 3.2: t > 1)
bt_tilde = (1 - abar_prev) / (1 - abars) * betas  # sigma_t^2 = beta~_t (Section 3.2: t > 1)
assert bt_tilde[0] == 0  # the formula would give beta~_1 = 0 because abar_0 = 1 (empty product)
w_small = betas[1:] ** 2 / (2 * bt_tilde[1:] * al[1:] * (1 - abars[1:]))  # t = 2..T
# t = 1: the paper gives no sigma_1^2; both fixed options of the code use beta~_2 there.
w_large[0] = w_1 = betas[0] / (2 * bt_tilde[1] * al[0])  # beta_1 / (2 sigma_1^2 alpha_1), check 6
t_min = int(np.argmin(w_large)) + 1
assert np.all(np.diff(w_large[:t_min]) < 0) and np.all(np.diff(w_large[t_min - 1:]) > 0)
assert w_large[:10].min() > 7 * w_large[99:].max() and w_small[:9].min() > 7 * w_small[98:].max()
print(f"t=   1  weight in approx. eq.(13), code's sigma_1^2 = beta~_2 = {bt_tilde[1]:.3e}: {w_1:.4f}")
for t in (2, 10, 100, t_min, 1000):
    print(f"t={t:4d}  weight in eq.(12): sigma^2=beta {w_large[t - 1]:.4f}   sigma^2=beta~ {w_small[t - 2]:.4f}")
print("smallest weight (sigma^2=beta) at t =", t_min,
      "; min over t<=10 / max over t>=100:", round(w_large[:10].min() / w_large[99:].max(), 2))
print("weight (sigma^2=beta) at t=1000 / at t=%d:" % t_min, round(w_large[-1] / w_large[t_min - 1], 2))
print("all checks passed")
```

Run with `uv run --no-project --with numpy python check.py`. Output:

```text
per-sample ||eps - eps_theta||^2: [0.58 0.5 ]  batch mean: 0.5399999999999999
same with mean over the D pixels: 0.26999999999999996
gradients: [-1.68  0.4 ]  theta after one step: 0.564  batch loss after: 0.50543
E[...] exact: 0.79  Monte Carlo (1e6 draws): 0.789321
abar_1 = 0.9999  abar_1000 = 4.036e-05  Var(x_t) at t=1000 for unit-variance x0: 1.0
t=2: eq.(10) term 0.303819 = weight 0.607639 x 0.49999999999999994
t=1: approx. decoder NLL (density x bin width) 10.965584 = const 10.512459 + weight 0.78125 x 0.58
     per pixel: approx. [5.326542 5.639042]  exact eq.(13) [0.432943 5.639043]  exact total 6.071987
t=   1  weight in approx. eq.(13), code's sigma_1^2 = beta~_2 = 5.453e-05: 0.9170
t=   2  weight in eq.(12): sigma^2=beta 0.2727   sigma^2=beta~ 0.5997
t=  10  weight in eq.(12): sigma^2=beta 0.0737   sigma^2=beta~ 0.0864
t= 100  weight in eq.(12): sigma^2=beta 0.0101   sigma^2=beta~ 0.0103
t= 349  weight in eq.(12): sigma^2=beta 0.0050   sigma^2=beta~ 0.0050
t=1000  weight in eq.(12): sigma^2=beta 0.0102   sigma^2=beta~ 0.0102
smallest weight (sigma^2=beta) at t = 349 ; min over t<=10 / max over t>=100: 7.22
weight (sigma^2=beta) at t=1000 / at t=349: 2.05
all checks passed
```

What the numbers show:

- **The loss and one step.** The batch loss 0.54 is the mean of the two
  per-sample errors. One step with $\eta=0.1$ moves $\theta$ from 0.5 to
  0.564 and lowers the batch loss to 0.505.
- **The average over all draws.** Over every $t$, $\mathbf x_0$ and
  $\boldsymbol\epsilon$, the toy loss is 0.79. A mean over a million draws
  gives 0.789, and a two-sample batch is a much noisier estimate of the same
  average.
- **What (14) drops at $t>1$.** At $t=2$ the bound's term is the same
  squared error times 0.6076, the weight that (14) drops.
- **The $t=1$ approximation.** Eq. (13) integrates a Gaussian of variance
  $\sigma_1^2$ over each pixel's bin. §3.4 replaces the integral with the
  density times the bin width, $2/255$, and ignores $\sigma_1^2$ and the edge
  bins (quoted under "What the paper claims"). This density × bin width
  version of $L_0$ is the squared error times $\beta_1/(2\sigma_1^2\alpha_1)$,
  plus a constant that has no $\theta$ in it; without the weight and the
  constant it is the term of (14). The weight is my derivation; the paper does
  not print it (I searched every use of $\sigma_1$ and $\beta_1$ in the TeX
  source). The derivation takes the mean of the decoder's Gaussian,
  $\boldsymbol\mu_\theta(\mathbf x_1,1)$ in eq. (13), to be eq. (11) at $t=1$: Algorithm 2's line 4 computes that formula at $t=1$,
  where $\mathbf z=\mathbf 0$, and §3.3 says "At the end of sampling, we
  display $\boldsymbol\mu_\theta(\mathbf x_1,1)$ noiselessly". Because
  $\bar\alpha_0=1$, $\alpha_1=\bar\alpha_1$ and $\beta_1=1-\bar\alpha_1$. So
  $\mathbf x_1=\sqrt{\alpha_1}\mathbf x_0+\sqrt{\beta_1}\boldsymbol\epsilon$,
  eq. (11) becomes
  $\boldsymbol\mu_\theta(\mathbf x_1,1)=(\mathbf x_1-\sqrt{\beta_1}\boldsymbol\epsilon_\theta)/\sqrt{\alpha_1}$,
  and
  $\mathbf x_0-\boldsymbol\mu_\theta(\mathbf x_1,1)=-\sqrt{\beta_1/\alpha_1}\,(\boldsymbol\epsilon-\boldsymbol\epsilon_\theta)$.
  For each pixel, minus the log of the density × width is the squared gap
  between pixel and mean divided by $2\sigma_1^2$, plus terms without
  $\theta$; summed over the pixels, that is the weight times
  $\Vert\boldsymbol\epsilon-\boldsymbol\epsilon_\theta\Vert^2$. The paper
  gives no $\sigma_1^2$ (see "What it computes and what it changes"), and
  (14) contains none. The toy uses $\sigma_1^2=\beta_1$, my choice, so for
  sample 1 the weight is 0.78125 and the constant 10.51. The exact eq. (13)
  term for the same sample is
  6.07, against 10.97 for the approximation. The second pixel, inside
  $(-1,1)$, agrees closely (5.639043 against 5.639042). The first pixel,
  $x_0^1=1$ (superscript = coordinate, as in eq. (13)), does not (0.43 against
  5.33): its bin runs to $+\infty$, and the approximation counts only a
  width of $2/255$. That is the "edge effects" the paper ignores.
- **Which $t$ lose weight (my computation).** With the paper's schedule, the
  dropped weight at $t=1$ is 0.92 under the code's $\sigma_1^2=\tilde\beta_2$,
  used by both fixed options (see "Reverse-process variance $\sigma_t^2$").
  For $t>1$ with $\sigma_t^2=\beta_t$, it is 0.27 at $t=2$, 0.074 at $t=10$
  and 0.010 at $t=100$. It reaches its minimum, 0.0050, at
  $t=349$, then rises to 0.0102 at $t=1000$. Every weight for $t\le10$ is at
  least 7.2 times every weight for $t\ge100$. $L_\mathrm{simple}$ sets them
  all to 1, so compared with the bound it gives the small-$t$ terms much less
  weight. The weight is not monotone in $t$: at $t=1000$ it is 2.05 times
  its value at $t=349$. The paper does not mention this (I searched every
  use of "weight" in the PDF text).

## What the code must decide

Files are cited at commit `1e0dceb`. Its
[`requirements.txt`](https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/requirements.txt)
pins `numpy==1.18.1` and `tensorflow==1.15.0` (lines 57 and 92), so library
behaviour below is cited from the NumPy 1.18 and TensorFlow r1.15
documentation, and from the TensorFlow v1.15.0 source where the documentation
does not say.

- **Sum or mean over the $D$ pixels.** The printed $\Vert\cdot\Vert^2$ is a
  sum. The code takes a mean:
  `losses = nn.meanflat(tf.squared_difference(target, model_output))`
  ([`diffusion_utils_2.py`](https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py)
  line 299), where `meanflat` is
  `tf.reduce_mean(x, axis=list(range(1, len(x.shape))))`
  ([`nn.py`](https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/nn.py)
  lines 34–35). [`tf.squared_difference`](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/squared_difference)
  "Returns (x - y)(x - y) element-wise", and
  [`tf.reduce_mean`](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/reduce_mean)
  "Reduces input_tensor along the dimensions given in axis", here every axis
  but the first, the batch axis. This divides the loss by $D=3072$ on
  CIFAR10, equally for every $t$, so the weighting across $t$ does not change.
- **Batch and one $t$ per example.** Algorithm 1 draws one image per step.
  Appendix B sets the batch size to 128 for CIFAR10 and 64 for larger images.
  The code runs the training step as parallel copies, called _replicas_, on
  the accelerator of Appendix B ("We used TPU v3-8 (similar to 8 V100 GPUs)
  for all experiments"), and gives each an equal share of the batch:
  `assert total_bs == num_tpu_replicas() * local_bs`
  ([`tpu_utils.py`](https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/tpu_utils/tpu_utils.py)
  lines 131–136). Each replica draws one $t$ for each of its `B` examples,
  `t = tf.random_uniform([B], 0, self.diffusion.num_timesteps, dtype=tf.int32)`,
  and averages their losses with `return {'loss': tf.reduce_mean(losses)}`
  ([`run_cifar.py`](https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/scripts/run_cifar.py)
  lines 45, 49 and 53); with no axis, `tf.reduce_mean` reduces "all dimensions".
- **Timestep indexing.** The code counts from zero. It draws $t$ from
  $\{0,\dots,999\}$ (`run_cifar.py` line 49; for
  [`tf.random_uniform`](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/random/uniform),
  "The lower bound minval is included in the range, while the upper bound
  maxval is excluded"), and `self.alphas_cumprod = np.cumprod(alphas, axis=0)`
  puts $\bar\alpha_1$ at index 0 (`diffusion_utils_2.py` line 65; the
  [`numpy.cumprod`](https://numpy.org/doc/1.18/reference/generated/numpy.cumprod.html)
  example turns `[1,2,3]` into `[1, 2, 6]`). The
  `q_sample` docstring says "t == 0 means diffused for 1 step" (line 103). So
  code $t$ = paper $t-1$, and the network receives the zero-based value,
  `model_output = denoise_fn(x_t, t)` (line 297).
- **Noise.** The noise is one tensor of the batch's shape, so each example
  gets its own:
  `noise = tf.random_normal(shape=x_start.shape, dtype=x_start.dtype)`
  (`diffusion_utils_2.py` line 282;
  [`tf.random_normal`](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/random/normal)
  defaults to `mean=0.0, stddev=1.0`). The target is that same tensor,
  `'eps': noise` (line 295).
- **Schedule precision.** The schedule is built in float64, with the comment
  "computations here in float64 for accuracy" (`diffusion_utils_2.py` line
  59), and converted to float32 when looked up,
  `tf.convert_to_tensor(a, dtype=tf.float32)` (line 91;
  [`dtype`](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/convert_to_tensor)
  is the "element type for the returned tensor").
- **Data scaling.** `normalize_data = lambda x_: x_ / 127.5 - 1.`
  ([`tpu_utils.py`](https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/tpu_utils/tpu_utils.py)
  line 64). Random flips: `run_cifar.py` lines 46–47;
  [`tf.image.random_flip_left_right`](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/random_flip_left_right):
  "Randomly flip an image horizontally (left to right)".
- **How $t$ enters the network.** The paper names a sinusoidal embedding added
  into each residual block (Appendix B). The code has
  `get_timestep_embedding` (`nn.py` line 90).
- **Optimizer and step.** Algorithm 1 says only "gradient descent step".
  Appendix B uses Adam with learning rate $2\times10^{-4}$ ($2\times10^{-5}$
  for $256\times256$ images), plus the weight average described in "Where the
  samples come from". The code adds two things the paper does not mention
  (I searched the PDF text for "warm" and "clip"), both set by
  `grad_clip=1., lr=2e-4, warmup=5000` (`run_cifar.py` line 134). One is a
  linear learning-rate warmup, rate $=2\times10^{-4}\cdot\min(\text{step}/5000,1)$
  ([`utils.py`](https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py)
  lines 145–148, called at `tpu_utils.py` line 157). The other clips each
  replica's own gradient (gradient clipping; replicas: see
  "Batch and one $t$ per example") with the threshold
  `grad_clip / float(num_tpu_replicas())` (`tpu_utils.py` line 164), instead
  of clipping the full-batch gradient. `utils.py` lines 157–159 and
  166–177 wrap Adam in `tf.tpu.CrossShardOptimizer`, then
  call its `compute_gradients`, `tf.clip_by_global_norm`, and its
  `apply_gradients`. In TensorFlow r1.15,
  [`apply_gradients`](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/tpu/CrossShardOptimizer)
  "Calls tpu_ops.cross_replica_sum() to sum gradient contributions across
  replicas", so the clip runs before the replicas' gradients are combined.
  [`tf.clip_by_global_norm`](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/clip_by_global_norm)
  sets each tensor to `t_list[i] * clip_norm / max(global_norm, clip_norm)`,
  where `global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))`: all of a
  replica's gradients shrink by one factor until their joint length is at
  most the threshold. The `CrossShardOptimizer` page names a `reduction`
  argument, default `losses.Reduction.MEAN`, "to apply to the shard losses"
  (the replicas' losses), and `utils.py` line 167 keeps that default. Under
  MEAN, the TensorFlow v1.15.0 `compute_gradients` multiplies each replica's
  loss by one over the shard count $N$ before taking the gradient,
  `scale = 1.0 / num_shards` and `loss *= scale`
  ([`tpu_optimizer.py`](https://github.com/tensorflow/tensorflow/blob/v1.15.0/tensorflow/python/tpu/tpu_optimizer.py)
  lines 144–149), so the sum in `apply_gradients` is a mean, as the class
  docstring says: "An optimizer that averages gradients across TPU shards"
  (line 35). $N$ is the same TPU-context value that `num_tpu_replicas()`
  reads (`tpu_optimizer.py` line 134, `tpu_utils.py` line 22), so the
  threshold $1/N$ applies to a gradient already divided by $N$: the clip caps
  each replica's unscaled gradient at length 1 (my reasoning from these
  lines). The EMA is in
  `tpu_utils.py` lines 89–92 and 168–171.
- **Dropout.** "We set the dropout rate on CIFAR10 to 0.1" (Appendix B);
  `dropout=0.1` in `run_cifar.py` line 137. The training loss passes it to
  the network (line 51; `_denoise` forwards it at lines 38–40), and sampling
  and bound evaluation pass `dropout=0` (lines 58, 66 and 74). Each residual
  block applies
  `h = tf.nn.dropout(h, rate=dropout)`
  ([`unet.py`](https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py)
  line 52), which "For each element of x, with probability rate, outputs 0"
  ([`tf.nn.dropout`](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/dropout)).
  So $\boldsymbol\epsilon_\theta$ is random during training.
- **Reverse-process variance $\sigma_t^2$.** $L_\mathrm{simple}$ contains no
  $\sigma_t$, so training does not set it. The sampler (Algorithm 2, line 4)
  uses it, and so does every evaluation of the bound: the $1/(2\sigma_t^2)$
  of eq. (8) and the $\sigma_1^2$ of eq. (13) enter the NLL of "Ours
  ($L_\mathrm{simple}$)" in Table 1 (see "What the paper claims"), and the rate and
  distortion in §4.3, which treat "the variational bound terms
  $L_1+\cdots+L_T$ as rate and $L_0$ as distortion". The code default is
  `model_var_type='fixedlarge'` (`run_cifar.py` line 136):
  $\sigma_t^2=\beta_t$ for $t>1$, but at $t=1$ the log variance is
  $\log\tilde\beta_2$,
  `'fixedlarge': (self.betas, np.log(np.append(self.posterior_variance[1], self.betas[1:])))`,
  under the comment "for fixedlarge, we set the initial (log-)variance like
  so to get a better decoder log likelihood" (`diffusion_utils_2.py` lines
  141–142; `posterior_variance[1]` is $\tilde\beta_2$ in the zero-based
  indexing above, lines 66 and 77; [`numpy.append`](https://numpy.org/doc/1.18/reference/generated/numpy.append.html)
  does "Append values to the end of an array"). The other fixed option,
  `'fixedsmall'` ($\sigma_t^2=\tilde\beta_t$ for $t>1$), takes its log variance from
  `self.posterior_log_variance_clipped` (line 143), which is
  `np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))`
  under the comment "below: log calculation clipped because the posterior variance
  is 0 at the beginning of the diffusion chain" (lines 78–79), so it too uses
  $\log\tilde\beta_2$ at $t=1$. Line 139 says "only
  log_variance is used in the KL computations", the code's bound terms for
  $t>1$ (`normal_kl`, line 261; line 271 uses its result for every $t>1$).
  The sampler also uses the log variance,
  `tf.exp(0.5 * model_log_variance) * noise` (line 196), and adds no noise at
  $t=1$ (`nonzero_mask`, line 195).

## What the paper claims and what it shows

- **That the $t>1$ terms "correspond to an unweighted version of Eq. (12)".**
  This follows from the definitions, and checks 5 and 7 reproduce it.
- **That the $t=1$ case "corresponds to $L_0$ with the integral in the
  discrete decoder definition (13) approximated by the Gaussian probability
  density function times the bin width, ignoring $\sigma_1^2$ and edge
  effects".** The paper states this without a written derivation. Check 6
  reproduces it (my derivation); see "The $t=1$ approximation" above.
- **That training on (14) was "beneficial to sample quality (and simpler to
  implement)".** The sample-quality part is measured, in Table 2
  (unconditional CIFAR10), by two scores the paper computes "on 50000
  samples" with code from its references [51] and [21] (Appendix B): the
  Inception score (IS) and the FID score. Higher IS and lower FID are better:
  Table 1 prints the highest IS and the lowest FID of each group in bold, and
  §4.1 says "With our FID score of 3.17, our unconditional model achieves
  better sample quality than most models in the literature". In the table,
  "fixed isotropic $\Sigma$" is §3.2's choice
  "$\Sigma_\theta(\mathbf x_t,t)=\sigma_t^2\mathbf I$" for the covariance of
  the denoising step (eq. (1)): one fixed variance $\sigma_t^2$ (see "What it
  computes and what it changes"), the same for every coordinate. "Simpler to
  implement" is the authors' assertion, with no measurement: Table 2 has only
  IS and FID columns, and no section reports implementation effort (I
  searched every use of "simpl" in the PDF text).

  | Prediction target | Objective | IS | FID |
  | --- | --- | --- | --- |
  | noise | $L_\mathrm{simple}$ | $9.46\pm0.11$ | 3.17 |
  | noise | true bound, fixed isotropic $\Sigma$ | $7.67\pm0.13$ | 13.51 |
  | mean $\tilde{\boldsymbol\mu}$ (best) | true bound, fixed isotropic $\Sigma$ | $8.06\pm0.09$ | 13.22 |

  "Final experiments were trained once and evaluated throughout training for
  sample quality. Sample quality scores and log likelihood are reported on
  the minimum FID value over the course of training" (Appendix B). The paper
  does not say what $\pm$ denotes, and FID has no error bar.
  It gives no run count for the ablation rows, and it reports no interval for
  any difference.
- **That "the baseline option of predicting $\tilde{\boldsymbol\mu}$ works
  well only when trained on the true variational bound instead of unweighted
  mean squared error, a simplified objective akin to Eq. (14)" (§4.2).** This
  is an experiment reported without scores: the Table 2 entry for
  $\Vert\tilde{\boldsymbol\mu}-\tilde{\boldsymbol\mu}_\theta\Vert^2$ is
  blank, and the caption says "Blank entries were unstable to train and
  generated poor samples with out-of-range scores". The unweighted loss is
  shown to work for predicting the noise, not for predicting the mean.
- **That (14) "is a weighted variational bound that emphasizes different
  aspects of reconstruction compared to the standard variational bound".**
  This is the paper's name for the reweighted sum. The sentence cites
  [18, 22], which I did not open. The paper does not state an inequality for
  $L_\mathrm{simple}$.
- **That the §4 setup "causes the simplified objective to down-weight loss
  terms corresponding to small $t$".** The paper states this but does not show
  the weights. Check 7 computes them (my computation; see "Which $t$ lose
  weight" above).
- **That "it is beneficial to down-weight them so that the network can focus
  on more difficult denoising tasks at larger $t$ terms".** This is argued.
  Table 2 tests the whole reweighting (bound against $L_\mathrm{simple}$), not
  this explanation or the small-$t$ part on its own.
- **That the true bound "yields better codelengths than training on the
  simplified objective, as expected, but the latter yields the best sample
  quality".** Both parts are measured, in Table 1. A codelength here is the
  NLL: "negative log likelihoods (lossless codelengths)" (§4.1). Table 1
  gives it in "bits/dim": the bound in base-2 logarithms, averaged over the
  $D$ coordinates (the code divides a mean over coordinates by $\log 2$,
  `diffusion_utils_2.py` lines 262 and 267). Lower is better; Table 1 bolds
  the lowest. The test NLL is $\le3.70$ (train 3.69) for "Ours ($L$, fixed
  isotropic $\Sigma$)" against $\le3.75$ (3.72) for "Ours
  ($L_\mathrm{simple}$)", and Table 1 repeats the IS and FID of the table
  above. The $\le$ marks a value of the bound, which depends on $\sigma_t^2$
  (see "Reverse-process variance $\sigma_t^2$"). "As expected" is the
  authors' expectation, not a measurement.
- **That "$L_T$ does not appear because the forward process variances $\beta_t$
  are fixed".** This follows from §3.1: the noising process has no learnable
  parameters, so $L_T$ does not depend on $\theta$.
- **The link to score matching.** The gradient of a log density with respect
  to the input point is "a vector field pointing in the direction where the
  log data density grows the most", called the _score_ (Song and Ermon, NCSN,
  [arXiv 1907.05600v3](https://arxiv.org/abs/1907.05600v3), §1; defined as
  $\nabla_{\mathbf x}\log p(\mathbf x)$ in §2). Denoising score matching
  "first perturbs the data point $\mathbf x$ with a pre-specified noise
  distribution $q_\sigma(\tilde{\mathbf x}\mid\mathbf x)$ and then employs
  score matching to estimate the score of the perturbed data distribution"
  (NCSN §2.1): a network sees noisy data and learns the score of the noisy
  distribution. Langevin dynamics "can produce samples from a probability
  density $p(\mathbf x)$ using only the score function" (NCSN §2.2): each step
  of its eq. (4) adds the score, times half a step size, and fresh Gaussian
  noise. DDPM says Algorithm 2 "resembles Langevin dynamics with
  $\boldsymbol\epsilon_\theta$ as a learned gradient of the data density"
  (§3.2). The paper states the link as an equivalence: a certain
  parameterization "reveals an equivalence with denoising score matching over
  multiple noise levels during training and with annealed Langevin dynamics
  during sampling", and "we consider this equivalence to be one of our primary
  contributions" (§1). The support is a derivation, eqs. (8)–(12): with the
  $\boldsymbol\epsilon$ parameterization (11), each bound term for $t>1$
  becomes eq. (12), which "resembles denoising score matching over multiple
  noise scales indexed by $t$", so "optimizing an objective resembling
  denoising score matching is equivalent to using variational inference to fit
  the finite-time marginal of a sampling chain resembling Langevin dynamics"
  (§3.2). §5 states the reverse direction: "a certain weighted form of
  denoising score matching is the same as variational inference to train a
  Langevin-like sampler". The comparison of (14)'s $t>1$ weighting with
  NCSN's is worded as an analogy: "analogous to the loss weighting used by the
  NCSN denoising score matching model" (§3.4).

## Sources

- Ho, Jain, Abbeel, [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239v2),
  arXiv 2006.11239v2: the TeX source and PDF of v2. Quotes were checked
  against the PDF text, and section, equation, table and page numbers were
  read from the PDF.
- Official code, [hojonathanho/diffusion at `1e0dceb3b3495bbe19116a5e1b3596cd0706c543`](https://github.com/hojonathanho/diffusion/tree/1e0dceb3b3495bbe19116a5e1b3596cd0706c543):
  `diffusion_tf/diffusion_utils_2.py`, `diffusion_tf/nn.py`,
  `diffusion_tf/utils.py`, `diffusion_tf/tpu_utils/tpu_utils.py`,
  `diffusion_tf/models/unet.py`, `scripts/run_cifar.py` and
  `requirements.txt`, linked above.
- TensorFlow r1.15 documentation for `tf.random.uniform`,
  `tf.math.reduce_mean`, `tf.math.squared_difference`, `tf.random.normal`,
  `tf.convert_to_tensor`, `tf.image.random_flip_left_right`,
  `tf.tpu.CrossShardOptimizer`, `tf.clip_by_global_norm` and `tf.nn.dropout`,
  and NumPy 1.18 documentation for `numpy.cumprod` and `numpy.append`, linked
  in "What the code must decide".
- TensorFlow source at tag `v1.15.0`, `tensorflow/python/tpu/tpu_optimizer.py`,
  linked in "What the code must decide".
- Song and Ermon, [Generative Modeling by Estimating Gradients of the Data
  Distribution](https://arxiv.org/abs/1907.05600v3), arXiv 1907.05600v3
  (10 Oct 2020), NCSN, DDPM's [55]: §1, §2, §2.1 and §2.2 of the PDF text.
- Kingma and Welling, [Auto-Encoding Variational
  Bayes](https://arxiv.org/abs/1312.6114v10), arXiv 1312.6114v10 (1 May 2014,
  the latest version before DDPM v2), DDPM's [33]: §2.1 and §2.2 of the PDF
  text.
- PyTorch 2.15 documentation pages for `torch.cumprod`, `torch.randint`,
  `torch.randn_like` and `torch.mean`, linked in "Read from the inside out".
- Named but not opened: Sohl-Dickstein et al. (2015), [53], which Table 1
  lists as "Diffusion (original)"; Vincent (2011), "A connection between
  score matching and denoising autoencoders", [61]; Gregor et al. (2016),
  "Towards conceptual compression", [18]; Higgins et al. (2017),
  "beta-VAE", [22]; Salimans et al. (2016), "Improved techniques for
  training GANs", [51]; Heusel et al. (2017), "GANs trained by a two
  time-scale update rule converge to a local Nash equilibrium", [21].
