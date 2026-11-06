# Auto-Encoding Variational Bayes: a number samples can estimate that never exceeds a datapoint's log-likelihood

**Source:** Diederik P. Kingma and Max Welling,
[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114v11), v11,
§2.2, equations (1)–(3) and the unnumbered sum before (1). Other locations used:
§2.1, Figure 1 caption, §2.3 (equations (4)–(8), Algorithm 1), §3 (equations
(9)–(10)), §5 and the Figure 2 caption, Appendix B (closed-form KL), Appendix C
(equations (11)–(12)), Appendix D, Appendix F (equation (16)). It never uses
the name "ELBO" (evidence lower bound); it calls $\mathcal L$ the "(variational) lower bound".

## What it computes and what it changes

The paper models each datapoint $\mathbf x$ (an image, say) as made in two
steps: draw a hidden vector $\mathbf z$ from a distribution $p_\theta(\mathbf z)$
(the _prior_), then let the _decoder_ $p_\theta(\mathbf x\mid\mathbf z)$, with
parameters $\theta$, turn $\mathbf z$ into a distribution over $\mathbf x$. In
the paper's §3 example, the _variational auto-encoder_ (VAE), the decoder and
the encoder below are neural networks and the prior is the fixed
$\mathcal N(\mathbf 0,\mathbf I)$. Training would ideally maximize
$\log p_\theta(\mathbf x)$, the log of the datapoint's probability, where that
probability (the _marginal likelihood_) is the decoder's probability of
$\mathbf x$ averaged over every $\mathbf z$ drawn from the prior; but that
average is an integral over all $\mathbf z$ that, §2.1 says, is _intractable_
(cannot be computed) in "cases of moderately complicated likelihood
functions", for example a decoder that is a neural network with a nonlinear
hidden layer.
Equations (1)–(3) define a replacement, $\mathcal L$: an average over
$\mathbf z$ drawn from a second model, the _encoder_ $q_\phi$ with parameters $\phi$, which proposes which
$\mathbf z$ fit a given $\mathbf x$; $\mathcal L$ never exceeds
$\log p_\theta(\mathbf x)$ (it is a _lower bound_) and falls short of it by
exactly how far the encoder's proposal is from the true distribution of
$\mathbf z$ given $\mathbf x$. Compared with evaluating
$\log p_\theta(\mathbf x)$ itself or with the EM algorithm (an alternating
method that needs that true distribution), both ruled out in §2.1's
intractable case, the quantity optimized changes: $\mathcal L$ instead of
$\log p_\theta(\mathbf x)$. Earlier variational methods also fit "an
approximation to the intractable posterior" (§1). In "the common mean-field
approach", that approximation is a product of independent distributions, one
per latent variable ($q_\phi$, by contrast, "is not necessarily factorial",
§2.1), and it "requires analytical solutions of expectations w.r.t. the
approximate posterior" (§1). Compared with them the bound is the same. What
changes is that its averages are estimated from sampled $\mathbf z$ and
log-densities at those samples (§2.3), and the encoder's parameters $\phi$ are
trained by gradient together with $\theta$
rather than computed from "some closed-form expectation" (§2.1).

## The equation

$$
\log p_\theta(\mathbf x^{(1)},\cdots,\mathbf x^{(N)})=\sum_{i=1}^N\log p_\theta(\mathbf x^{(i)})
\qquad\text{(§2.2, unnumbered)}
$$

$$
\log p_\theta(\mathbf x^{(i)})=D_{KL}(q_\phi(\mathbf z|\mathbf x^{(i)})||p_\theta(\mathbf z|\mathbf x^{(i)}))+\mathcal L(\theta,\phi;\mathbf x^{(i)})
\tag{1}
$$

$$
\log p_\theta(\mathbf x^{(i)})\ge\mathcal L(\theta,\phi;\mathbf x^{(i)})=\mathbb E_{q_\phi(\mathbf z|\mathbf x)}\left[-\log q_\phi(\mathbf z|\mathbf x)+\log p_\theta(\mathbf x,\mathbf z)\right]
\tag{2}
$$

$$
\mathcal L(\theta,\phi;\mathbf x^{(i)})=-D_{KL}(q_\phi(\mathbf z|\mathbf x^{(i)})||p_\theta(\mathbf z))+\mathbb E_{q_\phi(\mathbf z|\mathbf x^{(i)})}\left[\log p_\theta(\mathbf x^{(i)}|\mathbf z)\right]
\tag{3}
$$

The paper prints $\theta$, $\phi$, $\mathbf x$ and $\mathbf z$ in bold; bold
marks vectors or sets of parameters. "RHS" in the quotes below means right-hand
side.

## Symbols

| Symbol | Shape or type | Meaning | Role | Defined at |
| ------ | ------------- | ------- | ---- | ---------- |
| $\mathbf x^{(i)}$ | a vector; the paper names its length $D$ (a count, not the $D$ of $D_{KL}$) only for the Bernoulli decoder (App. C.1, eq. (11)) | datapoint number $i$ of the training set. The superscript $(i)$ is a label, not a power. "i.i.d." means each datapoint is drawn independently from the same process | input data | "Let us consider some dataset $\mathbf X=\{\mathbf x^{(i)}\}_{i=1}^N$ consisting of $N$ i.i.d. samples of some continuous or discrete variable $\mathbf x$." (§2.1) |
| $\mathbf x$ (no superscript, in eq. (2)) | as $\mathbf x^{(i)}$ | eq. (2) writes $\mathbf x$ inside the expectation while its left side is about $\mathbf x^{(i)}$; eq. (3), which "can also be written" for (2), has $\mathbf x^{(i)}$ in those places, so I read it as $\mathbf x^{(i)}$ | input data | no separate definition; settled from eq. (3) (§2.2) |
| $i$, $N$ | integers, $1\le i\le N$ | $i$ numbers the datapoints; $N$ is how many there are | index; fixed setting | same quote as $\mathbf x^{(i)}$ (§2.1) |
| $\mathbf z$ | vector of $J$ real numbers | the hidden vector behind a datapoint (_latent variable_ or _code_); never observed | random sample: inside $\mathbb E_{q_\phi(\mathbf z\mid\mathbf x)}$ it is drawn from the encoder | "involving an unobserved continuous random variable $\mathbf z$" (§2.1); "the unobserved variables $\mathbf z$ have an interpretation as a latent representation or _code_" (§2.1); "Let $J$ be the dimensionality of $\mathbf z$." (App. B) |
| $\theta$ | a set of parameters, form not stated in general; in the MLPs of App. C (_MLP_: "a fully-connected neural network with a single hidden layer", §3), weight matrices and bias vectors | parameters of the generative model (prior and decoder) | trained weight | "The variational parameters $\phi$ are learned jointly with the generative model parameters $\theta$." (Figure 1 caption); "$\theta=\{\mathbf W_1,\mathbf W_2,\mathbf b_1,\mathbf b_2\}$ are the weights and biases of the MLP" (App. C.1) |
| $\phi$ | as $\theta$: in the encoder MLP of App. C.2, weight matrices and bias vectors | parameters of the encoder, called _variational parameters_ | trained weight | same Figure 1 caption; "when this network is used as an encoder $q_\phi(\mathbf z\vert\mathbf x)$, then $\mathbf z$ and $\mathbf x$ are swapped, and the weights and biases are variational parameters $\phi$" (App. C.2) |
| $\mid$ | separator | "given": $p(\mathbf a\mid\mathbf b)$ is a distribution over $\mathbf a$ that depends on the value of $\mathbf b$ | notation | "a value $\mathbf x^{(i)}$ is generated from some conditional distribution $p_{\theta^*}(\mathbf x\vert\mathbf z)$" (§2.1) |
| $p_\theta(\mathbf z)$ | distribution over $\mathbf z\in\mathbb R^J$; at one $\mathbf z$, one number (a density) | the _prior_: how $\mathbf z$ is drawn before any data is seen. In the VAE, $\mathcal N(\mathbf 0,\mathbf I)$: $J$ independent standard normal numbers | in general part of the model, with parameters $\theta$; in the VAE fixed, with no parameters | "(1) a value $\mathbf z^{(i)}$ is generated from some prior distribution $p_{\theta^*}(\mathbf z)$" and "We assume that the prior $p_{\theta^*}(\mathbf z)$ and likelihood $p_{\theta^*}(\mathbf x\vert\mathbf z)$ come from parametric families of distributions $p_\theta(\mathbf z)$ and $p_\theta(\mathbf x\vert\mathbf z)$" (§2.1); "$p_\theta(\mathbf z)=\mathcal N(\mathbf z;\mathbf 0,\mathbf I)$. Note that in this case, the prior lacks parameters." (§3) |
| $p_\theta(\mathbf x\mid\mathbf z)$ | distribution over $\mathbf x$ for a given $\mathbf z$; at one pair, one number: a density for real-valued $\mathbf x$, a probability for binary $\mathbf x$ | the _decoder_ (or _likelihood_): how likely the datapoint is if its code was $\mathbf z$ | computed from $\mathbf z$ with trained parameters $\theta$; in the VAE by an MLP (App. C, eqs. (11)–(12)) | "we will refer to $p_\theta(\mathbf x\vert\mathbf z)$ as a probabilistic _decoder_, since given a code $\mathbf z$ it produces a distribution over the possible corresponding values of $\mathbf x$" (§2.1) |
| $p_\theta(\mathbf x,\mathbf z)$ | one number per pair | the _joint_: how likely it is to draw this $\mathbf z$ and then this $\mathbf x$ | computed intermediate | "Let $p_\theta(\mathbf x,\mathbf z)=p_\theta(\mathbf z)p_\theta(\mathbf x\vert\mathbf z)$ be the generative model" (App. D) |
| $p_\theta(\mathbf x^{(i)})$ | one number | the _marginal likelihood_ of datapoint $i$: the decoder's probability of the datapoint, averaged over $\mathbf z$ drawn from the prior (equivalently, the joint integrated over all $\mathbf z$) | defined by the model, but not computable in general (_intractable_); training never evaluates it | "the integral of the marginal likelihood $p_\theta(\mathbf x)=\int p_\theta(\mathbf z)p_\theta(\mathbf x\vert\mathbf z)\,d\mathbf z$ is intractable (so we cannot evaluate or differentiate the marginal likelihood)" (§2.1) |
| $p_\theta(\mathbf x^{(1)},\cdots,\mathbf x^{(N)})$ | one number | marginal likelihood of the whole dataset | defined by the model, but not computable in general: its log is the sum of the logs of the row above, each intractable | "The marginal likelihood is composed of a sum over the marginal likelihoods of individual datapoints" (§2.2) |
| $p_\theta(\mathbf z\mid\mathbf x^{(i)})$ | distribution over $\mathbf z$ | the _true posterior_: under the model, which codes could have produced this datapoint, and how likely each is | defined by the model, but not computable in general (intractable): the formula in the quote (_Bayes' rule_) divides by $p_\theta(\mathbf x)$, the intractable integral of the $p_\theta(\mathbf x^{(i)})$ row (this reason is mine; the paper states both facts in one sentence); training never evaluates it | "the true posterior density $p_\theta(\mathbf z\vert\mathbf x)=p_\theta(\mathbf x\vert\mathbf z)p_\theta(\mathbf z)/p_\theta(\mathbf x)$ is intractable (so the EM algorithm cannot be used)" (§2.1) |
| $q_\phi(\mathbf z\mid\mathbf x^{(i)})$ | distribution over $\mathbf z$ computed from $\mathbf x^{(i)}$; in the VAE a Gaussian with mean $\boldsymbol\mu^{(i)}$ and standard deviation $\boldsymbol\sigma^{(i)}$, both $J$-vectors, independent per coordinate (eq. (9)) | the _encoder_ (also _recognition model_ or _approximate posterior_): a guess at the true posterior, one distribution per datapoint | computed intermediate: computed from $\mathbf x^{(i)}$ with trained parameters $\phi$; in the VAE by an MLP (§3, App. C.2) | "let us introduce a recognition model $q_\phi(\mathbf z\vert\mathbf x)$: an approximation to the intractable true posterior $p_\theta(\mathbf z\vert\mathbf x)$" (§2.1); "given a datapoint $\mathbf x$ it produces a distribution (e.g. a Gaussian) over the possible values of the code $\mathbf z$ from which the datapoint $\mathbf x$ could have been generated" (§2.1) |
| $D_{KL}(q\Vert p)$ | one number, $\ge0$ | _KL divergence_: draw $\mathbf z$ from the first distribution, take $\log q(\mathbf z)-\log p(\mathbf z)$, average. It measures how far $q$ is from $p$. The $\Vert$ only separates the two arguments; it is not a norm | operator | "The first RHS term is the KL divergence of the approximate from the true posterior. Since this KL-divergence is non-negative, …" (§2.2); the formula, for the prior only: "$-D_{KL}((q_\phi(\mathbf z)\Vert p_\theta(\mathbf z))=\int q_\theta(\mathbf z)(\log p_\theta(\mathbf z)-\log q_\theta(\mathbf z))\,d\mathbf z$" (App. B, which writes $q_\theta$ for $q_\phi$) |
| $\mathbb E_{q_\phi(\mathbf z\mid\mathbf x)}[\cdot]$ | one number | _expectation_: the average of the bracket over $\mathbf z$ drawn from the distribution in the subscript, here the encoder's distribution for this datapoint | operator; the subscript names the sampler | the same bound written as an integral against $q_\phi$: "$\mathcal L(\theta,\phi;\mathbf x^{(i)})=\int q_\phi(\mathbf z\vert\mathbf x)\left(\log p_\theta(\mathbf x^{(i)}\vert\mathbf z)+\log p_\theta(\mathbf z)-\log q_\phi(\mathbf z\vert\mathbf x)\right)d\mathbf z$" (App. F, eq. (16)); its sample-mean estimate is eq. (5) (§2.3) |
| $\mathcal L(\theta,\phi;\mathbf x^{(i)})$ | one number per datapoint | the _variational lower bound_ | the equation's result; the objective, maximized over $\theta$ and $\phi$, and estimated from samples (§2.3) | "the second RHS term $\mathcal L(\theta,\phi;\mathbf x^{(i)})$ is called the (variational) _lower bound_ on the marginal likelihood of datapoint $i$" (§2.2); "We want to differentiate and optimize the lower bound $\mathcal L(\theta,\phi;\mathbf x^{(i)})$ w.r.t. both the variational parameters $\phi$ and generative parameters $\theta$." (§2.2); "The variational lower bound (the objective to be maximized)" (App. B) |

**Not defined in the paper:**

- **The base of $\log$.** Not stated in §2, §3 or Appendices B–C. Appendix B's
  result, $\frac12\sum_j(1+\log\sigma_j^2-\mu_j^2-\sigma_j^2)$, holds only for
  the natural log (my derivation): the 1 is the average of
  $(z_j-\mu_j)^2/\sigma_j^2$ under $q$, and $\mu_j^2+\sigma_j^2$ the average of
  $z_j^2$; with base-$b$ logs each of these three terms would be divided by
  $\ln b$. The check below confirms it with natural logs, so I read $\log$ as
  natural.
- **A general formula for $D_{KL}$.** The paper writes the integral only for
  $D_{KL}(q_\phi\Vert p_\theta(\mathbf z))$ (App. B). For eq. (1) I use the
  same form with the true posterior as the second argument.
- **The semicolon in $\mathcal L(\theta,\phi;\mathbf x^{(i)})$.** Not
  explained. The sentence quoted in the last row says what it separates:
  $\theta,\phi$ are what training changes, $\mathbf x^{(i)}$ is the datapoint.

## Read from the inside out

Each side of (1)–(3) is one number per datapoint; the bracket inside the
expectation in (2) is one number per (datapoint, $\mathbf z$) pair, and the sum
before (1) gives one number for the dataset. For a continuous
variable, $p$ gives probability per unit of volume (a _density_), not a
probability; a density can exceed 1, so its log can be positive.

1. **$\log p_\theta(\mathbf x,\mathbf z)$**, inside (2): one number for one
   code $\mathbf z$ (a $J$-vector) and the datapoint. Because the joint is a
   product (App. D), its log is a sum:
   $\log p_\theta(\mathbf z)+\log p_\theta(\mathbf x^{(i)}\mid\mathbf z)$.
2. **$\log q_\phi(\mathbf z\mid\mathbf x)$**: the encoder's log-density at the
   same $\mathbf z$, one number.
3. **$-\log q_\phi(\mathbf z\mid\mathbf x)+\log p_\theta(\mathbf x,\mathbf z)$**:
   one number per $\mathbf z$. It is large when the model finds this
   (datapoint, code) pair likely; averaged, the $-\log q_\phi$ part is larger
   when the encoder spreads its probability over many $\mathbf z$ (mine).
4. **$\mathbb E_{q_\phi(\mathbf z\mid\mathbf x)}[\ \cdot\ ]$** averages step 3
   over $\mathbf z$ drawn from the encoder's distribution for this datapoint.
   It runs over $\mathbf z$ (all of $\mathbb R^J$), not over datapoints and
   not over the entries of a vector. Exactly, it is the integral of App. F
   eq. (16); in code, draw $L$ codes ($L$ is a sample count, not the bound
   $\mathcal L$), compute step 3 for each, and take the mean
   (eq. (5)).
5. **$\mathcal L(\theta,\phi;\mathbf x^{(i)})=\dots$** names that number. It
   depends on $\theta$ through $p_\theta$ and on $\phi$ through $q_\phi$,
   including where the samples land.
6. **$\ge$** in (2) asserts that $\mathcal L$ is never above
   $\log p_\theta(\mathbf x^{(i)})$, for every $\theta$ and $\phi$. The reason
   is eq. (1).
7. **Eq. (1)** states that the gap is exactly
   $D_{KL}(q_\phi(\mathbf z\mid\mathbf x^{(i)})\Vert p_\theta(\mathbf z\mid\mathbf x^{(i)}))$,
   an average over $\mathbf z\sim q_\phi$, one number $\ge0$. The paper states
   (1) without a derivation. Mine: $\log p_\theta(\mathbf x)$ does not depend on
   $\mathbf z$, so it equals its own average over $\mathbf z\sim q_\phi$; by the
   posterior's definition (§2.1),
   $\log p_\theta(\mathbf x)=\log p_\theta(\mathbf x,\mathbf z)-\log p_\theta(\mathbf z\mid\mathbf x)$
   for every $\mathbf z$; adding and subtracting $\log q_\phi(\mathbf z\mid\mathbf x)$
   inside the average gives
   $\mathbb E_q[\log q_\phi-\log p_\theta(\mathbf z\mid\mathbf x)]+\mathbb E_q[-\log q_\phi+\log p_\theta(\mathbf x,\mathbf z)]$,
   the KL term plus $\mathcal L$. This assumes $p_\theta(\mathbf z\mid\mathbf x)>0$
   wherever $q_\phi>0$, which holds in the VAE: the $\mathcal N(\mathbf 0,\mathbf I)$
   prior and the decoders of eqs. (11)–(12) are positive for every $\mathbf z$,
   so the posterior is positive everywhere too.
8. **Eq. (3)** substitutes step 1 into (2) and splits the average in two (my
   derivation). The first term,
   $\mathbb E_q[\log p_\theta(\mathbf z)-\log q_\phi(\mathbf z\mid\mathbf x^{(i)})]=-D_{KL}(q_\phi\Vert p_\theta(\mathbf z))\le0$,
   is largest when the encoder's distribution equals the prior; §2.3 reads it
   as "regularizing $\phi$". The second,
   $\mathbb E_q[\log p_\theta(\mathbf x^{(i)}\mid\mathbf z)]$, is how well the
   decoder recovers the datapoint from the encoder's codes. §2.3 calls it once
   "the expected reconstruction error" and once "an expected negative
   reconstruction error"; it is the second, so higher is better.
9. **The sum before (1)** adds per-datapoint log marginal likelihoods. It
   holds because the datapoints are i.i.d. (§2.1); the paper does not spell
   out this step. Adding (2) over
   $i$ (mine) bounds the dataset's log-likelihood by
   $\sum_i\mathcal L(\theta,\phi;\mathbf x^{(i)})$, which eq. (8) (§2.3)
   estimates from a minibatch.

Superscripts: $(i)$ and $(1),\dots,(N)$ are labels; (1)–(3) contain no powers
and no matrix products, so row or column vectors do not matter here.
Relation signs: $=$ in (1) and (3) holds for every $\theta,\phi,i$; the $=$ in
(2) gives $\mathcal L$'s formula; $\ge$ is the bound.

**Kind of statement:** a bound that is also an objective. The gap between the
two sides of (2) is $D_{KL}(q_\phi\Vert p_\theta(\mathbf z\mid\mathbf x^{(i)}))$
by (1); it is zero when the encoder's distribution equals the true posterior
(shown in the check). The direction is maximize, over $\theta$ and $\phi$, with
the average over $\mathbf z\sim q_\phi(\mathbf z\mid\mathbf x^{(i)})$ and a
sum over the datapoints. A consequence of (1) (mine): $\log p_\theta(\mathbf x^{(i)})$
does not depend on $\phi$, so raising $\mathcal L$ by changing $\phi$ lowers
the KL to the true posterior by the same amount. Changing $\theta$ moves both
$\log p_\theta(\mathbf x^{(i)})$ and the gap, so a higher $\mathcal L$ does not
by itself guarantee a higher $\log p_\theta(\mathbf x^{(i)})$.

## Where the samples come from

- **Datapoints $\mathbf x^{(i)}$** (the sum over $i$): a fixed dataset, "here we
  assume a fixed dataset for simplicity" (§2), collected before training.
  Training visits it in random minibatches of $M$ (Algorithm 1, eq. (8)).
- **$\mathbf z$ in $\mathbb E_{q_\phi(\mathbf z\mid\mathbf x^{(i)})}$** in (2)
  and (3): generated during
  training by the encoder with the current $\phi$, the same parameters being
  trained (no frozen copy), separately for each datapoint. In the VAE,
  $\mathbf z^{(i,l)}=\boldsymbol\mu^{(i)}+\boldsymbol\sigma^{(i)}\odot\boldsymbol\epsilon^{(l)}$
  with $\boldsymbol\epsilon^{(l)}\sim\mathcal N(\mathbf 0,\mathbf I)$ and
  $\odot$ the entrywise product (§3).
- **$D_{KL}(q_\phi\Vert p_\theta(\mathbf z\mid\mathbf x^{(i)}))$** in (1) is
  never computed or sampled; it is the unknown gap.
- **$D_{KL}(q_\phi\Vert p_\theta(\mathbf z))$** in (3) is computed in closed
  form when it can be integrated analytically (App. B gives the Gaussian
  encoder, $\mathcal N(\mathbf 0,\mathbf I)$ prior case), so only the second
  term of (3) is sampled (eq. (7)). It is sampled only when form (2) is
  estimated with eq. (6).
- **Detached:** nothing. Algorithm 1 takes the gradient of the minibatch
  estimate with respect to both $\theta$ and $\phi$. Because the samples depend
  on $\phi$, §2.2 ends by calling the gradient with respect to $\phi$ "a bit
  problematic"; the paper's remedy is to draw the noise $\boldsymbol\epsilon$
  from a distribution that does not depend on $\phi$ and compute $\mathbf z$
  from it (§2.3–§2.4), not blocking the gradient through $\mathbf z$ (a
  _stop-gradient_).

## Check with small numbers

Inputs: a toy model I chose so that every quantity has an exact answer. $J=D=1$;
prior $\mathcal N(0,1)$ as in §3; decoder $p_\theta(x\mid z)=\mathcal N(x;z,1)$;
two datapoints, $x=2$ and $x=0$, so the sum over $i$ has two items; and a linear
encoder $q_\phi(z\mid x)=\mathcal N(z;ax+b,e^c)$ at two settings of $\phi$. In
the first, $q$ equals the true posterior $\mathcal N(x/2,1/2)$ for every $x$,
so the gap is zero. In the second, $q$ equals the prior, so the gap is positive
and the $-D_{KL}(q\Vert p_\theta(\mathbf z))$ term is zero. Every integral is
computed on a grid from the definitions, independently of the closed forms; the
values in `EXACT` are my derivations.

```python
import numpy as np

# Toy model (chosen for this check, not from the paper): J = 1 latent, D = 1 data entry.
# prior p(z) = N(0, 1), as in §3; decoder p(x|z) = N(x; w*z, s2) with theta = (w, s2) = (1, 1).
# Encoder q_phi(z|x) = N(z; a*x + b, exp(c)) with phi = (a, b, c).
w, s2 = 1.0, 1.0
X = np.array([2.0, 0.0])                      # N = 2 datapoints
PHI = {"posterior": (0.5, 0.0, np.log(0.5)),  # q equals the true posterior N(x/2, 1/2)
       "prior":     (0.0, 0.0, 0.0)}          # q equals the prior N(0, 1)

z = np.linspace(-15, 15, 300_001)             # grid for the integrals over z
dz = z[1] - z[0]
integral = lambda f: float(np.sum(f) * dz)


def log_normal(v, mean, var):
    return -0.5 * np.log(2 * np.pi * var) - (v - mean) ** 2 / (2 * var)


def terms(x, phi):
    a, b, c = phi
    mu, var = a * x + b, np.exp(c)
    log_q = log_normal(z, mu, var)
    q = np.exp(log_q)
    log_prior = log_normal(z, 0.0, 1.0)
    log_dec = log_normal(x, w * z, s2)        # log p(x|z) at every grid z
    log_joint = log_prior + log_dec           # log p(x,z) = log p(z) + log p(x|z) (App. D)
    log_px = np.log(integral(np.exp(log_joint)))   # log of p(x) = integral of p(z) p(x|z) dz
    log_post = log_joint - log_px             # log p(z|x) = log p(x,z) - log p(x)
    return dict(
        mu=mu, var=var, log_px=log_px,
        elbo_eq2=integral(q * (-log_q + log_joint)),
        kl_post=integral(q * (log_q - log_post)),
        kl_prior=integral(q * (log_q - log_prior)),
        recon=integral(q * log_dec),
    )


h = 0.5 * np.log(2)  # exact values below are my own closed-form derivations
EXACT = {("posterior", 2.0): dict(elbo=-0.5 * np.log(4 * np.pi) - 1, gap=0.0),
         ("prior", 2.0):     dict(elbo=-0.5 * np.log(2 * np.pi) - 2.5, gap=1.5 - h),
         ("posterior", 0.0): dict(elbo=-0.5 * np.log(4 * np.pi), gap=0.0),
         ("prior", 0.0):     dict(elbo=-0.5 * np.log(2 * np.pi) - 0.5, gap=0.5 - h)}

rng = np.random.default_rng(0)
print(f"{'q':9s} {'x':>3s} {'log p(x)':>9s} {'L':>9s} {'gap':>9s} {'-KL prior':>9s} {'recon':>9s} {'1 sample':>9s} {'1e5 mean':>9s}")
for name, phi in PHI.items():
    sum_logpx = sum_elbo = 0.0
    for x in X:
        t = terms(x, phi)
        elbo_eq3 = -t["kl_prior"] + t["recon"]
        neg_kl_appB = 0.5 * (1 + np.log(t["var"]) - t["mu"] ** 2 - t["var"])   # App. B, J = 1
        assert abs(t["log_px"] - log_normal(x, 0.0, w**2 + s2)) < 1e-9          # p(x) = N(x; 0, 2)
        assert abs(t["log_px"] - (t["kl_post"] + t["elbo_eq2"])) < 1e-9         # eq. (1)
        assert t["elbo_eq2"] <= t["log_px"] + 1e-9 and t["kl_post"] >= -1e-9  # eq. (2): bound
        assert abs(t["elbo_eq2"] - elbo_eq3) < 1e-9                             # eq. (3) = eq. (2)
        assert abs(-t["kl_prior"] - neg_kl_appB) < 1e-9                         # App. B closed form
        ex = EXACT[(name, x)]
        assert abs(t["elbo_eq2"] - ex["elbo"]) < 1e-9 and abs(t["kl_post"] - ex["gap"]) < 1e-9
        # Monte Carlo, as code computes it: z = mu + sigma * eps, eps ~ N(0, 1)  (eq. (4), §3)
        zs = t["mu"] + np.sqrt(t["var"]) * rng.standard_normal(100_000)
        per_sample = log_normal(zs, 0, 1) + log_normal(x, w * zs, s2) - log_normal(zs, t["mu"], t["var"])
        assert abs(per_sample.mean() - t["elbo_eq2"]) < 0.01
        if name == "posterior":  # every single sample already gives log p(x) exactly
            assert np.allclose(per_sample, t["log_px"], atol=1e-12)
        sum_logpx += t["log_px"]
        sum_elbo += t["elbo_eq2"]
        if name == "prior" and x == 2.0:  # one sample can land above log p(x)
            assert per_sample[0] > t["log_px"]
        shown = lambda v: round(v, 4) + 0.0  # + 0.0 turns a -0.0 from grid error into 0.0
        print(f"{name:9s} {x:3.0f} {t['log_px']:+9.4f} {t['elbo_eq2']:+9.4f} {shown(t['kl_post']):+9.4f} "
              f"{shown(-t['kl_prior']):+9.4f} {t['recon']:+9.4f} {per_sample[0]:+9.4f} {per_sample.mean():+9.4f}")
    # dataset level: log p(x1, x2) = log(p(x1) p(x2)) for i.i.d. data, and the bounds add up
    joint = np.log(np.exp(log_normal(X[0], 0, 2)) * np.exp(log_normal(X[1], 0, 2)))
    assert abs(joint - sum_logpx) < 1e-9 and sum_elbo <= sum_logpx + 1e-9
    print(f"{name:9s} sum {sum_logpx:+9.4f} {sum_elbo:+9.4f}")

# The order inside KL matters: q = N(0,1) against the posterior N(1, 1/2) for x = 2
log_a, log_b = log_normal(z, 0, 1), log_normal(z, 1, 0.5)
kl_ab = integral(np.exp(log_a) * (log_a - log_b))
kl_ba = integral(np.exp(log_b) * (log_b - log_a))
assert abs(kl_ab - kl_ba) > 0.5
print(f"KL(N(0,1)||N(1,1/2))={kl_ab:.4f}  KL(N(1,1/2)||N(0,1))={kl_ba:.4f}")
print("all checks passed")
```

Run with `uv run --no-project --with numpy python check.py`. Output:

```text
q           x  log p(x)         L       gap -KL prior     recon  1 sample  1e5 mean
posterior   2   -2.2655   -2.2655   +0.0000   -0.5966   -1.6689   -2.2655   -2.2655
posterior   0   -1.2655   -1.2655   +0.0000   -0.0966   -1.1689   -1.2655   -1.2655
posterior sum   -3.5310   -3.5310
prior       2   -2.2655   -3.4189   +1.1534   +0.0000   -3.4189   -2.0409   -3.4184
prior       0   -1.2655   -1.4189   +0.1534   +0.0000   -1.4189   -1.6677   -1.4212
prior     sum   -3.5310   -4.8379
KL(N(0,1)||N(1,1/2))=1.1534  KL(N(1,1/2)||N(0,1))=0.5966
all checks passed
```

Columns: `L` is eq. (2), `gap` is $D_{KL}(q\Vert p_\theta(z\mid x))$, `-KL prior`
and `recon` are the two terms of eq. (3), and the last two columns estimate `L`
from 1 and from $10^5$ samples of $z=\mu+\sigma\epsilon$. In every row
`L + gap = log p(x)` (eq. (1)) and `-KL prior + recon = L` (eq. (3)). With $q$
equal to the true posterior even one sample returns $\log p_\theta(x)$ exactly; with $q$ equal to the prior it falls short by 1.1534
and 0.1534. One sample can land above $\log p_\theta(x)$ (−2.0409 > −2.2655):
the inequality holds for the average, not for each sample. KL's order matters:
1.1534 one way, 0.5966 the other.

## What the code must decide

The paper links no code (the only URL in the PDF and TeX source is the dataset
footnote in §5), so "open" below means the paper does not say.

- **Sign:** the paper maximizes $\mathcal L$ by "stochastic gradient ascent"
  (§5). Code whose optimizer minimizes a loss must use $-\mathcal L$.
- **How the expectation is estimated:** $L$ sampled codes per datapoint with
  $\mathbf z=\boldsymbol\mu+\boldsymbol\sigma\odot\boldsymbol\epsilon$; the
  paper uses $L=1$ and $M=100$ (Algorithm 1 caption, §5).
- **Form (2) or form (3):** eq. (6) estimates (2) entirely by sampling; eq. (7)
  estimates (3) with the KL term in closed form. That needs a KL that can be
  integrated analytically; App. B gives the formula for a Gaussian encoder and
  a $\mathcal N(\mathbf 0,\mathbf I)$ prior, which is the VAE's case. The
  VAE's Gaussian encoder outputs $\log\boldsymbol\sigma^2$ (App. C.2,
  eq. (12)). With $\boldsymbol\mu$ and $\log\boldsymbol\sigma^2$ stored as
  $M\times J$ arrays, one row per datapoint, App. B's sum runs over the $J$
  columns of each row and gives one KL value per datapoint.
- **Reduction over the entries of $\mathbf x$ and $\mathbf z$:** sums, decided
  by the paper for the VAE: eq. (11) sums over the $D$ entries and App. B over the $J$
  latent coordinates. A mean over the $D$ entries instead divides the
  reconstruction term by $D$ and changes its weight against the KL term, so
  the objective is no longer $\mathcal L$.
- **Reduction over datapoints:** eq. (8) multiplies the minibatch sum by $N/M$.
  A batch mean differs by the constant factor $N$, which leaves the maximizer of
  $\mathcal L$ alone unchanged; but §5 adds "a small weight decay term
  corresponding to a prior $p(\theta)=\mathcal N(0,\mathbf I)$" and calls the
  result "approximate MAP estimation" (maximum a posteriori: maximize the
  log-likelihood plus $\log p(\theta)$). That prior fixes the term at
  $-\tfrac12\Vert\theta\Vert^2$ (gradient $-\theta$), added to eq. (8)'s
  $N/M$-scaled sum; with a minibatch mean instead, it becomes
  $-\Vert\theta\Vert^2/(2N)$ (my derivation). The paper names the prior only
  for $\theta$, while §5 says all parameters, "both variational and generative",
  were "jointly stochastically optimized using the MAP criterion"; whether
  $\phi$ gets the same term is open.
- **$\log 0$ in the Bernoulli decoder:** eq. (11) takes $\log y_i$ and
  $\log(1-y_i)$ of a sigmoid output $y_i$. The sigmoid never reaches 0 or 1
  exactly, but in floating point it rounds to 1 for large inputs, and then
  $\log(1-y_i)=-\infty$ (my check, computing $1/(1+e^{-t})$ in NumPy). How the
  paper handled this is open.

  ```python
  import numpy as np

  # First input t (step 0.01) at which y = 1 / (1 + exp(-t)) rounds to exactly 1
  for name, dt in (("float32", np.float32), ("float64", np.float64)):
      t = np.arange(0, 60, 0.01).astype(dt)
      y = dt(1) / (dt(1) + np.exp(-t))
      first = t[np.argmax(y == 1)]
      assert np.all(y[t < first] < 1) and np.all(y[t >= first] == 1)
      with np.errstate(divide="ignore"):
          print(f"{name}: y == 1 from t = {first:.2f}, log(1 - y) = {np.log(dt(1) - y[t >= first][0])}")
  ```

  Output:

  ```text
  float32: y == 1 from t = 16.64, log(1 - y) = -inf
  float64: y == 1 from t = 36.74, log(1 - y) = -inf
  ```
- **Binary MNIST:** the Bernoulli decoder is for "binary data" (§3) and the
  experiments use MNIST (§5); how the pixels were made binary is open.
- **Continuous data:** for Frey Face the decoder is Gaussian, "except that the
  means were constrained to the interval $(0,1)$ using a sigmoidal activation
  function at the decoder output" (§5). Any floor on $\boldsymbol\sigma$ is
  open.
- **What is detached:** nothing; see "Where the samples come from".
- **What is reported:** Figure 2 plots "the estimated average variational lower
  bound per datapoint", so the reported number is divided by the number of
  datapoints.

## What the paper claims and what it shows

- **The paper does not claim the bound as a contribution.** The abstract's
  contributions are a reparameterized estimator of it (§2.3) and the encoder
  trained with it.
- **Eq. (1):** stated without derivation; the derivation in step 7 is mine.
  "Since this KL-divergence is non-negative" is asserted, not proven.
- **When the bound is tight:** for eq. (2) the paper does not say; it follows
  from (1) (see "Kind of statement"). For the parameter-level bound of App. F
  it says "the bound equals the true marginal when the approximate and true
  posteriors match exactly".
- **Eq. (3):** stated ("which can also be written as"); it follows from the
  factorization in App. D (step 8, mine).
- **High variance of the naive gradient at the end of §2.2:** cited to [BJP12]
  ("see e.g."), not measured in this paper.
- **The closed-form version "typically has less variance than the generic
  estimator"** (§2.3): asserted, with no measurement in the paper.
- **$L=1$ is enough:** "In our experiments we found that the number of samples
  $L$ per datapoint can be set to 1 as long as the minibatch size $M$ was large
  enough, e.g. $M=100$" (§2.3): an observation, with no table or figure.
- **Regularization:** Figure 2 shows train and test curves of $\mathcal L$ for
  several latent sizes. "Interestingly, superfluous latent variables did not
  result in overfitting, which is explained by the regularizing nature of the
  variational bound" (§5) is the authors' explanation; no experiment removes the
  KL term to test it.
- **Comparison with wake-sleep** (an earlier method that also trains a
  recognition model, but optimizes two objectives that together are not a
  bound on the marginal likelihood, §4): "Our method converged considerably faster and
  reached a better solution in all experiments" (Figure 2 caption). The curves
  have no error bars ("The estimator variance was small (< 1) and omitted"),
  and the number of runs is not stated.
- **Reading Figure 2's axes (mine):** the MNIST panels' tick labels run from
  −150 to −100, but the frames reach above the −100 tick and the highest
  curves end above it; every curve enters at the lower edge, −150, so early
  values fall below the plotted range. The Frey Face panels plot 0 to 1600. For binary $\mathbf x$ both terms of (3)
  are $\le0$, so $\mathcal L\le0$; a Gaussian density can exceed 1, so for
  continuous data $\mathcal L$ can be positive.

## Sources

- Diederik P. Kingma and Max Welling, Auto-Encoding Variational Bayes,
  [arXiv:1312.6114v11](https://arxiv.org/abs/1312.6114v11): the PDF (14 pages,
  read through `pdftotext` and as page images) and the TeX source of v11
  (`iclr14_sva.tex`, `iclr14_sva_appendix.tex`, `shared/commands.tex`).
- Named, not opened: Blei, Jordan and Paisley, Variational Bayesian inference
  with Stochastic Search, ICML 2012 ([BJP12]).
