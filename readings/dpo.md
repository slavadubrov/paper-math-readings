# DPO: a two-class loss that trains the language model directly on preference pairs

**Source:** Rafailov, Sharma, Mitchell, Ermon, Manning, Finn,
[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290v3),
v3, §4, equation (7). Other locations used: §1; §3, equations (1)–(3); §4,
equations (4) and (5), the gradient after equation (7) and the "DPO outline"
paragraph; §5.1 (Definition 1, Lemma 1, Theorem 1 and its proof sketch); §6
(tasks, Methods and §6.1); Appendix A.1, A.2, A.4 and A.6; Appendix B (the paper's PyTorch listing
and hyperparameters); Appendix C.1 and Table 3. The paper links no official
code repository, so the Appendix B listing is the only code cited here.

## What it computes and what it changes

Equation (7) is the loss DPO minimizes to fine-tune a language model on human
preferences. Each training example is a prompt with two answers, one of them
marked as preferred by a person. The loss measures how far the model being
trained, $\pi_\theta$, has raised the probability of the preferred answer,
relative to a frozen copy of the starting model, $\pi_\text{ref}$, compared
with how far it has raised the probability of the rejected answer. The larger
the first is relative to the second, the lower the loss. The paper calls a
language model a _policy_, a term from reinforcement learning: a rule that
picks an output (here, a completion) for an input (a prompt). $\pi$ is the
letter it uses for a policy.

It changes the pipeline in §3. There, equation (1) models a person's choice
between two answers $y_1$, $y_2$ to a prompt $x$: they prefer $y_1$ with
probability $\frac{e^{r^*(x,y_1)}}{e^{r^*(x,y_1)}+e^{r^*(x,y_2)}}$, where $r^*$
is a reward, one score per answer, that "we do not have access to" (§3). This
is the _Bradley–Terry model_. Equation (2) trains a separate reward
network $r_\phi$ on the same pairs with the loss
$-\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))$, where $\sigma(z)=1/(1+e^{-z})$ is
the logistic function, which maps any number into $(0,1)$. Equation (3) then
fine-tunes the language model with reinforcement learning (PPO), generating
answers from it during training, to raise that reward. A penalty keeps the
model close to the starting model. That penalty is a KL divergence: a measure
of how different two probability distributions are, which is zero when they are
identical. A number $\beta>0$ sets how strong the penalty is. Equation
(7) puts $\beta\log\frac{\pi_\theta(y\mid x)}{\pi_\text{ref}(y\mid x)}$ where
equation (2) has $r_\phi(x,y)$. As a result, only the language model is
trained, and no answers are generated during training.

Why a log-ratio can stand in for a reward (§4): for any reward function $r$,
"the optimal solution to the KL-constrained reward maximization objective in
Eq. 3 takes the form":

$$
\pi_r(y\mid x) = \frac{1}{Z(x)}\pi_\text{ref}(y\mid x)\exp\left(\frac{1}{\beta}r(x, y)\right), \qquad (4)
$$

"where $Z(x) =\sum_{y}\pi_\text{ref}(y\mid x)\exp\left(\frac{1}{\beta}r(x, y)\right)$
is the partition function". $Z(x)$ is a sum over every possible completion $y$
of the prompt. Dividing by it makes the probabilities in (4) add up to 1:
Appendix A.1, which names the same expression $\pi^*$, calls it "a valid
probability distribution as $\pi^*(y|x)\ge0$ for all $y$ and
$\sum_y\pi^*(y|x)=1$". The paper notes that "it is still expensive to estimate the
partition function $Z(x)$". Taking the log of both sides of (4) and
rearranging gives (5):

$$
r(x,y) =\beta \log \frac{\pi_r(y\mid x)}{\pi_\text{ref}(y\mid x)} + \beta \log Z(x). \qquad (5)
$$

So a reward is $\beta$ times the log-ratio of its optimal policy to
$\pi_\text{ref}$, plus $\beta\log Z(x)$, which depends on the prompt alone. §4
applies (5) to the ground-truth reward $r^*$ and its optimal policy $\pi^*$,
then fits a parametrized $\pi_\theta$ in place of $\pi^*$. The implicit reward
$\hat r_\theta$ (see Symbols) has the form of (5) with $\pi_\theta$ in place of
$\pi_r$ and without the prompt-only term $\beta\log Z(x)$. §5.1 states that
dropping that term leaves "a reward function in the equivalence class of
$r(x, y)$" (proof sketch of Theorem 1). Two rewards are in one class when they
differ only by a function of the prompt (Definition 1), and such rewards
"induce the same preference distribution" (Lemma 1). The term also cancels in
the difference of step 5 below. Equation (6) is the step between (5) and (7): the preference model (1) with (5)
substituted for $r^*$, which gives the preference probability in terms of
$\pi^*$ and $\pi_\text{ref}$ alone (§4). My reading of why (7) has no KL term
of its own: (4) is the optimal solution of the KL-penalized problem (3), so the
penalty is already built into (4), and from there into (5)–(7), through
$\pi_\text{ref}$ and $\beta$.

## The equation

$$
\mathcal{L}_\text{DPO}(\pi_{\theta}; \pi_\text{ref}) = -\mathbb{E}_{(x, y_w, y_l)\sim \mathcal{D}}\left[\log \sigma \left(\beta \log \frac{\pi_{\theta}(y_w\mid x)}{\pi_\text{ref}(y_w\mid x)} - \beta \log \frac{\pi_{\theta}(y_l\mid x)}{\pi_\text{ref}(y_l\mid x)}\right)\right] \qquad (7)
$$

## Symbols

| Symbol | Shape or type | Meaning | Role | Defined at |
| ------ | ------------- | ------- | ---- | ---------- |
| $\mathcal L_\text{DPO}(\pi_\theta;\pi_\text{ref})$ | one number | The DPO loss | computed intermediate (the output) | "we can formulate a maximum likelihood objective for a parametrized policy $\pi_\theta$. Analogous to the reward modeling approach (i.e. Eq. 2), our policy objective becomes:" (§4, the sentence that introduces (7)) |
| $\pi_\theta$ | function: (prompt, completion) → probability of that whole completion. In code: one log-probability per completion | The language model being trained | trained weight (through $\theta$) | "In practice, the language model policy $\pi_\theta$ is also initialized to $\pi^\text{SFT}$." (§3, after (3)); "a parametrized policy $\pi_\theta$" (§4) |
| $\theta$ | all trainable parameters of the model | What the optimizer changes | trained weight | "The gradient with respect to the parameters $\theta$ can be written as:" (§4) |
| $\pi_\text{ref}$ | same type as $\pi_\theta$ | Reference model: the supervised fine-tuned (SFT) model, or, when none exists, a model fine-tuned on the preferred completions | frozen model | "where $\beta$ is a parameter controlling the deviation from the base reference policy $\pi_\text{ref}$, namely the initial SFT model $\pi^\text{SFT}$." (§3, after (3)); "optimize the language model $\pi_\theta$ to minimize $\mathcal L_\text{DPO}$ for the given $\pi_\text{ref}$ and $\mathcal D$ and desired $\beta$." and "Since the preference datasets are sampled using $\pi^\text{SFT}$, we initialize $\pi_\text{ref}=\pi^\text{SFT}$ whenever available. However, when $\pi^\text{SFT}$ is not available, we initialize $\pi_\text{ref}$ by maximizing likelihood of preferred completions $(x, y_w)$" (§4, DPO outline) |
| $x$ | a prompt (text) | The input the model answers | input data | "the SFT model is prompted with prompts $x$ to produce pairs of answers $(y_1, y_2)\sim\pi^\text{SFT}(y\mid x)$." (§3, Reward Modelling Phase) |
| $y_w$, $y_l$ | two completions (text) for the same $x$ | The preferred and the dispreferred answer. The subscripts $w$ and $l$ are labels, not indices | input data | "$y_w$ and $y_l$ denotes the preferred and dispreferred completion amongst $(y_1, y_2)$ respectively." (§3) |
| $\mid$ in $\pi(y\mid x)$ | notation | "Given": $\pi(y\mid x)$ is the probability the model assigns to completion $y$ for prompt $x$ | (notation) | Not defined as notation. The usage is fixed by the §3 sentence quoted in the $x$ row: a model "prompted with prompts $x$" produces answers $\sim\pi^\text{SFT}(y\mid x)$ |
| $\mathcal D$ | a list of $N$ triples $(x, y_w, y_l)$ | The preference dataset | input data | "Assuming access to a static dataset of comparisons $\mathcal D=\{x^{(i)}, y_w^{(i)}, y_l^{(i)}\}_{i=1}^N$ sampled from $p^*$" (§3, after (1)). $p^*$ is the human preference distribution of equation (1) |
| $(x,y_w,y_l)\sim\mathcal D$ under $\mathbb E$ | one triple | The expectation averages over triples drawn from $\mathcal D$ | random sample (drawn from fixed data) | The paper uses $\sim$ for "drawn from", as in "$(y_1, y_2)\sim\pi^\text{SFT}(y\mid x)$" (§3); $\mathcal D$ as in its row |
| $\beta$ | one positive number | How strongly the trained model is held to $\pi_\text{ref}$. 0.1 unless noted otherwise, and 0.5 for TL;DR summarization. The IMDb sentiment experiment runs DPO with $\beta\in\{0.05,0.1,1,5\}$, one setting per run, to trace the reward-KL frontier of Figure 2 (left) | fixed setting or hyperparameter | The §3 quote in the $\pi_\text{ref}$ row; "beta: temperature controlling strength of KL penalty" (Appendix B listing); "Unless noted otherwise, we use a $\beta = 0.1$" and "For TL;DR summarization, we use $\beta = 0.5$" (Appendix B); "using a different hyperparameter for policy conservativeness in each run (target KL $\in \{3, 6, 9, 12\}$ for PPO, $\beta \in \{0.05, 0.1, 1, 5\}$, …" (§6.1) |
| $\sigma$ | function from a real number to $(0,1)$ | The logistic function (sigmoid), $\sigma(z)=1/(1+e^{-z})$. Not a standard deviation | (operator, no parameters) | "where $\sigma$ is the logistic function." (§3, after (2)). Appendix A.2's last two lines show $\frac{1}{1+\exp(b-a)}=\sigma(a-b)$, which gives the formula |
| $\hat r_\theta(x,y)$ | one number per (prompt, completion) | The _implicit reward_: $\beta\log\frac{\pi_\theta(y\mid x)}{\pi_\text{ref}(y\mid x)}$. The two terms inside $\sigma$ in (7) are $\hat r_\theta(x,y_w)$ and $\hat r_\theta(x,y_l)$ | computed intermediate | "where $\hat r_\theta(x, y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_\text{ref}(y\mid x)}$ is the reward implicitly defined by the language model $\pi_\theta$ and reference model $\pi_\text{ref}$" (§4, after the gradient) |
| $m$ | one number per triple | The margin $\hat r_\theta(x,y_w)-\hat r_\theta(x,y_l)$, the argument of $\sigma$ | computed intermediate | (introduced here) |

**Not defined in the paper:**

- **How $\pi_\theta(y\mid x)$ and $\pi_\text{ref}(y\mid x)$ are computed from a
  token-by-token language model.** The paper does not say whether a
  completion's log-probability is the sum or the mean of its token
  log-probabilities, or which positions count (prompt tokens, padding). I
  looked in §3, §4, Appendix A and the Appendix B listing, which takes them as
  inputs ("pi_logps: policy logprobs, shape (B,)"), and searched the full text
  for "token".
- **The semicolon in $\mathcal L_\text{DPO}(\pi_\theta;\pi_\text{ref})$.** It is
  not defined. The DPO outline (quoted in the $\pi_\text{ref}$ row) settles
  what it separates: the loss is minimized over $\pi_\theta$, with
  $\pi_\text{ref}$ given.
- **The base of $\log$.** It is not stated. My own derivation: §4 says "we first
  take the logarithm of both sides of Eq. 4" to reach (5), and (4) contains
  $\exp(\frac1\beta r(x,y))$. Only the natural log turns that into
  $\frac1\beta r(x,y)$, so (5), and with it (7), use the natural log.
- **$\mathbb E$ itself.** The expectation sign is not defined; it is read in
  step 7 below.

## Read from the inside out

1. **Four probabilities per triple.** $\pi_\theta(y_w\mid x)$,
   $\pi_\text{ref}(y_w\mid x)$, $\pi_\theta(y_l\mid x)$ and
   $\pi_\text{ref}(y_l\mid x)$. Each is the probability of a whole completion,
   one number. For a batch of pairs, the code holds four tensors with one
   entry per pair (`T` entries in the Appendix B listing, after indexing with
   `yw_idxs` and `yl_idxs`), and they hold log-probabilities. The completions
   are the saved texts of a triple in $\mathcal D$ (see its row), so each
   number scores text that already exists; the model generates nothing (see
   "Where the samples come from"). My derivation, which the paper does not
   state: by the chain rule of probability, $\log\pi_\theta(y_w\mid x)$ is the
   sum, over the tokens of the saved $y_w$, of each token's log-probability
   given $x$ and the earlier tokens of that saved $y_w$, not tokens the model
   samples; one forward pass over $x$ followed by $y_w$ gives all of these
   terms. How the code combines them is open (see "What the code must
   decide"). $\pi_\text{ref}$ makes the same pass over the same text (for its
   gradients, see "Where the samples come from").
2. **Ratio, then log.** $\log\frac{\pi_\theta(y_w\mid x)}{\pi_\text{ref}(y_w\mid x)}=\log\pi_\theta(y_w\mid x)-\log\pi_\text{ref}(y_w\mid x)$,
   one number. It is positive when the trained model gives $y_w$ more
   probability than the reference does, zero when they agree, and negative
   when it gives less. The fraction is a ratio of two numbers. The subscripts
   $\theta$ and "ref" are labels for the two models.
3. **Scale by $\beta$.** This gives $\hat r_\theta(x,y_w)$, and steps 1–3 on
   $y_l$ give $\hat r_\theta(x,y_l)$. The paper calls these implicit rewards
   because they play the part that $r_\phi(x,y)$ plays in equation (2).
4. **Subtract.** $m=\hat r_\theta(x,y_w)-\hat r_\theta(x,y_l)$, one number per
   triple. The Appendix B listing groups the same four numbers differently:
   `beta * (pi_logratios - ref_logratios)`, that is
   $\beta\big((\log\pi_\theta(y_w)-\log\pi_\theta(y_l))-(\log\pi_\text{ref}(y_w)-\log\pi_\text{ref}(y_l))\big)$.
   The two groupings are equal (checked below).
5. **Sigmoid.** $\sigma(m)\in(0,1)$. This is a preference probability. The
   Bradley–Terry probability of equation (1) (first section), written with
   rewards $r_1$ and $r_2$ for the two answers, equals $\sigma(r_1-r_2)$ ("the
   Bradley-Terry model depends only on the difference of rewards between two
   completions", §4). So $\sigma(m)$ is the probability, under that model and
   with the implicit rewards, that $y_w$ is the one preferred. Because only the
   difference of rewards matters, the term $\beta\log Z(x)$ in equation (5)
   (shown in the first section), which depends on the prompt alone, cancels ("the partition function
   cancels", §4). That is why (7) needs no sum over all possible completions.
6. **Log, then minus.** $\log\sigma(m)\le0$ is the log-probability of the
   label that was actually observed ($y_w$ preferred). Its negative,
   $-\log\sigma(m)\ge0$, is binary cross-entropy with the target always 1, the
   label of the pair. The paper uses that name: "DPO can therefore optimize a
   policy using a simple binary cross entropy objective" (§1). The loss is
   $\log2$ when $m=0$ and approaches 0 as $m$ grows.
7. **Expectation over $\mathcal D$.** $\mathbb E_{(x,y_w,y_l)\sim\mathcal D}[\cdot]$
   averages the bracket over the triples of the dataset. In code it is
   estimated by a mean over a batch of pairs (my inference; see "What the code
   must decide"). Result: one number.
8. **The relation sign.** "$=$" introduces the name $\mathcal L_\text{DPO}$ for
   the right-hand side, so this line is also a definition.

The gradient that §4 prints after (7) is
$-\beta\,\mathbb E\big[\sigma(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w))\,[\nabla_\theta\log\pi(y_w\mid x)-\nabla_\theta\log\pi(y_l\mid x)]\big]$.
The paper calls its reading of it intuitive: "Intuitively, the gradient of the
loss function $\mathcal L_\text{DPO}$ increases the likelihood of the preferred
completions $y_w$ and decreases the likelihood of dispreferred completions
$y_l$" (§4). My derivation: for one pair, (7) depends on $\theta$ only through
the gap $\log\pi_\theta(y_w\mid x)-\log\pi_\theta(y_l\mid x)$, so a small
enough gradient-descent step raises that gap. Both log-probabilities depend on
the same $\theta$, so the step can lower both, as long as
$\log\pi_\theta(y_l\mid x)$ falls faster. Pal et al.
([Smaug](https://arxiv.org/abs/2402.13228v2), v2, abstract) "show
theoretically that the standard DPO loss can lead to a reduction of the
model’s likelihood of the preferred examples", and then "show empirically
that this phenomenon occurs when fine-tuning LLMs on common datasets, especially datasets
in which the edit distance between pairs of completions is low". When you
monitor a run, log the two log-probabilities, not only the loss. Each pair is
weighted by $\sigma(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w))$, which is 0.5
at a tie, approaches 1 when the implicit rewards rank the pair the wrong way by
a wide margin, and approaches 0 when they rank it right by a wide margin (the
paper's label: "higher weight when reward estimate is wrong").

**Kind of statement:** an objective, a loss to minimize. The direction is given
in the DPO outline: "optimize the language model $\pi_\theta$ to minimize
$\mathcal L_\text{DPO}$". It is minimized over $\theta$ only. Steps 7 and 8
give the data it averages over and the name it defines.

## Where the samples come from

- **The expectation runs over $\mathcal D$,** a dataset fixed before training.
  Its completions are not generated by $\pi_\theta$: the abstract says DPO
  eliminates "the need for sampling from the LM during fine-tuning". The DPO
  outline describes the ideal: "Sample completions $y_1, y_2\sim\pi_\text{ref}(\cdot\mid x)$
  for every prompt $x$, label with human preferences to construct the offline
  dataset of preferences". The experiments mostly reuse datasets made by other
  models: "The human preference dataset was gathered by Stiennon et al. on
  samples from a different, but similarly-trained, SFT model." (§6, TL;DR). For
  Anthropic HH, "Each transcript ends with a pair of responses generated by a
  large (although unknown) language model along with a preference label
  denoting the human-preferred response." (§6). In the IMDb sentiment task the
  labels come from a classifier, not people: "We then use this model to sample
  4 completions for 25000 prefixes and create 6 preference pairs for each
  prefix using the ground-truth reward model." (Appendix C.1).
- **$\pi_\text{ref}$ is held fixed** ("for the given $\pi_\text{ref}$", DPO
  outline). Its two log-probabilities per pair do not depend on $\theta$, so
  gradients flow only through the two $\pi_\theta$ terms. My inference: they
  can be computed once per pair, without gradient tracking.
- **No stop-gradient appears in (7).** The Appendix B listing calls
  `.detach()`, which "Returns a new Tensor, detached from the current graph.
  The result will never require gradient." (PyTorch v2.4.0, `Tensor.detach`
  docstring; the paper names no PyTorch version), only on `rewards`, a second
  returned value (the implicit rewards of every completion), not on `losses`.

## Check with small numbers

The inputs are three pairs with $\beta=0.5$ and whole-completion probabilities
chosen so that the answers are exact. Pair 0 moved probability toward $y_w$
($m=\log3$). Pair 1 moved it toward $y_l$ ($m=-\log3$). Pair 2 equals the
reference ($m=0$), which is every pair at the first step when $\pi_\theta$
starts as $\pi_\text{ref}$ (§3: $\pi_\theta$ "is also initialized to
$\pi^\text{SFT}$", the usual $\pi_\text{ref}$). Together they use all four
probabilities, $\beta$, $\sigma$, the log, the minus sign and the mean, and
they cover both signs of the margin. The script also checks:

- the Appendix B grouping against (7);
- that equation (1) is $\sigma$ of a reward difference, and that a shift such
  as $\beta\log Z(x)$, the same for both answers, cancels;
- the §4 gradient against finite differences, and the weight in Appendix A.4's
  last line against the same differences;
- the numerical stability of the log-sigmoid at a margin of $-800$.

```python
import math

import numpy as np


def log_sigmoid(z):
    # log sigma(z) = -log(1 + e^{-z}), computed without overflow
    return -np.logaddexp(0.0, -z)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def dpo_losses(pi_w, ref_w, pi_l, ref_l, beta):
    """Per-pair term inside the expectation of eq. (7).

    Each argument is a log-probability of a whole completion, shape (n_pairs,).
    """
    r_w = beta * (pi_w - ref_w)  # beta * log(pi_theta(y_w|x) / pi_ref(y_w|x))
    r_l = beta * (pi_l - ref_l)  # beta * log(pi_theta(y_l|x) / pi_ref(y_l|x))
    return -log_sigmoid(r_w - r_l)


beta = 0.5
# Three pairs. Probabilities are of whole completions (toy values).
# pair 0: policy moved toward y_w and away from y_l   -> margin +log 3
# pair 1: policy moved the wrong way                   -> margin -log 3
# pair 2: policy equals reference (start of training)  -> margin 0
pi_w = np.log([0.3, 0.1, 0.2])
ref_w = np.log([0.1, 0.3, 0.2])
pi_l = np.log([0.1, 0.3, 0.4])
ref_l = np.log([0.3, 0.1, 0.4])

margin = beta * (pi_w - ref_w) - beta * (pi_l - ref_l)
losses = dpo_losses(pi_w, ref_w, pi_l, ref_l, beta)
L = losses.mean()  # the expectation over D, estimated by the mean over pairs
print("margins      ", margin)
print("sigma(margin)", sigmoid(margin))
print("losses       ", losses)
print("L_DPO (mean) ", L)
assert np.allclose(margin, [math.log(3), -math.log(3), 0.0])
assert np.allclose(sigmoid(margin), [0.75, 0.25, 0.5])
assert np.allclose(losses, [math.log(4 / 3), math.log(4), math.log(2)])
assert math.isclose(L, math.log(32 / 3) / 3)

# Appendix B groups the same four numbers differently:
# beta * ((pi_w - pi_l) - (ref_w - ref_l)).
code_margin = beta * ((pi_w - pi_l) - (ref_w - ref_l))
assert np.allclose(code_margin, margin)

# Eq. (1) is sigma of a reward difference, and a shift that depends only on x
# (beta * log Z(x)) cancels.
a, b, beta_logZ = 1.3, -0.4, 7.0
bt = math.exp(a) / (math.exp(a) + math.exp(b))
assert math.isclose(bt, sigmoid(a - b))
assert math.isclose(sigmoid((a + beta_logZ) - (b + beta_logZ)), sigmoid(a - b))

# Gradient with respect to each completion's policy log-probability,
# by central finite differences, against the weight printed in section 4.
h = 1e-6
d_w = (dpo_losses(pi_w + h, ref_w, pi_l, ref_l, beta)
       - dpo_losses(pi_w - h, ref_w, pi_l, ref_l, beta)) / (2 * h)
d_l = (dpo_losses(pi_w, ref_w, pi_l + h, ref_l, beta)
       - dpo_losses(pi_w, ref_w, pi_l - h, ref_l, beta)) / (2 * h)
r_w, r_l = beta * (pi_w - ref_w), beta * (pi_l - ref_l)
sec4 = beta * sigmoid(r_l - r_w)  # section 4 weight
a4 = beta * sigmoid(r_w - r_l)  # weight in Appendix A.4's last line
print("dL/dlogpi(y_w)", d_w)
print("dL/dlogpi(y_l)", d_l)
print("sec 4 weight  ", sec4)
print("A.4 weight    ", a4)
assert np.allclose(d_w, -sec4, atol=1e-8)
assert np.allclose(d_l, +sec4, atol=1e-8)
assert np.allclose(sec4, [0.125, 0.375, 0.25])
assert not np.allclose(d_w, -a4, atol=1e-8)

# Numerical stability: a margin of -800 (log-probabilities of long sequences
# are large negative numbers).
z = -800.0
with np.errstate(over="ignore", divide="ignore"):
    naive = -np.log(1.0 / (1.0 + np.exp(-z)))
stable = -log_sigmoid(z)
print("naive -log(sigmoid(-800)) =", naive, "; stable =", stable)
assert np.isinf(naive) and math.isclose(stable, 800.0)
print("all checks passed")
```

Run with `uv run --no-project --with numpy python check.py`. Output:

```text
margins       [ 1.09861229 -1.09861229  0.        ]
sigma(margin) [0.75 0.25 0.5 ]
losses        [0.28768207 1.38629436 0.69314718]
L_DPO (mean)  0.7890412047105388
dL/dlogpi(y_w) [-0.125 -0.375 -0.25 ]
dL/dlogpi(y_l) [0.125 0.375 0.25 ]
sec 4 weight   [0.125 0.375 0.25 ]
A.4 weight     [0.375 0.125 0.25 ]
naive -log(sigmoid(-800)) = inf ; stable = 800.0
all checks passed
```

The pair the implicit reward ranks correctly costs $\log(4/3)\approx0.29$, the
wrongly ranked pair costs $\log4\approx1.39$, and an untouched pair costs
$\log2\approx0.69$. The wrongly ranked pair also gets three times the gradient
weight (0.375 against 0.125). The finite differences match the §4 gradient.
They do not match the weight in Appendix A.4's last line,
$\beta\sigma(\hat r_\theta(x,y_w)-\hat r_\theta(x,y_l))$. That is my finding:
A.4's starting line (21) also swaps $y_w$ and $y_l$ inside $\sigma$ relative to
(7), although A.4 says it arrives at "the final form of the gradient from
Section 4". Use (7) and the §4 gradient.

## What the code must decide

- **Completion log-probability from tokens:** open (the first item under "Not
  defined in the paper"). A sum gives $\log\pi(y\mid x)$ as written (step 1),
  while a mean gives a different, length-normalized quantity.
- **Batch layout:** in the Appendix B listing, all completions sit in one
  batch of size `B`, and pairs are picked by index tensors `yw_idxs` and
  `yl_idxs` of shape `(T,)`. Here `T` is the number of pairs, not a sequence
  length.
- **Reduction over pairs:** the Appendix B listing returns per-pair `losses`
  of shape `(T,)` (`F.logsigmoid` "Applies element-wise", PyTorch v2.4.0
  docstring) and does not reduce them. The reduction that estimates the
  expectation in (7) is not shown: open. Whether "batch size of 64" (Appendix B) counts pairs or completions: open.
- **Numerical stability:** open. The paper does not discuss it (I searched the
  full text for "stab", "overflow" and "numeric"); the listing calls
  `F.logsigmoid` on the margin. The check above shows that computing $\sigma$
  first and then the log overflows to `inf` at a margin of $-800$, while
  `np.logaddexp(0, -z)`, the same loss in one step, returns 800.
- **What is detached:** $\pi_\text{ref}$ log-probabilities arrive as an input,
  `ref_logps`. The listing does not show how they were computed, or whether
  gradients were tracked for them: open.
  For `rewards = beta * (pi_logps - ref_logps).detach()`, see "Where the
  samples come from"; the paper does not say how `rewards` is used (my
  inference: for logging).
- **$\beta$ and the optimizer:** $\beta$ is given in its Symbols row. Unless
  noted otherwise, Appendix B uses a "batch size of 64 and the RMSprop
  optimizer with a learning rate of 1e-6", warmed up linearly from 0 over 150
  steps.
- **Choice of $\pi_\text{ref}$:** see its Symbols row.

## What the paper claims and what it shows

- **Equation (7) is maximum likelihood for a Bradley–Terry model whose reward
  is reparameterized through the policy:** derived (§4, Appendix A.2). This
  assumes the Bradley–Terry model (1). It also assumes the closed-form optimum
  (4), which Appendix A.1 derives "under any reward function $r(x, y)$,
  reference model $\pi_\text{ref}$ and a general non-parametric policy class",
  that is, assuming the policy can be any distribution, not one limited by a
  network's architecture.
- **DPO "implicitly optimizes the same objective as existing RLHF algorithms
  (reward maximization with a KL-divergence constraint)" (§1):** derived.
  §5's Theorem 1 holds "Under mild assumptions", which Appendix A.6 states as
  "Assume, we have a reference model, such that $\pi_\text{ref}(y\mid x) > 0$
  for all pairs of prompts $x$ and answers $y$ and a parameter $\beta > 0$."
  The derivation is about the optimum, not about what a finite number of
  gradient steps on a finite dataset reaches.
- **"This way, we fit an implicit reward using an alternative
  parameterization, whose optimal policy is simply $\pi_\theta$." (§4):**
  derived, same assumptions.
- **Statistical consistency:** "such as consistencies under suitable assumption
  of the preference data distribution [4]" (§4). The paper cites this to Bong
  and Rinaldo (ICML 2022) and does not show it.
- **The per-pair weight in the gradient matters:** suspected, with qualitative
  support. "Our experiments suggest the importance of this weighting, as a
  naïve version of this method without the weighting coefficient can cause the
  language model to degenerate (Appendix Table 3)." Table 3 shows samples from
  the unlikelihood baseline, which "simply optimizes the policy to maximize the
  probability assigned to $y_w$ and minimize the probability assigned to
  $y_l$", with "an optional coefficient $\alpha \in [0, 1]$ on the
  ‘unlikelihood’ term" (§6, Methods). That description has no
  $\pi_\text{ref}$ and no per-pair weight. Table 3 is captioned "In general, we find unlikelihood fails
  to generate meaningful responses for more complex problems such as
  summarization and dialogue." It reports no metric.
- **The §4 gradient formula:** derived in Appendix A.4, whose intermediate and
  final lines disagree with §4; see the check above. The §4 form is the one
  that matches (7).
- **DPO optimizes the RLHF objective more efficiently than PPO:** measured on
  IMDb sentiment with GPT-2-large (§6.1, Figure 2). "This sweep includes 22
  runs in total." The paper says "DPO's reward/KL tradeoff strictly dominates
  PPO." Each run was evaluated "After each 100 training steps until
  convergence"; Figure 2 (left), the IMDb frontier, plots single points with
  no error bars.
- **$\beta$ was barely tuned:** "we did not meaningfully tune DPO's $\beta$
  hyperparameter, so these results may underestimate DPO's potential" (§6.2,
  TL;DR).

## Sources

- R. Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290v3),
  v3. I read the TeX source (`main.tex`; `old_method.tex` is never `\input`
  and was not used) and the PDF. Every quote was confirmed in the PDF text. §3
  from the sentence after equation (3) through §5.1's Lemma 2, and Appendix
  A.4 with equations (21) and (22), were read as rendered images.
- A. Pal et al., [Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive](https://arxiv.org/abs/2402.13228v2),
  v2. Only the abstract was used, confirmed in the PDF text.
- The paper links no official DPO code, so no repository was opened. The code
  cited is the listing printed in Appendix B.
- PyTorch [v2.4.0](https://github.com/pytorch/pytorch/tree/v2.4.0), the latest
  release before v3 was posted (29 Jul 2024; v2.4.0 was published 24 Jul
  2024): the `Tensor.detach` docstring in
  [`torch/_tensor.py`](https://github.com/pytorch/pytorch/blob/v2.4.0/torch/_tensor.py)
  and the `logsigmoid` docstring in
  [`torch/nn/functional.py`](https://github.com/pytorch/pytorch/blob/v2.4.0/torch/nn/functional.py),
  downloaded raw at that tag.
- Named, not opened: Bradley and Terry (1952), the source of equation (1); Bong
  and Rinaldo, "Generalized results for the existence and consistency of the
  MLE in the Bradley-Terry-Luce model" (ICML 2022), reference [4]; Welleck et
  al., unlikelihood training (2019), reference [46]; Schulman et al., PPO.
