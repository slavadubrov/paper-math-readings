# Distillation: a softmax with a temperature, and the two-part loss that trains the small model

**Source:** Geoffrey Hinton, Oriol Vinyals, Jeff Dean,
[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531v1),
v1, §2, equation (1) and the training loss that §2 describes in words.
Other locations used: §1 (soft targets, transfer set, logits),
§2.1 equations (2)–(4), §3 (MNIST temperatures), §4.1 (speech weight and
temperatures). The paper links no code, so every implementation choice below
is either stated in the text or open.

## What it computes and what it changes

A classifier's last layer outputs one raw score per class. Equation (1) turns
the scores into probabilities, positive numbers that sum to 1, after first
dividing every score by a number $T$; a larger $T$ gives flatter
probabilities. This scores-to-probabilities step is called a softmax, and the
paper calls the scores logits, "the inputs to the final softmax" (§1). The
loss in §2 uses this
to train a small "distilled" model (the student) on two targets at once: the
flattened probabilities that a large, already trained "cumbersome" model (the
teacher) gives each example of the transfer set, the examples the small model
is trained on (§1), and the correct label.

What it changes:

- Ordinary training uses $T=1$ and only the label: "the normal training
  objective is to maximize the average log probability of the correct answer"
  (§1).
- §1 describes the method of Caruana and his collaborators as matching logits
  instead of probabilities: "they minimize the squared difference between the
  logits produced by the cumbersome model and the logits produced by the small
  model" (§1).
- In its simplest form, distillation instead trains the small model only on
  the large model's whole probability vector at a raised temperature (§2).
  When labels are known for all or some of the transfer set, §2 adds a second
  objective on the label at $T=1$. §2.1 derives logit matching as the
  high-temperature limit of distillation.

The reason to match the whole vector is in §1: "The relative probabilities of
incorrect answers tell us a lot about how the cumbersome model tends to
generalize." A BMW image may have only a very small chance of being taken for a
garbage truck, but that is still many times more likely than taking it for a
carrot (§1).

## The equation

$$
q_i = \frac{exp(z_i/T)}{\sum_j exp(z_j/T)} \tag{1}
$$

The training loss is **not printed as an equation**. §2 gives it in words:

> "we found that a better way is to simply use a weighted average of two
> different objective functions. The first objective function is the cross
> entropy with the soft targets and this cross entropy is computed using the
> same high temperature in the softmax of the distilled model as was used for
> generating the soft targets from the cumbersome model. The second objective
> function is the cross entropy with the correct labels. This is computed using
> exactly the same logits in softmax of the distilled model but at a
> temperature of 1. We found that the best results were generally obtained by
> using a condiderably [sic] lower weight on the second objective function.
> Since the magnitudes of the gradients produced by the soft targets scale as
> $1/T^2$ it is important to multiply them by $T^2$ when using both hard and
> soft targets. This ensures that the relative contributions of the hard and
> soft targets remain roughly unchanged if the temperature used for
> distillation is changed while experimenting with meta-parameters." (§2)

A cross entropy compares a target distribution $a$ with a prediction $b$ over
the same classes: $-\sum_i a_i\log b_i$ adds up, over the classes $i$, the
target's weight times the prediction's _surprise_ $-\log b_i$, and it is
smallest when $b$ puts its probability where $a$ does. Written as a formula,
for one case of the transfer set, the loss is below. **This formula is this
reading's, not the paper's**; the symbols $y$ (the correct label),
$\alpha$, $\mathcal L$ and the argument $(T)$ in
$q_i(T)$ and $p_i(T)$, for equation (1) evaluated at temperature $T$, are
mine. $q_i(T)$ is the student's probability for class $i$; $p_i(T)$ is the
teacher's, which the paper calls the soft target (§2.1):

$$
\mathcal L = \alpha\,T^2\Big(-\sum_i p_i(T)\log q_i(T)\Big)
           + (1-\alpha)\Big(-\log q_y(1)\Big),\qquad 0\le\alpha\le1
$$

The only printed formula that involves the soft-target term is its gradient,
in §2.1:

$$
\frac{\partial C}{\partial z_i} = \frac{1}{T}\left(q_i - p_i\right)
= \frac{1}{T}\left(\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
- \frac{e^{v_i/T}}{\sum_j e^{v_j/T}}\right) \tag{2}
$$

§2.1 then approximates it in two steps, (3) and (4); their conditions are
listed under _What the paper claims and what it shows_:

$$
\frac{\partial C}{\partial z_i} \approx \frac{1}{T}\left( \frac{1+z_i/T}{N + \sum_j z_j/T}
- \frac{1+v_i/T}{N + \sum_j v_j/T} \right) \tag{3}
$$

$$
\frac{\partial C}{\partial z_i} \approx \frac{1}{NT^2}\left( z_i - v_i \right) \tag{4}
$$

## Symbols

| Symbol | Shape or type | Meaning | Role | Defined at |
| ------ | ------------- | ------- | ---- | ---------- |
| $z_i$ | A real number. One per class, so a vector of length (number of classes) per case; `[batch, classes]` in code | Logit of class $i$; in the loss, the distilled (small) model's logit | Function input of eq. (1). In the loss, the output of the distilled model's last layer, computed from its trained weights | "converts the logit, $z_i$, computed for each class into a probability, $q_i$, by comparing $z_i$ with the other logits" (§2, before eq. 1); "each logit, $z_i$ of the distilled model" (§2.1) |
| $q_i$ | A number in $(0,1)$; the vector $q$ has one entry per class and sums to 1 | Output of eq. (1) for class $i$, for whichever logits it is applied to; in eq. (2), the distilled model's probability at temperature $T$; in the loss, $q_i(T)$ in the soft term and $q_y(1)$ in the hard term | Computed intermediate | Same §2 quote; §2.1 eq. (2) uses $q_i$ for the distilled model at temperature $T$ |
| $T$ | Positive scalar | Temperature: divides every logit before the softmax | Fixed setting | "where $T$ is a temperature that is normally set to 1. Using a higher value for $T$ produces a softer probability distribution over classes." (§2, after eq. 1). Values used: "at a temperature of 20" (§3, MNIST); "we tried temperatures of [1, 2, 5, 10]", with 2 printed in bold as "the best value" (§4.1, speech) |
| $i$ | Class index | The class whose probability is computed. It is free (not summed), so eq. (1) gives one number per class | Index | "computed for each class" (§2) |
| $j$ | Class index | The loop variable of the sum in the denominator | Index | The sum is the "comparing $z_i$ with the other logits" (§2). Its range is not printed; see the undefined list |
| $v_i$ | Real number, one per class | Logit of the cumbersome (teacher) model | Frozen model output | "If the cumbersome model has logits $v_i$ which produce soft target probabilities $p_i$ and the transfer training is done at a temperature of $T$" (§2.1) |
| $p_i$ | Number in $(0,1)$; vector sums to 1 | Soft target: eq. (1) applied to the teacher's logits $v$ at temperature $T$ (the second fraction in eq. 2) | Computed from the frozen model; a fixed target | Same §2.1 quote; "a soft target distribution for each case in the transfer set that is produced by using the cumbersome model with a high temperature in its softmax" (§2) |
| $C$ | Scalar per case | The soft-target cost whose gradient eq. (2) gives | Computed intermediate | "Each case in the transfer set contributes a cross-entropy gradient, $dC/dz_i$, with respect to each logit, $z_i$ of the distilled model." (§2.1) |
| $y$ (mine) | Integer class index per case | The correct label | Input data, only for cases where it is known | "When the correct labels are known for all or some of the transfer set" (§2) |
| $\alpha$ (mine) | Scalar in $[0,1]$ | Weight of the soft term; the hard term gets $1-\alpha$. The two are the weights of the "weighted average", and the weights of an average are non-negative and sum to 1 | Fixed setting | "a weighted average of two different objective functions" and "a condiderably [sic] lower weight on the second objective function" (§2); "used a relative weight of 0.5 on the cross-entropy for the hard targets" (§4.1) |
| $\mathcal L$ (mine) | Scalar per case | Training loss of the distilled model | Computed; minimized | The §2 paragraph quoted under _The equation_ |

**Not defined in the paper:**

- $N$ in equations (3)–(4). No sentence defines it; I read §2, §2.1 and
  searched the whole TeX source for `$N`. My derivation: equation (3) replaces
  each $e^{x}$ by $1+x$, so $\sum_j e^{z_j/T}$ becomes (number of terms in the
  sum) $+\sum_j z_j/T$. That makes $N$ the number of classes. The check below
  confirms equation (4) with $N=3$ for three classes.
- The range of $j$. It is not printed. My derivation: it must run over all
  classes, $i$ included, or $q$ would not sum to 1 as "a softer probability
  distribution over classes" (§2) requires.
- The formula for $C$. Only its gradient, equation (2), is printed. The words
  "the cross entropy with the soft targets" (§2) and equation (2) fit
  $C=-\sum_i p_i\log q_i$ with the natural log (with another log base,
  equation (2) would carry an extra constant factor). The KL divergence,
  $D_{\mathrm{KL}}(p\Vert q)$, is that cross entropy minus the teacher's own
  entropy $H(p)=-\sum_i p_i\log p_i$ (a measure of how spread out $p$ is). It
  is 0 only when $q=p$. Since $H(p)$ does not depend on the student, the KL
  divergence has the same gradient, so equation (2) alone does not pick
  between them; see the check.
- The value of $\alpha$. §2 gives none, only that the hard weight is
  considerably lower; I searched the whole paper for "weight", and the only
  value it gives is §4.1's 0.5. Both quotes are in the table. The 0.5 does not say
  relative to what, or whether it is applied before or after the $T^2$ factor.

## Read from the inside out

Equation (1), for one case with a vector $z$ of one logit per class:

1. $z_i/T$: divide each logit by the scalar $T$. Vector in, vector of the same
   length out. $T$ here is a divisor, not a power.
2. $exp(\cdot)$: $e$ raised to each value, entrywise. Every result is
   positive; the length is unchanged.
3. $\sum_j exp(z_j/T)$: a loop over all classes $j$ that adds the positive
   numbers from step 2 into one number per case. The index $j$ belongs to
   this sum and nothing else.
4. The fraction: entry $i$ divided by that sum gives $q_i$. Doing it for every
   $i$ gives a vector with one entry per class, all positive, summing to 1.
   The $=$ says $q_i$ is computed by this formula.
5. What $T$ does: dividing by a larger $T$ shrinks the gaps between the
   scores, so the probabilities move toward equal shares; $T=1$ is the ordinary
   softmax. The order of the classes never changes.

**Kind of statement (eq. 1):** a computation. Per case, a vector of logits goes
in and a vector of probabilities of the same length comes out; the only
intermediate of a different shape is the scalar sum, one per case.

The loss, read from the inside out (the formula is mine; each step cites the
paper's words):

1. $q_i(T)$ and $q_y(1)$: two softmaxes of the **same** student logits $z$,
   one at temperature $T$, one at 1 (the second objective in the §2 quote).
   Both have one entry per class.
2. $p_i(T)$: the teacher's softmax at the **same** $T$ (the first objective
   in the §2 quote). For how it is held fixed, see _Where the samples come
   from_.
3. The two cross entropies (defined under _The equation_). Each sum runs over
   classes and gives one number per case. When the target is the label, its
   weight is 1 on class $y$ and 0 elsewhere (a _one-hot_ vector), and the sum
   collapses to $-\log q_y(1)$.
4. $T^2$: a power (the square of $T$). It multiplies only the soft term.
   Multiplying a loss by a constant multiplies its gradient by the same
   constant, which is what "multiply them by $T^2$" asks of the soft-target
   gradients (my reading of "them"). $T^2$ is a factor of its own, not one of the
   two weights, so $\alpha$ and $1-\alpha$ still sum to 1. Its purpose, in the
   last sentence of the §2 quote, is that the two targets' relative
   contributions stay "roughly unchanged" when $T$ changes, so one $\alpha$
   can be kept across temperatures (my reading; the gradient-ratio bullet
   under _What the numbers show_ tests how far this holds).
5. The weighted average gives one number per case. Training then averages or
   sums it over the cases of a mini-batch (open; see _What the code must decide_)
   and minimizes it with respect to the distilled model's weights only.

Equation (2) is where the $1/T$ comes from. $z_i$ enters the softmax as
$u_i=z_i/T$, whose derivative is $1/T$. Since
$\log q_k=u_k-\log\sum_j e^{u_j}$, the derivative $\partial\log q_k/\partial u_i$
is $1-q_i$ when $k=i$ and $-q_i$ otherwise. Summing $-p_k$ times that over $k$
gives the derivative of $-\sum_k p_k\log q_k$ with respect to $u_i$:
$-p_i+q_i\sum_k p_k=q_i-p_i$, the last step because the targets sum to 1. The chain rule
multiplies this by $1/T$ (my derivation). Under the two conditions of
equations (3)–(4) (see _What the paper claims and what it shows_), $q_i-p_i$
is itself about $(z_i-v_i)/(NT)$. Together that is the
$\frac{1}{NT^2}(z_i-v_i)$ of equation (4), and the source of the "$1/T^2$" in
§2.

**Kind of statement (loss):** an objective, minimized over the distilled
model's weights, on the cases of the transfer set; whether they are summed or
averaged is open (step 5). Equation (2) is an
exact derivative; equations (3)–(4) are approximations, as their $\approx$
says.

## Where the samples come from

There is no expectation symbol; the loss runs over the cases of the transfer
set (§2.1).

- **Data:** the transfer set is a fixed dataset chosen before training: "The
  transfer set that is used to train the small model could consist entirely of
  unlabeled data [1] or we could use the original training set" (§1). No model
  generates it. In the MNIST experiments it is the training set, then variants
  with "all examples of the digit 3" omitted or "only the 7s and 8s" (§3). The
  teacher was trained on images "jittered by up to two pixels", but "the
  transfer set does not contain any translations" (§3).
- **Soft targets:** produced by the cumbersome model, which is trained first
  ("Once the cumbersome model has been trained", §1) and not updated during
  distillation. The gradient is taken with respect to the distilled model's
  logits (the $z_i$ row under _Symbols_). For an ensemble teacher, "we can
  use an arithmetic or geometric mean of their individual predictive
  distributions as the soft targets" (§1).
- **Labels:** fixed with the data; for which cases have one, see the $y$ row
  under _Symbols_.

## Check with small numbers

A two-class case comes first because it can be checked by hand: logits
$(2,0)$ give $q_0=e^{2/T}/(e^{2/T}+1)$. The main inputs are three classes
with student logits $z=(0,2\ln2,4\ln2)$, whose exponentials are 1, 4, 16 at $T=1$ and 1, 2, 4 at $T=2$, so the softmaxes are
exactly $(1,4,16)/21$ and $(1,2,4)/7$. Teacher logits $(0,0,2\ln3)$ give
$p=(0.2,0.2,0.6)$ at $T=2$. Two cases make the batch mean average two items;
the second case's label is not the teacher's top class. The check covers
equation (1) at four temperatures, both cross entropies, the $T^2$ factor, the
weights $\alpha$ and $1-\alpha$ (each case's loss must lie between its two
terms), the batch mean, equation (2), cross entropy against KL, and the
$1/T^2$ claim. These equations have no min, max, clip or indicator, so there
are no branches to cover. The one case split, labeled versus unlabeled cases,
has no rule in the paper (see below), so it is not tested. $\alpha=0.9$ is an
illustration, not a value from the paper.

```python
import numpy as np

ln2, ln3 = np.log(2), np.log(3)


def softmax_T(z, T):  # eq (1): q_i = exp(z_i/T) / sum_j exp(z_j/T), over the class axis
    e = np.exp(z / T - np.max(z / T, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def ce(target, logits, T):  # cross entropy -sum_i target_i * log q_i(T), one number per case
    return -(target * np.log(softmax_T(logits, T))).sum(axis=-1)


# --- eq (1) with two logits (2, 0): q_0 = e^(2/T) / (e^(2/T) + 1), by hand ---
assert np.isclose(softmax_T(np.array([2.0, 0.0]), 1.0)[0], np.e**2 / (np.e**2 + 1))
for T in (1, 2, 5):
    print(f"z=(2,0)  T={T}  q={np.round(softmax_T(np.array([2.0, 0.0]), T), 2)}")

# --- eq (1) on one case, N = 3 classes -------------------------------------
z = np.array([0.0, 2 * ln2, 4 * ln2])  # student logits; exp(z) = 1, 4, 16
q1, q2 = softmax_T(z, 1), softmax_T(z, 2)
assert np.allclose(q1, np.array([1, 4, 16]) / 21)  # T = 1: ordinary softmax
assert np.allclose(q2, np.array([1, 2, 4]) / 7)  # T = 2: exp(z/2) = 1, 2, 4
for T in (1, 2, 5, 20):
    q = softmax_T(z, T)
    H = -(q * np.log(q)).sum()
    print(f"T={T:>2}  q={np.round(q, 4)}  sum={q.sum():.4f}  entropy={H:.4f} (max {np.log(3):.4f})")

# --- the section 2 loss on a batch of two labelled cases ---------------------
Z = np.array([[0.0, 2 * ln2, 4 * ln2],  # student logits, case 1
              [0.0, 2 * ln2, 4 * ln2]])  # case 2: same student logits
V = np.array([[0.0, 0.0, 2 * ln3],  # teacher logits, case 1; exp(v/2) = 1, 1, 3
              [2 * ln3, 0.0, 0.0]])  # case 2: teacher prefers class 0
y = np.array([2, 1])  # correct labels (case 2's label differs from the teacher's top class)
T = 2.0
P = softmax_T(V, T)  # soft targets from the frozen teacher at temperature T
assert np.allclose(P, [[0.2, 0.2, 0.6], [0.6, 0.2, 0.2]])
soft = ce(P, Z, T)  # first objective: CE with soft targets, student at the SAME T
hard = ce(np.eye(3)[y], Z, 1.0)  # second objective: CE with correct labels, same logits, T = 1
assert np.allclose(hard, [-np.log(16 / 21), -np.log(4 / 21)])
alpha = 0.9  # illustration only: the paper gives no value for alpha
per_case = alpha * T**2 * soft + (1 - alpha) * hard  # weighted average of T^2*soft and hard
assert np.all(np.minimum(T**2 * soft, hard) <= per_case) and np.all(per_case <= np.maximum(T**2 * soft, hard))
L = per_case.mean()  # mean over cases: an open choice
print("soft CE per case:", np.round(soft, 4), " hard CE per case:", np.round(hard, 4))
print(f"loss = mean(alpha*T^2*soft + (1-alpha)*hard) = {L:.4f}")

# --- eq (2): dC/dz_i = (q_i - p_i)/T, checked by central finite differences --
def num_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        d = np.zeros_like(x); d[i] = h
        g[i] = (f(x + d) - f(x - d)) / (2 * h)
    return g

p, v = P[0], V[0]
g_soft = num_grad(lambda zz: ce(p, zz, T), z)
assert np.allclose(g_soft, (softmax_T(z, T) - p) / T, atol=1e-6)
g_hard = num_grad(lambda zz: ce(np.eye(3)[2], zz, 1.0), z)
assert np.allclose(g_hard, softmax_T(z, 1) - np.eye(3)[2], atol=1e-6)  # no T anywhere
print("eq (2) gradient:", np.round(g_soft, 4), " hard-label gradient:", np.round(g_hard, 4))

# CE with soft targets and KL(p || q) differ by the teacher entropy H(p), a constant in z
kl = lambda zz: (p * (np.log(p) - np.log(softmax_T(zz, T)))).sum()
H_p = -(p * np.log(p)).sum()
assert np.isclose(ce(p, z, T) - kl(z), H_p)
assert np.allclose(num_grad(kl, z), g_soft, atol=1e-6)
print(f"CE={ce(p, z, T):.4f}  KL={kl(z):.4f}  difference={ce(p, z, T) - kl(z):.4f} = H(p)={H_p:.4f}")

# --- the 1/T^2 claim: eqs (3) and (4) with zero-meaned logits ----------------
zc, vc = z - z.mean(), v - v.mean()  # zero-meaned per case, as eq (4) assumes
N = 3  # number of classes: the count of 1s that e^x ~ 1 + x adds up in sum_j (my reading)
print(" T   |grad soft|   T^2*grad soft                 eq (4) (zc-vc)/N")
for TT in (1, 2, 5, 10, 20, 100):
    g = (softmax_T(zc, TT) - softmax_T(vc, TT)) / TT  # eq (2)
    print(f"{TT:>4}  {np.linalg.norm(g):.6f}    {np.round(TT**2 * g, 4)}   {np.round((zc - vc) / N, 4)}")
assert np.allclose(100**2 * (softmax_T(zc, 100) - softmax_T(vc, 100)) / 100, (zc - vc) / N, atol=2e-3)
g1, g2 = [(softmax_T(zc, TT) - softmax_T(vc, TT)) / TT for TT in (1, 2)]
print(f"T=1: |eq (4)| / |eq (2)| = {np.linalg.norm((zc - vc) / N) / np.linalg.norm(g1):.2f};"
      f" T=1 to T=2: |eq (2)| falls {np.linalg.norm(g1) / np.linalg.norm(g2):.2f}-fold (1/T^2 gives 4)")

# without zero-meaning, softmax ignores a shift, so the limit is (z - mean z) - (v - mean v), not z - v
zs = zc + 5.0
g = 100**2 * (softmax_T(zs, 100) - softmax_T(vc, 100)) / 100
assert np.allclose(g, (zc - vc) / N, atol=2e-3) and not np.allclose(g, (zs - vc) / N, atol=0.1)
print("shifted student logits, T=100, T^2*grad:", np.round(g, 4), " (zs-vc)/N:", np.round((zs - vc) / N, 4))

# ratio of soft to hard gradient size, with and without the T^2 factor
print(" T   |T^2*soft grad| / |hard grad|   |soft grad| / |hard grad|")
r_with, r_without = {}, {}
for TT in (1, 2, 5, 10, 20):
    gs = (softmax_T(z, TT) - softmax_T(v, TT)) / TT
    r_with[TT] = np.linalg.norm(TT**2 * gs) / np.linalg.norm(g_hard)
    r_without[TT] = np.linalg.norm(gs) / np.linalg.norm(g_hard)
    print(f"{TT:>4}  {r_with[TT]:.4f}                      {r_without[TT]:.4f}")
assert all(0.9 < r_with[t] < 1.05 for t in (5, 10, 20))  # flat only once T is large
print(f"T=1 to T=20: with T^2 the ratio grows {r_with[20] / r_with[1]:.2f}-fold;"
      f" without it, it falls {r_without[1] / r_without[20]:.0f}-fold")

# --- numbers printed in the paper -----------------------------------------------
print(f"speech, share of ensemble frame-accuracy gain kept: {(60.8 - 58.9) / (61.1 - 58.9):.3f}")
print("all asserts passed")
```

Run with `timeout 60 uv run --no-project --with "numpy>=2,<3" python -I check.py`.
Output:

```text
z=(2,0)  T=1  q=[0.88 0.12]
z=(2,0)  T=2  q=[0.73 0.27]
z=(2,0)  T=5  q=[0.6 0.4]
T= 1  q=[0.0476 0.1905 0.7619]  sum=1.0000  entropy=0.6680 (max 1.0986)
T= 2  q=[0.1429 0.2857 0.5714]  sum=1.0000  entropy=0.9557 (max 1.0986)
T= 5  q=[0.2463 0.325  0.4288]  sum=1.0000  entropy=1.0735 (max 1.0986)
T=20  q=[0.3105 0.3328 0.3567]  sum=1.0000  entropy=1.0970 (max 1.0986)
soft CE per case: [0.9755 1.53  ]  hard CE per case: [0.2719 1.6582]
loss = mean(alpha*T^2*soft + (1-alpha)*hard) = 4.6065
eq (2) gradient: [-0.0286  0.0429 -0.0143]  hard-label gradient: [ 0.0476  0.1905 -0.2381]
CE=0.9755  KL=0.0252  difference=0.9503 = H(p)=0.9503
 T   |grad soft|   T^2*grad soft                 eq (4) (zc-vc)/N
   1  0.122290    [-0.0433  0.0996 -0.0563]   [-0.218   0.2441 -0.0262]
   2  0.053452    [-0.1143  0.1714 -0.0571]   [-0.218   0.2441 -0.0262]
   5  0.011305    [-0.1764  0.217  -0.0407]   [-0.218   0.2441 -0.0262]
  10  0.003059    [-0.1976  0.2311 -0.0335]   [-0.218   0.2441 -0.0262]
  20  0.000793    [-0.208   0.2378 -0.0298]   [-0.218   0.2441 -0.0262]
 100  0.000033    [-0.216   0.2429 -0.0269]   [-0.218   0.2441 -0.0262]
T=1: |eq (4)| / |eq (2)| = 2.68; T=1 to T=2: |eq (2)| falls 2.29-fold (1/T^2 gives 4)
shifted student logits, T=100, T^2*grad: [-0.216   0.2429 -0.0269]  (zs-vc)/N: [1.4487 1.9108 1.6405]
 T   |T^2*soft grad| / |hard grad|   |soft grad| / |hard grad|
   1  0.3963                      0.3963
   2  0.6928                      0.1732
   5  0.9158                      0.0366
  10  0.9913                      0.0099
  20  1.0281                      0.0026
T=1 to T=20: with T^2 the ratio grows 2.59-fold; without it, it falls 154-fold
speech, share of ensemble frame-accuracy gain kept: 0.864
all asserts passed
```

What the numbers show:

- Raising $T$ flattens $q$ toward equal shares: logits $(2,0)$ go from
  $(0.88, 0.12)$ at $T=1$ to $(0.6, 0.4)$ at $T=5$. For three classes the
  entropy rises from 0.6680 at $T=1$ to 0.9557, 1.0735 and 1.0970, against a
  maximum of 1.0986. The class order never changes.
- Equation (2) is exact; checked by finite differences.
- For zero-meaned logits (each case's student logits, and separately its
  teacher logits, minus their own mean, so that each set sums to 0; `zc` and
  `vc` in the check), $T^2$ times the soft gradient approaches equation (4)'s
  $(z-v)/N$ only when $T$ is large compared with the logits; at $T=1$ and
  $T=2$ it is far from it.
  So the $1/T^2$ scaling of §2.1 does not hold at small $T$: at $T=1$
  equation (4) overstates the gradient 2.68-fold, and from $T=1$ to $T=2$ the
  gradient falls 2.29-fold, not 4-fold.
- Zero-meaning the logits costs nothing. Shifting all student logits by 5
  leaves the gradient unchanged, because the softmax ignores a constant added to
  every logit. Without it, equation (4)'s $z_i-v_i$ would be wrong (my
  derivation).
- With the $T^2$ factor, the ratio of soft to hard gradient size stays
  between 0.9158 and 1.0281 for $T$ from 5 to 20, but is 0.3963 at $T=1$ and
  0.6928 at $T=2$, where the $1/T^2$ scaling does not hold (the point on
  equation (4) above). From $T=1$ to $T=20$ it grows 2.59-fold; without
  $T^2$ it falls 154-fold. So in this example the paper's "roughly unchanged"
  (§2) holds once $T$ is large compared with the logits, not at every temperature.
- The CE and KL lines confirm the entry for $C$ under _Not defined in the
  paper_.

## What the code must decide

The softmax axis, the shared logits and the shared temperature are fixed by the
text (see _Read from the inside out_).

- **Temperature at test time:** stated: "after it has been trained it uses a
  temperature of 1" (§2).
- **Value of $\alpha$:** open (see the undefined list).
- **Where $T^2$ goes:** my reading, in step 4 of the loss under _Read from the
  inside out_.
- **Cross entropy or KL for the soft term:** the paper says cross entropy. A
  KL loss logs a different value; for the gradient, see
  the entry for $C$ under _Not defined in the paper_.
- **Reduction over cases:** mean or sum over the mini-batch: open. The paper
  gives none for the distillation loss (searched for "average", "mean",
  "sum" and "batch"); its only mini-batch average is the JFT baseline's
  training: "Each replica computes the average gradient on its current
  mini-batch" (§5.1).
- **Cases without a label:** open (see the $y$ row under _Symbols_); dropping
  the hard term for those cases is one choice.
- **Ensemble soft targets:** either mean, as quoted under _Where the samples
  come from_. Whether each member's distribution is taken at temperature $T$
  before averaging is open.
- **Soft targets computed per batch or precomputed once:** open.
- **Numerical stability:** open. Computing $\log q_i$ directly as
  $z_i/T-\log\sum_j e^{z_j/T}$, instead of taking the log of $q_i$, avoids
  `log(0)` when a probability underflows to 0.

## What the paper claims and what it shows

- **A higher $T$ gives a softer distribution:** stated with eq. (1); follows
  from the formula (the check shows it).
- **A weighted average of the two objectives beats modifying the soft targets
  with the labels:** measured but not reported: "we found that a better way"
  (§2), with no numbers.
- **A considerably lower weight on the hard term works best:** "We found that
  the best results were generally obtained" (§2); no numbers in §2. For the
  one reported weight, see the value of $\alpha$ under _Not defined in the
  paper_.
- **Soft-target gradients scale as $1/T^2$:** derived in §2.1, under two
  assumptions: "If the temperature is high compared with the magnitude of the
  logits" (eq. 3) and logits "zero-meaned separately for each transfer case"
  (eq. 4). Zero-meaning, and how close the scaling is at small $T$: see
  _What the numbers show_.
- **Logit matching is a special case:** derived: "in the high temperature
  limit, distillation is equivalent to minimizing $1/2(z_i-v_i)^2$, provided
  the logits are zero-meaned separately for each transfer case" (§2.1).
- **Intermediate temperatures work best when the student is very small:**
  measured. The paper says "We show" that "intermediate temperatures work
  best" (§2.1), from one MNIST experiment: with 30 units per layer,
  "temperatures in the range 2.5 to 4 worked significantly better than higher
  or lower temperatures" (§3). No per-temperature errors or run counts are
  given. Only the mechanism is hedged: the result "strongly suggests that
  ignoring the large negative logits can be helpful" (§2.1).
- **Distillation transfers most of an ensemble's gain:** measured in Table 1:
  frame accuracy 58.9% (baseline), 61.1% (10-model ensemble), 60.8%
  (distilled). "More than 80% of the improvement" (§4.1) recomputes to 0.864.
  These are single numbers, with no $\pm$, run counts or intervals.
- **Soft targets regularize:** MNIST test errors fall from 146 to 74 when the
  small net also matches soft targets, at the temperature the $T$ row gives
  for MNIST (§3). This is a single measurement with no error bars.

## Sources

- Hinton, Vinyals, Dean,
  [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531v1),
  v1: the PDF (§2, §2.1 and Table 1 read on the rendered pages, the rest as
  extracted text) and
  the TeX source from the v1 e-print (`distillation11.tex`, `speech11.tex`,
  `prelim_mnist11.tex`, `introduction11.tex`, `main11.tex`).
- Not opened: C. Buciluǎ, R. Caruana, A. Niculescu-Mizil, "Model compression",
  KDD 2006 (the paper's [1]); J. Li, R. Zhao, J. Huang, Y. Gong, "Learning
  small-size DNN with output-distribution-based criteria", Interspeech 2014
  (the paper's [8]).
- NumPy 2.5.4, run locally for the check above.
