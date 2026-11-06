# DeepSeek LLM: four curves that turn a compute budget into a learning rate, a batch size, a model size and a token count

**Source:** DeepSeek-AI (Xiao Bi, Deli Chen, Guanting Chen et al.),
[DeepSeek LLM: Scaling Open-Source Language Models with Longtermism](https://arxiv.org/abs/2401.02954v1),
v1, §3.1, equation (1), and §3.2, equation (4). Other locations used: §2.2
(Table 2), §2.3, the §3 introduction, §3.2 equations (2) and (3) and Table 3,
§3.3 (Table 4), Figures 2–5, Appendix A.2. Some units are read from the axis
labels of Figures 2–4, which are images in the PDF; those are marked
*(figure image)*. The paper links no code for these experiments.

## What it computes and what it changes

Before training a model you must choose how much compute to spend. Compute is
counted in FLOPs, the number of floating-point additions and multiplications
the training run performs. Equation (1) takes that budget $C$ and returns a
learning rate and a batch size. Equation (4) takes the same $C$ and says how to
split it between a bigger model and more training tokens. All four lines are
curves fitted to measurements of many small training runs. For the 7B and
67B, Table 2 lists learning rates and batch sizes close to what (1) gives
(for the 67B, only when $M$ is taken from Figure 4(b); check 7), and its
caption says "We choose the hyper-parameters based on our findings in
Section 3". The paper does not say that (4) chose their sizes or
token counts: the sizes are "two widely used large-scale model
configurations" (§1), and both models were trained on 2.0T tokens (Table 2).
Each small run is scored by its _generalization
error_: the trained model's error on held-out text, lower being better. The
runs behind (4) form an _IsoFLOP_ study, a method the paper takes from
Chinchilla: at each fixed budget, train models of several sizes, each for as
many tokens as the budget allows, and find the size where the error is
lowest. Whether that is the best run or the minimum of a curve drawn through
the runs is open (see _Not defined in the paper_).

What changes, for (4): the §3 introduction gives the earlier way to count
compute, "C can be approximated as C = 6ND", where $N$ is the number of
parameters. DeepSeek replaces $N$ with $M$, the FLOPs the model spends per
token. $M$ also counts attention over the sequence and leaves out the
vocabulary layers (the embedding that turns token ids into vectors and the
output layer that turns vectors into scores over the vocabulary). It also
replaces $C=6ND$ with $C=MD$. For (1): the paper says earlier works modelled
the best batch size as a function of the loss alone (quoted under _What the
paper claims and what it shows_). DeepSeek fits both the batch size and the
learning rate as functions of the compute budget alone.

## The equation

$$
\begin{aligned}
\eta_{\mathrm{opt}} &= 0.3118\cdot C^{\,-0.1250} \\
B_{\mathrm{opt}} &= 0.2920\cdot C^{\,0.3271}
\end{aligned}
\qquad (1)
$$

$$
\begin{aligned}
M_{\mathrm{opt}} &= M_{\mathrm{base}}\cdot C^{a}, & M_{\mathrm{base}} &= 0.1715, & a &= 0.5243 \\
D_{\mathrm{opt}} &= D_{\mathrm{base}}\cdot C^{b}, & D_{\mathrm{base}} &= 5.8316, & b &= 0.4757
\end{aligned}
\qquad (4)
$$

Two neighbouring equations in §3.2 define symbols that (4) uses, so they are
printed here as well:

$$
\begin{aligned}
6N_1 &= 72\,n_{\mathrm{layer}}\,d_{\mathrm{model}}^2 \\
6N_2 &= 72\,n_{\mathrm{layer}}\,d_{\mathrm{model}}^2 + 6\,n_{\mathrm{vocab}}\,d_{\mathrm{model}} \\
M &= 72\,n_{\mathrm{layer}}\,d_{\mathrm{model}}^2 + 12\,n_{\mathrm{layer}}\,d_{\mathrm{model}}\,l_{\mathrm{seq}}
\end{aligned}
\qquad (2)
$$

$$
M_{\mathrm{opt}}(C), D_{\mathrm{opt}}(C) = \underset{M, D\,\mathrm{s.t.}\,C=MD}{\mathrm{argmin}}\; L(N, D)
\qquad (3)
$$

## Symbols

Every quantity here is a single number (a scalar), not a vector or tensor.

| Symbol | Shape or type | Meaning | Role | Defined at |
| ------ | ------------- | ------- | ---- | ---------- |
| $C$ | scalar, FLOPs | Compute budget: total training FLOPs of one run, counted as $M\cdot D$ | Function input, chosen by the experimenter before a run; in the paper's runs it ranges over 1e17–3e20 | "model performance can be predictably improved with increases in compute budget C, model scale N, and data scale D" (§3); "the compute budget C can be simply expressed as C=MD" (§3.2). Unit: Figure 2(a) subcaption "1e17 FLOPs (177M FLOPs/token)" |
| $\eta_{\mathrm{opt}}$ | scalar, no unit | The learning rate the fit recommends for budget $C$ | Computed output | "We then fitted the batch size B and learning rate η with respect to the compute budget C." (§3.1) |
| $B_{\mathrm{opt}}$ | scalar, tokens | The batch size the fit recommends for budget $C$ | Computed output | Same quote (§3.1). Unit: Figure 3(a) axis "Optimal Batch Size (Tokens)" (figure image) |
| $0.3118$, $0.2920$ | scalar | Coefficients: the values of $\eta_{\mathrm{opt}}$ and $B_{\mathrm{opt}}$ at $C=1$ | Fixed setting (fitted constant) | "The final formulae we fitted for batch size and learning rate are as follows" (§3.1, equation (1)) |
| $-0.1250$, $0.3271$ | scalar | Exponents: how fast $\eta_{\mathrm{opt}}$ falls and $B_{\mathrm{opt}}$ grows with $C$ | Fixed setting (fitted constant) | Same quote (§3.1) |
| $_{\mathrm{opt}}$ in (1) | label | *Fitted through the near-optimal runs*: a run is near-optimal if its error is within 0.25% of the best run | Label | "we regarded the parameters used by models whose generalization error exceeded the minimum by no more than 0.25% as near-optimal hyperparameters" (§3.1) |
| $_{\mathrm{opt}}$ in (4) | label | *The split that gives the lowest error at this budget* | Label | "Given a computing budget C=MD, find the optimal model scale M_opt and data scale D_opt that minimize the generalization error of the model." (§3.2) |
| $M$ | scalar, FLOPs per token | Model scale: training FLOPs spent per token, counting attention, not counting the vocabulary layers | Computed intermediate, from the architecture via (2) | "we introduced a new model scale representation: non-embedding FLOPs/token M. M includes the computational overhead of attention operation but does not take into account the vocabulary computation." (§3.2) |
| $M_{\mathrm{opt}}$ | scalar, FLOPs per token | The model scale the fit recommends for budget $C$ | Computed output | "The specific formulae for the optimal non-embedding FLOPs/token M_opt and optimal tokens D_opt are as follows" (§3.2) |
| $D$, $D_{\mathrm{opt}}$ | scalar, tokens | Data scale: number of training tokens; $D_{\mathrm{opt}}$ is the recommended count | $D$: fixed setting of a run; $D_{\mathrm{opt}}$: computed output | "The data scale D can be consistently represented by the number of tokens in the dataset." (§3.2); "optimal tokens D_opt", same sentence as $M_{\mathrm{opt}}$ |
| $a$, $b$ | scalar | Exponents of the model-size and data-size curves | Fixed setting (fitted constant) | "finding model scaling exponent a and data scaling exponent b that satisfy N_opt ∝ C^a and D_opt ∝ C^b, respectively" (§3.2). The sentence writes $N_{\mathrm{opt}}$; in (4) and in Table 4's header, "N_opt(M_opt) ∝ C^a", $M_{\mathrm{opt}}$ takes its place |
| $M_{\mathrm{base}}$, $D_{\mathrm{base}}$ | scalar | Coefficients: $M_{\mathrm{opt}}$ and $D_{\mathrm{opt}}$ at $C=1$ | Fixed setting (fitted constant) | Only their values are printed, in equation (4); no sentence names them |
| $n_{\mathrm{layer}}$, $d_{\mathrm{model}}$, $n_{\mathrm{vocab}}$, $l_{\mathrm{seq}}$ | integers | Layers, hidden width, vocabulary size, sequence length | Fixed setting (architecture) | "where n_layer represents the number of layers, d_model represents the model width, n_vocab is the vocabulary size, and l_seq is the sequence length." (§3.2) |
| $N$, $N_1$, $N_2$ | scalar, parameter count | Model scale counted as parameters: $N_1$ without the embedding layers, $N_2$ with them | Computed intermediate (the older measure that $M$ replaces) | "When model scale N is represented by model parameters and data scale D by the number of tokens, C can be approximated as C = 6ND." (§3); "with non-embedding parameters N1 (Kaplan et al., 2020) and complete parameters N2 (Hoffmann et al., 2022)" (§3.2) |
| $L$ | scalar | Generalization error: the error of a trained model on held-out text | Measured value, one per training run | "the generalization error was calculated on an independent validation set, distributed similarly to the training set and containing 100M tokens" (§3.2); "The metric in IsoFLOP curve is bits-per-byte on the validation set." (Figure 4 caption) |

**Not defined in the paper:**

- **The unit of $B$.** The text never says whether $B$ counts tokens or
  sequences. Only the axis labels of Figure 2 and Figure 3(a) (figure images)
  say "Tokens", while Table 2 lists a "Sequence Batch Size". Check 7 below
  supports tokens (see _What the numbers show_).
- **Which learning rate $\eta$ is.** §2.3 describes the schedule: "the
  learning rate of the model reaches its maximum value after 2000 warmup
  steps, and then decreases to 31.6% of the maximum value after processing
  80% of the training tokens. It further reduces to 10% of the maximum value
  after 90% of the tokens." §3.1, Table 2 and Figure 3 never say which point
  of that schedule $\eta$ refers to.
- **How $C$ is counted in (1).** §3.1 does not restate it. The §3
  introduction introduces $C=MD$ for the model/data study, and the axis of
  Figure 3 reads "Non-Embedding Training FLOPs" (figure image). Check 7
  reproduces the 7B's printed values with $C=M\cdot D$.
- **The fitting procedure** for (1) and (4): the loss being minimized,
  whether the fit is done on logarithms, and how points are weighted. Looked
  in §3.1, §3.2, the captions of Figures 3–5 and Appendix A.2.
- **How each budget's best split is located** for (4). §3.2 says only that
  the model/data scaling curves "are fitted by using the optimal model/data
  allocation for each compute budget". Figure 4(a) draws a dashed curve through each budget's
  points (figure image); the paper says neither whether the best split is the
  lowest-error run or the minimum of that curve, nor what kind of curve it is.
- **The group over which the 0.25% minimum is taken** in (1), for example
  per budget or per budget and model size.
- **Bits-per-byte.** It is named (Figure 4 and 5 captions, and §5.1 "which
  means calculating the bits-per-byte on the test corpus") but has no
  formula anywhere in the paper.
- **The error used in §3.1.** The text speaks of "generalization error", but
  the caption of Figure 2 reads "Training loss w.r.t. batch size and learning
  rate with 1e17 and 1e20 FLOPs." The paper does not reconcile the two.
- **The dataset of the §3.1 runs.** It is not named. For (4), Table 4's row
  "Ours (Current Data)", 0.524 and 0.476, equals $a$ and $b$ rounded to three
  places (check 4); the text does not say in words which dataset (4) was fitted on.
- **$M$ for the 67B.** The label in Figure 4(b) (figure image) reads
  "4.3e11 FLOPs/Token", which is not what equation (2) gives for Table 2's
  shape; the paper does not explain the gap (values under _What the numbers
  show_).
- **$N$ in (3).** The argmin runs over $M$ and $D$, but the loss is printed as
  $L(N, D)$; the paper does not comment.

## Read from the inside out

**Equation (1).**

1. $C^{-0.1250}$: $C$ is a positive scalar. The superscript is a power, not
   a label. A negative power means a reciprocal: $C^{-0.125}=1/C^{0.125}$,
   the eighth root of $1/C$. The result is a scalar. No sum, mean or axis is
   involved; in NumPy it is `C ** -0.125`, applied elementwise if `C` is an
   array of budgets.
2. $0.3118\cdot(\ldots)$: multiply by the coefficient. The result is
   $\eta_{\mathrm{opt}}$, a scalar.
3. $0.2920\cdot C^{0.3271}$: the same two steps with a positive power, giving
   $B_{\mathrm{opt}}$ in tokens.
4. The sign $=$ here states a fitted curve, not an identity. Taking
   logarithms turns each line into a straight line,
   $\log\eta_{\mathrm{opt}}=\log0.3118-0.125\log C$: the exponent is the slope
   and the coefficient sets the height. That is why Figure 3 draws the fit as
   a straight dotted line on logarithmic axes. A curve of the form
   $k\cdot C^{e}$ is called a **power law**.
5. In plain terms: ten times more compute multiplies the learning rate by
   0.750 and the batch size by 2.124 (check 3).

**Equation (4).**

1. $C^{a}$ and $C^{b}$: powers again, giving scalars.
2. $M_{\mathrm{base}}\cdot C^a$ and $D_{\mathrm{base}}\cdot C^b$: scale by the
   coefficients. The results are $M_{\mathrm{opt}}$ in FLOPs per token and
   $D_{\mathrm{opt}}$ in tokens. The entries after the commas only list the
   four constants' values.
3. The sign $\propto$ in the §3.2 sentence that defines $a$ and $b$ means
   *proportional to*: equal up to a constant factor. Equation (4) supplies
   those factors as $M_{\mathrm{base}}$ and $D_{\mathrm{base}}$.
4. The two lines are tied together (my derivation). Every run spends
   $C=M\cdot D$, so the recommended pair must also multiply to $C$:
   $M_{\mathrm{opt}}D_{\mathrm{opt}}=M_{\mathrm{base}}D_{\mathrm{base}}\,C^{a+b}$
   equals $C$ only if $a+b=1$ and $M_{\mathrm{base}}D_{\mathrm{base}}=1$. The
   printed constants give $a+b=1.0000$ and
   $M_{\mathrm{base}}D_{\mathrm{base}}=1.00012$ (check 4). So (4) holds two
   free numbers, not four: given $a$ and $M_{\mathrm{base}}$, the data line
   follows.
5. In plain terms: ten times more compute multiplies the model's FLOPs per
   token by 3.344 and the token count by 2.990 (check 3). Because $a+b=1$
   (step 4), the two factors multiply back to 10 (check 3): of each factor of
   10 in compute, a factor of 3.344 goes to $M$ and 2.990 to $D$.

**Equations (2) and (3), which (4) relies on.** In (2), $d_{\mathrm{model}}^2$
is a square (a power) while _layer_, _model_, _vocab_ and _seq_ are labels.
The first term of $M$ equals $6N_1$, the parameter-based count on the first
line of (2). The second term, $12\,n_{\mathrm{layer}}d_{\mathrm{model}}l_{\mathrm{seq}}$,
is what the paper adds for attention; it grows with the sequence length. The
paper gives no derivation of the constants 72 and 12. In (3), argmin means
*the arguments that give the smallest value*, and s.t. means *such that*:
among all splits with $M\cdot D=C$, pick the one with the lowest error. (3)
states the goal; (4) is the fitted answer to it, not a closed-form solution.

**Kind of statement:** both (1) and (4) are **empirical fits**. They summarize
measurements; nothing in the paper derives them. What matters for a fit is its
range. (1) was fitted on budgets from 1e17 to 2e19 (§3.1) and (4) on 1e17
to 3e20 (§3.2); both are quoted under _Where the samples come from_. The 7B budget is about
$8.5\times10^{22}$ and the 67B budget about $8.6\times10^{23}$ FLOPs (with
$M$ from Figure 4(b); check 7), so both
models sit far outside the fitted ranges: the 67B budget is 43,000 times the top
of (1)'s range and 2,867 times the top of (4)'s (check 9). Using the curves
there is extrapolation. Note also that the grey circles in Figure 3(a) appear
to reach about 1e20 when read off the plot (figure image), past the 2e19 the
text gives; the
paper does not say whether its 1e20 validation runs were included in the fit.

## Where the samples come from

Neither equation contains an expectation or any sampling, so there is no
distribution, model copy or stop-gradient to track. What they summarize is a
set of training runs, each giving one measured error:

- **(1):** runs with "different batch sizes, learning rates, and compute
  budgets ranging from 1e17 to 2e19 by reusing the first stage" (§3.1). The
  first stage is the first 80% of the multi-step schedule (§2.3, "the
  aforementioned distribution of 80%, 10%, and 10% for the three stages
  respectively"), and §2.3 says the reuse works "When adjusting the training
  scale while keeping the model size fixed". So runs of one model size at
  different budgets share first-stage training: the points at different $C$ are not all
  independent runs. The number of runs, and how many share a stage, is not
  stated.
  Runs within 0.25% of the minimum error become the grey circles of Figure 3,
  and the curve is fitted to those.
- **(4):** "We selected 8 different compute budgets ranging from 1e17 to 3e20,
  and designed around 10 different model/data scale allocations for each
  budget. The hyperparameters for each budget were determined by Formula(1),
  and the generalization error was calculated on an independent validation
  set, distributed similarly to the training set and containing 100M tokens."
  (§3.2). This is the IsoFLOP study described at the top; (4) is a curve
  fitted through each budget's best size and token count.

The training and validation text is fixed in advance, not generated by the
model. Each setting is reported as a single run; no repeated runs or seeds are
mentioned (searched the whole paper for "seed" and "repeat").

## Check with small numbers

The inputs: $C=1$ isolates each coefficient, because every power of 1 is 1.
$C=2^8$ makes $C^{-0.125}$ exactly $1/2$. The budgets $10^{18}$ and $10^{19}$
isolate each exponent. The 7B and 67B budgets recompute numbers the paper
prints in Table 2 and Figures 3 and 4. All seven rows of Table 3 recompute
equation (2) and the gaps behind the "up to 50%" of §3.2. The equations have no branches. The only branch is the 0.25% rule, tested
with three made-up errors: the minimum, one exactly on the 0.25% boundary and
one just past it. Exact fractions keep the boundary case exact.

```python
from fractions import Fraction

import numpy as np

# Equation (1), §3.1
def eta_opt(C): return 0.3118 * C ** -0.1250
def B_opt(C):   return 0.2920 * C ** 0.3271

# Equation (4), §3.2
M_base, a = 0.1715, 0.5243
D_base, b = 5.8316, 0.4757
def M_opt(C): return M_base * C ** a
def D_opt(C): return D_base * C ** b

# Equation (2), §3.2: three ways to count "model scale" per token
def N1(n_layer, d):       return 12 * n_layer * d**2           # 6*N1 = 72 n d^2
def N2(n_layer, d, V):    return N1(n_layer, d) + V * d        # 6*N2 adds 6 V d
def M_eq2(n_layer, d, l): return 72 * n_layer * d**2 + 12 * n_layer * d * l

# 1. Coefficient term alone: at C = 1 every power of C is 1.
assert eta_opt(1.0) == 0.3118 and B_opt(1.0) == 0.2920
assert M_opt(1.0) == M_base and D_opt(1.0) == D_base

# 2. Exponent term alone: C = 2**8 makes C**-0.125 exactly 1/2.
assert eta_opt(2.0**8) == 0.3118 / 2
print("eta_opt(256) =", eta_opt(2.0**8))

# 3. Ten times the compute multiplies each output by 10**exponent.
for name, f, e in [("eta_opt", eta_opt, -0.1250), ("B_opt", B_opt, 0.3271),
                   ("M_opt", M_opt, a), ("D_opt", D_opt, b)]:
    r = f(1e19) / f(1e18)
    assert np.isclose(r, 10**e)
    print(f"10x compute -> {name} x {r:.3f}")
rM, rD = M_opt(1e19) / M_opt(1e18), D_opt(1e19) / D_opt(1e18)
print(f"M_opt factor x D_opt factor = {rM * rD:.4f}")   # C = M*D: shares of the 10x
assert np.isclose(rM * rD, 10)

# 4. Eq. (4) against C = M*D (my derivation): M_opt*D_opt = M_base*D_base*C**(a+b).
print("a + b =", round(a + b, 10), "  M_base * D_base =", round(M_base * D_base, 5))
assert np.isclose(a + b, 1.0) and abs(M_base * D_base - 1) < 1e-3
for C in (1e17, 3e20):
    print(f"C={C:.0e}: M_opt={M_opt(C):.3e} FLOPs/token, D_opt={D_opt(C):.3e} tokens,"
          f" M_opt*D_opt/C={M_opt(C) * D_opt(C) / C:.5f}")
for a_, b_ in [(0.73, 0.27), (0.49, 0.51), (0.450, 0.550), (0.524, 0.476), (0.578, 0.422)]:
    assert np.isclose(a_ + b_, 1.0)   # every Table 4 row sums to 1
assert (round(a, 3), round(b, 3)) == (0.524, 0.476)   # Table 4, "Ours (Current Data)"

# 5. A power law is a straight line on log-log axes (illustration only; see
#    "Not defined in the paper"). Fit noiseless points from eq. (1).
C = np.array([1e17, 1e18, 1e19])
slope, icpt = np.polyfit(np.log10(C), np.log10(eta_opt(C)), 1)
print(f"log-log fit: slope={slope:.4f}, 10**intercept={10**icpt:.4f}")
assert np.isclose(slope, -0.125) and np.isclose(10**icpt, 0.3118)

# 6. Table 3, all seven rows (n_vocab=102400, l_seq=4096), and the gaps
#    behind "differences reaching up to 50%" (§3.2), under three measures.
table3 = [(8, 512, 25.2e6, 77.6e6, 352e6, 0.43, 1.32),
          (12, 768, 84.9e6, 164e6, 963e6, 0.53, 1.02),
          (24, 1024, 302e6, 407e6, 3.02e9, 0.60, 0.81),
          (24, 2048, 1.21e9, 1.42e9, 9.66e9, 0.75, 0.88),
          (32, 4096, 6.44e9, 6.86e9, 45.1e9, 0.85, 0.91),
          (40, 5120, 12.6e9, 13.1e9, 85.6e9, 0.88, 0.92),
          (80, 8192, 64.4e9, 65.3e9, 419e9, 0.92, 0.94)]
gaps, off = [], []
for n, d, pN1, pN2, pM, pr1, pr2 in table3:
    n1, n2, m = N1(n, d), N2(n, d, 102400), M_eq2(n, d, 4096)
    r1, r2 = 6 * n1 / m, 6 * n2 / m
    print(f"Table 3 n={n}, d={d}: N1={n1:.4g} N2={n2:.4g} M={m:.4g}"
          f" 6N1/M={r1:.3f} 6N2/M={r2:.3f}")
    assert all(np.isclose(g, p, rtol=0.01) for g, p in [(n1, pN1), (n2, pN2), (m, pM)])
    off += [(n, d, name, round(r, 4), p) for name, r, p in [("6N1/M", r1, pr1), ("6N2/M", r2, pr2)]
            if round(r, 2) != p]   # printed ratio is not our ratio rounded to 2 places
    gaps.append((1 - r1, r2 - 1, 1 / r1 - 1))
print("ratios that do not round to the printed value:", off)
assert off == [(32, 4096, "6N1/M", 0.8571, 0.85)]
g1, g2, g3 = (max(abs(g[k]) for g in gaps) for k in range(3))
print(f"largest |1 - 6N1/M| = {g1:.3f}, |6N2/M - 1| = {g2:.3f}, M/6N1 - 1 = {g3:.3f}")
assert not any(abs(abs(x) - 0.5) < 0.005 for g in gaps for x in g)   # no row gives 50%

# 7. The released models (Table 2: 2.0T tokens, context 4096).
D = 2.0e12
M7 = M_eq2(30, 4096, 4096)
C7 = M7 * D
print(f"7B : M={M7:.4g}, C={C7:.3e}, eta_opt={eta_opt(C7):.3e} (Table 2: 4.2e-4),"
      f" B_opt={B_opt(C7):.3e} tokens = {B_opt(C7) / 4096:.0f} seqs (Table 2: 2304)")
assert round(eta_opt(C7), 5) == 4.2e-4 and round(B_opt(C7) / 1e6, 1) == 9.2  # Fig. 3 label 9.2M
print(f"     Table 2's 2304 vs curve: {2304 / (B_opt(C7) / 4096) - 1:+.1%}")

M67_eq2 = M_eq2(95, 8192, 4096)
M67_fig = 4.3e11                                  # Figure 4(b) label
# My count, not the paper's. GQA (grouped-query attention): the 67B's 64 query
# heads share 8 key/value heads (Table 2). Per layer, in units of d^2:
# Q,O = 2, K,V = 2*(8/64), FFN = 8 (eq. (2) assumes 2 + 2 + 8 = 12).
M67_gqa = 6 * (2 + 2 * 8 / 64 + 8) * 95 * 8192**2 + 12 * 95 * 8192 * 4096
print(f"67B: M from eq.(2) = {M67_eq2:.4g}; Figure 4(b) = {M67_fig:.4g};"
      f" eq.(2) with K,V shrunk to 8/64 (mine) = {M67_gqa:.4g}")
for label, M in [("eq.(2)", M67_eq2), ("Fig.4(b)", M67_fig)]:
    C67 = M * D
    print(f"     using {label}: C={C67:.3e}, eta_opt={eta_opt(C67):.3e} (Table 2: 3.2e-4),"
          f" B_opt={B_opt(C67):.3e} tokens = {B_opt(C67) / 4096:.0f} seqs (Table 2: 4608)")
print(f"     Table 2's 4608 vs curve (Fig.4(b) M): {4608 / (B_opt(M67_fig * D) / 4096) - 1:+.1%}")
assert round(B_opt(M67_fig * D) / 1e6, 1) == 19.7          # Fig. 3 label 19.7M
assert round(B_opt(M67_eq2 * D) / 1e6, 1) != 19.7

# 8. Eq. (4) run backwards: which budget makes M_opt = 4.3e11? (Fig. 4 labels)
C_star = (M67_fig / M_base) ** (1 / a)
print(f"M_opt = 4.3e11 at C={C_star:.3e} (Fig. 4: 4.5e23), D_opt={D_opt(C_star):.3e}"
      f" (Fig. 4: 1.04e12); trained D / D_opt = {D / D_opt(C_star):.2f}")
assert round(C_star / 1e23, 1) == 4.5 and round(D_opt(C_star) / 1e12, 2) == 1.04

# 9. How far the released models sit outside the fitted ranges.
print(f"7B C / 2e19 = {C7 / 2e19:.0f}; 67B C / 2e19 = {M67_fig * D / 2e19:.0f};"
      f" 7B C / 3e20 = {C7 / 3e20:.0f}; 67B C / 3e20 = {M67_fig * D / 3e20:.0f};"
      f" C_star / 3e20 = {C_star / 3e20:.0f}")

# 10. "Near-optimal" (§3.1): error exceeds the minimum by no more than 0.25%.
#     Exact fractions so the boundary case is tested exactly.
losses = [Fraction("4.000"), Fraction("4.010"), Fraction("4.011")]
lo = min(losses)
near = [x <= lo * (1 + Fraction(25, 10000)) for x in losses]
print("near-optimal:", [(float(x), k) for x, k in zip(losses, near)])
assert near == [True, True, False]
```

Run with `uv run --no-project --with numpy python check.py`. Output:

```text
eta_opt(256) = 0.1559
10x compute -> eta_opt x 0.750
10x compute -> B_opt x 2.124
10x compute -> M_opt x 3.344
10x compute -> D_opt x 2.990
M_opt factor x D_opt factor = 10.0000
a + b = 1.0   M_base * D_base = 1.00012
C=1e+17: M_opt=1.404e+08 FLOPs/token, D_opt=7.123e+08 tokens, M_opt*D_opt/C=1.00012
C=3e+20: M_opt=9.342e+09 FLOPs/token, D_opt=3.212e+10 tokens, M_opt*D_opt/C=1.00012
log-log fit: slope=-0.1250, 10**intercept=0.3118
Table 3 n=8, d=512: N1=2.517e+07 N2=7.759e+07 M=3.523e+08 6N1/M=0.429 6N2/M=1.321
Table 3 n=12, d=768: N1=8.493e+07 N2=1.636e+08 M=9.626e+08 6N1/M=0.529 6N2/M=1.020
Table 3 n=24, d=1024: N1=3.02e+08 N2=4.068e+08 M=3.02e+09 6N1/M=0.600 6N2/M=0.808
Table 3 n=24, d=2048: N1=1.208e+09 N2=1.418e+09 M=9.664e+09 6N1/M=0.750 6N2/M=0.880
Table 3 n=32, d=4096: N1=6.442e+09 N2=6.862e+09 M=4.51e+10 6N1/M=0.857 6N2/M=0.913
Table 3 n=40, d=5120: N1=1.258e+10 N2=1.311e+10 M=8.556e+10 6N1/M=0.882 6N2/M=0.919
Table 3 n=80, d=8192: N1=6.442e+10 N2=6.526e+10 M=4.188e+11 6N1/M=0.923 6N2/M=0.935
ratios that do not round to the printed value: [(32, 4096, '6N1/M', 0.8571, 0.85)]
largest |1 - 6N1/M| = 0.571, |6N2/M - 1| = 0.321, M/6N1 - 1 = 1.333
7B : M=4.228e+10, C=8.456e+22, eta_opt=4.246e-04 (Table 2: 4.2e-4), B_opt=9.223e+06 tokens = 2252 seqs (Table 2: 2304)
     Table 2's 2304 vs curve: +2.3%
67B: M from eq.(2) = 4.973e+11; Figure 4(b) = 4.3e+11; eq.(2) with K,V shrunk to 8/64 (mine) = 4.303e+11
     using eq.(2): C=9.946e+23, eta_opt=3.120e-04 (Table 2: 3.2e-4), B_opt=2.065e+07 tokens = 5043 seqs (Table 2: 4608)
     using Fig.4(b): C=8.600e+23, eta_opt=3.177e-04 (Table 2: 3.2e-4), B_opt=1.970e+07 tokens = 4808 seqs (Table 2: 4608)
     Table 2's 4608 vs curve (Fig.4(b) M): -4.2%
M_opt = 4.3e11 at C=4.457e+23 (Fig. 4: 4.5e23), D_opt=1.037e+12 (Fig. 4: 1.04e12); trained D / D_opt = 1.93
7B C / 2e19 = 4228; 67B C / 2e19 = 43000; 7B C / 3e20 = 282; 67B C / 3e20 = 2867; C_star / 3e20 = 1486
near-optimal: [(4.0, True), (4.01, True), (4.011, False)]
```

What the numbers show. For the 7B, equation (1) with $C=M\cdot D$
reproduces Table 2's learning rate (4.2e-4) and the "9.2M" label in Figure
3(a) (figure image). For the 67B,
Table 2's 3.2e-4 and Figure 3(a)'s "19.7M" (figure image) match only when $M$
is the 4.3e11 of Figure 4(b) (figure image), not the 4.97e11 that equation (2)
gives. Read as tokens, (1) gives 2252 sequences of 4096 tokens for the 7B and
4808 for the 67B (with $M=4.3\times10^{11}$). Table 2's 2304 and 4608
sequences are 2.3% above and 4.2% below these; the paper does not say how it
chose them. Run backwards, (4) reproduces the labels of Figure 4.
Extrapolated 1,486 times past its largest fitted budget (check 9), the fit
places the optimum for $M=4.3\times10^{11}$ at about 4.5e23 FLOPs and
1.04e12 tokens. The 67B was
trained on 2.0T tokens, 1.93 times that token count; the paper does not
comment on this. All seven rows of Table 3 match equation (2) to the
printed precision except one ratio: for $n_{\mathrm{layer}}=32$,
$d_{\mathrm{model}}=4096$, $6N_1/M$ is 0.8571, printed as 0.85 (check 6).

## What the code must decide

The paper links no code for these experiments, so every choice below is open
unless the paper states it.

- **Units and conversion.** The unit of $B_{\mathrm{opt}}$ and how to count
  $C$ (see _Not defined in the paper_), and how to round $B_{\mathrm{opt}}$
  to whole sequences per step (see _What the numbers show_).
- **Where $\eta_{\mathrm{opt}}$ goes in the schedule:** which point of §2.3's
  schedule receives it (see _Not defined in the paper_).
- **How to count $M$ for a real architecture.** (2) has no term for
  grouped-query attention (GQA), where several query heads share one
  key/value head; Table 2 gives the 67B 64 heads and 8 key/value heads (for
  the gap this leaves, see _Not defined in the paper_). Shrinking the key and
  value projections in (2) to 8/64 of their size reproduces 4.30e11 (check
  7); that count is mine, not the paper's.
- **Choosing a shape for $M_{\mathrm{opt}}$.** (4) returns a FLOPs-per-token
  number, not $n_{\mathrm{layer}}$ and $d_{\mathrm{model}}$; how to pick them
  is open.
- **Refitting on your own data.** The fitting loss, log or linear space,
  point weights, how each budget's minimum is located, and the group over
  which the 0.25% threshold is measured are all open (see *Not defined in the paper*).
- **Everything else is held fixed.** "Therefore, these parameters are
  consistent with those outlined in Section 2.3 and remain unchanged across
  different compute budgets." (§3). §2.3 gives: "DeepSeek LLM is initialized
  with a standard deviation of 0.006 and trained using the AdamW optimizer
  (Loshchilov and Hutter, 2017), with the following hyperparameters:
  β1 = 0.9, β2 = 0.95, and weight_decay = 0.1." and "The gradient clipping
  during the training phase is set to 1.0."
- **Validation set.** 100M tokens, "distributed similarly to the training
  set" (§3.2); how it was drawn is open.

## What the paper claims and what it shows

- **Equation (1) fits the near-optimal learning rate and batch size:**
  measured, on runs from 1e17 to 2e19 that share first-stage training (see
  _Where the samples come from_; Figure 3). The paper reports no
  goodness-of-fit number, and the grey band drawn in Figure 3 is not
  explained in the caption. The text says "all near-optimal hyperparameters
  fall within a broad band range, indicating that it is relatively easy to
  choose near-optimal parameters within this interval."
- **The fit holds at 1e20:** validated, by the paper's statement, on a
  series of 1e20 models; results are shown for one size, 2.94B FLOPs per
  token (Figure 2(b)): "We validated our formulae on
  a series of models with a 1e20 compute budget, and the results of a
  specific model size (2.94B FLOPs per token) are shown in Figure 2(b). The
  results indicate that the fitted parameters are centered in the optimal
  parameter space." (§3.1)
- **The fitted values worked for 7B and 67B:** "Subsequent sections also
  show that the parameters we fitted for DeepSeek LLM 7B and 67B models
  similarly achieved good performance." (§3.1). The paper shows no run of
  these models with other learning rates or batch sizes, so this is support by
  the final result, not a comparison.
- **The hyperparameters depend on $C$ only:** a modelling choice the paper
  itself qualifies: "However, it's important to note that we have not yet
  considered the impact of factors beyond the compute budget C on the optimal
  hyperparameters. This is inconsistent with some earlier works (Kaplan et
  al., 2020; McCandlish et al., 2018) which suggested that the optimal batch
  size can be modeled as being solely related to the generalization error L."
  It adds that "in models with the same compute budget but different
  model/data allocations, the optimal parameter space varies slightly." (§3.1)
- **Equation (4) gives the compute-optimal split:** measured, with the
  IsoFLOP runs from 1e17 to 3e20 (Figure 4). That $a+b=1$ and
  $M_{\mathrm{base}}D_{\mathrm{base}}=1$ follows from $C=MD$ (my derivation);
  the printed constants satisfy it (check 4).
- **$M$ is a better measure of model scale than $6N_1$ or $6N_2$:** the gap
  between the measures is definitional (Table 3, recomputed for all seven
  rows in check 6). The text adds "This discrepancy is particularly
  pronounced in small-scale models, with differences reaching up to 50%." It
  does not define the measure, and under three readings of it no row of
  Table 3 gives 50%: the largest gaps, all in the first row, are
  $1-6N_1/M=0.571$, $6N_2/M-1=0.321$ and $M/6N_1-1=1.333$ (check 6). That
  $M$ predicts better is measured in Appendix A.2, Figure 6, by refitting
  "reusing the experiments from the IsoFLOP profile" and comparing with the
  two released models, judged by eye: "When using 6N1 as the model scale
  representation, the
  fitted performance scaling curve tends to overestimate the performance of
  large-scale models. Conversely, when using 6N2, the curve tends to
  underestimate their performance. Using M as the model scale representation,
  however, achieves the most accurate predictions." No error numbers are
  given. Appendix A.2 adds that "the deviation of optimal model/data
  allocation among these three representations is not significant at higher
  compute budgets, but there are noticeable differences at lower budgets."
- **Small runs predict the large models' loss:** measured on two points
  (Figure 5): "The results indicate that using small-scale experiments can
  accurately predict the performance of models with 1000× compute budget."
  No numeric error is given. The paper does not say what the 1000× is
  measured against. Figure 5's grey circles run from 1e17 to 3e20 (figure
  image); against 3e20 the 7B and 67B budgets are 282 and 2,867 times larger
  (check 9; the 67B with $M$ from Figure 4(b)). Both lie outside the fitted
  range (see _Kind of statement_).
- **Better data shifts compute toward the model:** measured as $a$ and $b$
  on three datasets (Table 4). The quality ranking comes from an assessment
  the paper does not describe: "Our internal data assessment revealed that
  current in-house data has higher data quality than early in-house data."
  The explanation is a suspicion: "An intuitive speculation for this finding
  is that high-quality data usually implies logical clarity and less
  predictive difficulty after sufficient training." The link to earlier
  disagreements is hedged: "This finding might also explain the significant
  differences in optimal model/data scaling-up allocation observed in earlier
  studies of scaling laws."
- **Uncertainty:** none of these fits is reported with an interval, a
  standard deviation or repeated runs.

## Sources

- DeepSeek-AI, DeepSeek LLM: Scaling Open-Source Language Models with
  Longtermism, <https://arxiv.org/abs/2401.02954v1>: the PDF, its
  `pdftotext` text and the TeX source of v1 (`main.tex`, and the figure
  images `flops_bsz_fitting.png`, `flops_lr_fitting.png`,
  `loss_bs_lr_1e17.png`, `nosafe_flops_per_token_bpb.png`,
  `nosafe_flops_flops_per_token.png`, `nosafe_flops_tokens.png`,
  `flops_bpb.png`).
- Named but not opened: Kaplan et al. 2020 (Scaling Laws for Neural Language
  Models); Hoffmann et al. 2022 (Chinchilla); McCandlish et al. 2018 (An
  Empirical Model of Large-Batch Training).
