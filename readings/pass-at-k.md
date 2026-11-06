# Codex paper: pass@k, the chance that at least one of k generated programs passes the tests

**Source:** Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, et al.,
[Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374v2),
v2 (14 Jul 2021), §2.1, equation (1). Other locations used: Figure 3
(the paper's NumPy code); Appendix A and Figure 13 (the argument
for equation (1) over the alternatives); §2.2 and §3.2–§3.3 (the problems and how the
samples are generated); and the repository the paper links,
`openai/human-eval`, at commit
[`d321ec0b6c23dec317337be99f6d0c45ca73f3d5`](https://github.com/openai/human-eval/tree/d321ec0b6c23dec317337be99f6d0c45ca73f3d5)
(14 Jul 2021, the same day as v2; the repository has no tags). Its
`human_eval/evaluation.py`, `human_eval/execution.py` and `README.md` are
byte-identical to those of commit 463c980 (6 Jul 2021), the only commit that
ever touched `evaluation.py`.

## What it computes and what it changes

Equation (1) scores a code model, such as the paper's Codex, on a set of
programming problems. For each
problem it asks: if a user could try k of the model's programs, how likely is
it that at least one passes the problem's unit tests? It then averages that over
the problems. The earlier method, as §2.1 describes Kulal et al. (2019) (quoted
in the pass@k row), generates exactly k programs per problem and scores the problem 1 if any passes,
else 0; about its spread, the paper's text says only that this score "can have
high variance" (§2.1; I searched the text for "variance"). How much it varies is my note: if each program passes
with probability p, independently of the others, the score is 0 only when all k
fail, which has probability $(1-p)^k$. So it is 1 with probability
$q=1-(1-p)^k$ and 0 otherwise, and its standard deviation (sd), the typical
distance from its average, is $\sqrt{q(1-q)}$: 0.4766 at p = 0.1 and k = 10
(check item 8).
Equation (1) instead generates n ≥ k programs (n = 200 in §2.1; see the $n$
row for other sections), counts the c that pass, and computes exactly what
fraction of all possible choices of k programs out of those n contain a passing
one. So it can sample more programs than k (200 against k ≤ 100 in §2.1). When
n > k, it replaces one 0/1 outcome with an exact probability over all choices;
when n = k, it is Kulal's 0/1 score (step 4 of "Read from the inside out"). A formula that turns random samples into a
guess of an unknown number is an _estimator_; it is _unbiased_ when its average
over many reruns of the sampling equals that number. The paper calls equation
(1) an unbiased estimator.

## The equation

$$
\text{pass@}k := \mathop{\mathbb{E}}_{\text{Problems}}\left[1-\frac{\binom{n-c}{k}}{\binom{n}{k}}\right] \tag{1}
$$

## Symbols

| Symbol | Shape or type | Meaning | Role | Defined at |
| ------ | ------------- | ------- | ---- | ---------- |
| $\text{pass@}k$ | one number in $[0,1]$, reported as a percentage | The fraction of problems solved when k programs are generated per problem and a problem counts as solved if any of them passes. Equation (1) computes an estimate of it; the paper gives the metric and the estimate the same name. | computed intermediate (the final output) | "the pass@k metric, where k code samples are generated per problem, a problem is considered solved if any sample passes the unit tests, and the total fraction of problems solved is reported" (§2.1, describing Kulal et al.); equation (1) is introduced as "the unbiased estimator" (§2.1) |
| $\mathbb E_{\text{Problems}}$ | turns one number per problem into one number | Average over the problems of the benchmark: for HumanEval, 164 problems (§2.2). | input data (the problems, fixed before evaluation) | No formula in the paper; see the list below |
| $n$ | integer, one per problem | How many programs are generated for the problem: 200 as §2.1 states, but §3.5 samples 1000 per task of APPS, a second set of coding problems with unit tests, and Table 3 hand-grades 10 docstrings per task from Codex-D, the paper's model that writes a docstring for a given function (§5); there a sample is a docstring. | fixed setting | "we generate n ≥ k samples per task (in this paper, we use n = 200 and k ≤ 100)" (§2.1); departures: "We utilize this by sampling 1000 solutions from the model" (§3.5) and "evaluated by hand-grading 10 samples per task" (Table 3), where "we grade sample docstrings by hand" (§5); Figure 3: ":param n: total number of samples" |
| $c$ | integer, $0\le c\le n$, one per problem; across HumanEval an array of shape `[164]` | How many of the n programs pass the problem's unit tests; in Table 3, how many of the n docstrings are judged correct by hand. | computed intermediate: a count over random samples | "count the number of correct samples c ≤ n which pass unit tests" (§2.1); a sample is correct "if it passes a set of unit tests" (§2.1); for Table 3, "considering a docstring correct if it uniquely and accurately specifies the code body" (§5) |
| $k$ | integer, $1\le k\le n$ | How many programs the user may try per problem; the k in the name pass@k. §2.1 states $k\le100$, but Table 2 reports pass@1000 on APPS. | fixed setting (a parameter of the metric) | The pass@k row's quote, and "n ≥ k" (§2.1); Figure 3: ":param k: k in pass@$k$" (the dollar signs are printed in the PDF) |

**Not defined in the paper:** a formula for $\mathbb E_{\text{Problems}}$, and so
how problems are weighted. The problems are the benchmark's: HumanEval's 164
(§2.2; §3.2), or the APPS tasks in §3.5 (Table 2 reports each difficulty level
separately). §2.1 describes the metric as reporting "the total fraction of
problems solved", and §1 reports results such as "With just a single sample, a
12B parameter Codex solves 28.8% of these problems". Both suggest equal weight
per problem (my reading), but neither states it. I searched the whole paper
text, appendices and captions included, for "average", "mean", "weight" and
"fraction of problems". The linked code computes it as
`estimate_pass_at_k(total, correct, k).mean()` (`human_eval/evaluation.py`
L36, L76 and L79–89, in `evaluate_functional_correctness`, commit d321ec0), a plain mean with one entry
per problem, so each problem counts once whatever its n. Every other symbol is
defined.

## Read from the inside out

Everything inside the square brackets is computed for one problem, so $n$, $c$
and $k$ are plain integers there. Equation (1) has no superscripts; the $k$ under
$n$ in $\binom nk$ is the second argument of a function, and the "@" is part of
the name pass@k.

1. $n-c$: integer minus integer gives an integer, the number of programs that
   fail.
2. $\binom{n-c}{k}$ and $\binom nk$: $\binom ak=\frac{a!}{k!\,(a-k)!}$ counts the
   ways to choose k items out of a when order does not matter, and it is 0 when
   $k>a$. It is called the _binomial coefficient_; Python has it as
   `math.comb(a, k)`, which "Evaluates to n! / (k! * (n - k)!) when k <= n and
   evaluates to zero when k > n" (Python 3.13 documentation, `math.comb`). So
   $\binom nk$ counts every way to pick k of the n programs, and
   $\binom{n-c}{k}$ counts the picks made only of failing programs. Both are
   integers and can be huge: $\binom{200}{100}\approx9.055\times10^{58}$
   (computed in the check below).
3. The fraction $\binom{n-c}{k}/\binom nk$: the share of all k-picks that contain
   no passing program. Because a random draw makes every k-pick equally likely,
   this share is also the probability that k programs drawn at random from the
   n, without putting a drawn one back, all fail.
   Drawing without putting back is called _without replacement_, and Appendix A
   reads the term the same way: "the probability of drawing k failed samples
   without replacement". One number in $[0,1]$.
4. $1-(\ldots)$: the 1 is the number one, not an _indicator_ $\mathbf 1[C]$
   (1 when a condition C holds, else 0). The result is the
   probability that a random pick of k of the n programs contains at least one
   passing program: one number in $[0,1]$ per problem. Its edge cases, each
   checked below:
   - $c=0$ gives 0, because the fraction is then $\binom nk/\binom nk=1$;
   - fewer than k failing programs ($n-c<k$) gives 1, because every pick then
     contains a passing program; in the formula, $\binom{n-c}{k}=0$ by step 2's
     rule for $k>a$;
   - $k=1$ gives $1-\frac{n-c}{n}=\frac cn$, the fraction of programs that
     pass, because $\binom a1=a$;
   - $k=n$ gives 1 if any program passed and 0 otherwise, which is Kulal's 0/1
     score: $\binom nn=1$, and $\binom{n-c}{n}$ is 1 when $c=0$ and 0 when
     $c>0$.

   On the same programs the value never falls as k grows (my derivation:
   since $\binom{a}{k+1}=\binom ak\frac{a-k}{k+1}$, going from k to k + 1
   multiplies the all-fail share $\binom{n-c}{k}/\binom nk$ by
   $\frac{n-c-k}{n-k}\le1$; the value stays 0 when c = 0 and 1 once
   n − c < k): with n = 5 and c = 2 it is 0.4
   at k = 1 and 0.7 at k = 2 (check items 2 and 3). So two pass@k numbers are
   comparable only at the same k (my inference; the paper reports each k
   separately, as in Table 2's pass@1, @5, @100 and @1000).
5. $\mathbb E_{\text{Problems}}[\cdot]$: averages the per-problem numbers over the
   problems (weighting: see "Not defined in the paper"). On HumanEval this turns
   an array of shape `[164]` into one number.
6. $:=$ gives the name pass@k to that number.

Figure 3 computes step 3 without forming the huge integers. Cancelling the
factorials (my derivation, valid when $n-c\ge k$); the $k!$ cancels first:

$$
\frac{\binom{n-c}{k}}{\binom nk}=\frac{(n-c)!}{k!\,(n-c-k)!}\cdot\frac{k!\,(n-k)!}{n!}
=\frac{(n-k)!/(n-c-k)!}{n!/(n-c)!}.
$$

The denominator $n!/(n-c)!$ is the product of $i$ over the $c$ integers
$i=n-c+1,\dots,n$. The numerator $(n-k)!/(n-c-k)!$ is the product of the $c$
integers $n-c-k+1,\dots,n-k$, which are $i-k$ for those same $i$. Pairing the
factors with the same $i$:

$$
\frac{(n-k)!/(n-c-k)!}{n!/(n-c)!}
=\prod_{i=n-c+1}^{n}\frac{i-k}{i}=\prod_{i=n-c+1}^{n}\left(1-\frac ki\right).
$$

The product runs over $i$, which takes the $c$ values `np.arange(n - c + 1, n + 1)`:
one factor per passing program, because `np.arange` leaves out its end: "Values
are generated within the half-open interval `[start, stop)`" (NumPy v2.5.4,
`arange` docstring). With $c=0$ the product is empty and equals 1, as in NumPy:
"The product of an empty array is the neutral element 1" (NumPy v2.5.4, `prod`
docstring). The paper's version:

```python
def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k /
        np.arange(n - c + 1, n + 1))
```

**Kind of statement:** an estimator, named with $:=$. It counts the c passing
programs among the n per problem and averages over the problems. For one
problem, the quantity it estimates is $1-(1-p)^k$, where "p is pass@1"
(Appendix A): the probability that a single generated program passes, so
$(1-p)^k$, with k as a power, is the all-fail chance from the first section.
The paper claims the bracket is unbiased for this quantity, in the sense
defined in the first section (Appendix A; checked below). So $:=$ here names a
number computed from data; the claim about it concerns its average over reruns.

## Where the samples come from

Two averages are involved, and equation (1) prints only the first.

- **Over problems** ($\mathbb E_{\text{Problems}}$, printed in (1), computed as a
  mean): a fixed dataset made before evaluation, HumanEval's 164 hand-written
  problems, each with "an average of 7.7 tests per problem" (§2.2).
- **Over the model's programs** (not printed in (1); written $\mathbb E_c$ in
  Appendix A, never computed): for each problem the model under evaluation, a
  fixed trained model, generates the n programs. Nothing is trained during
  evaluation, so no gradient flows and nothing is detached. The prompt is "a
  header, a signature, and a docstring", and "We sample tokens from Codex until
  we encounter one of the following stop sequences: ‘`\nclass`’, ‘`\ndef`’,
  ‘`\n#`’, ‘`\nif`’, or ‘`\nprint`’" (§3.2), where `\n` is a line break.
  "We use nucleus sampling (Holtzman et al., 2020) with top p = 0.95 for all
  sampling evaluation in this work" (§3.2). The Codex paper defines
  neither top p nor temperature (I searched the whole text, appendices and
  captions included, for "temperature", "top p" and "nucleus"); it cites
  Holtzman et al. ([arXiv 1904.09751v2](https://arxiv.org/abs/1904.09751v2))
  for nucleus sampling, and that paper defines both. Its §3.1 defines the
  "top-p vocabulary $V^{(p)} \subset V$ as the smallest set such that
  $\sum_{x \in V^{(p)}} P(x|x_{1:i-1}) \ge p$" (equation (2)), and the next token
  is drawn from that set after its probabilities are rescaled (equation (3)). So
  top p = 0.95 draws each token only from the most likely tokens that together
  hold at least 95% of the probability. Its §3.3 divides the model's raw token
  scores (logits) by the temperature before the softmax: "Given the logits
  $u_{1:|V|}$ and temperature t, the softmax is re-estimated as"
  $\exp(u_l/t)/\sum_{l'}\exp(u_{l'}/t)$ (equation (4)). In Codex, higher
  temperature gives a set of samples with "higher diversity" (§3.3).
  For several HumanEval results the temperature depends on k: "When evaluating
  pass@k, it is important to optimize sampling temperature for the particular
  value of k" (§3.3), for example 0.2 for pass@1 and 0.8 for pass@100
  for a 679M model (§3.3, Figure 5); §4.5 does the same for Codex-S, Codex
  trained further on correctly implemented standalone functions (§4). My reading
  is that pass@1 and pass@100 then come from separate sets of programs. Some
  results use one temperature for every k, so there, in my reading, every k
  comes from the same programs: Table 2's Codex results on APPS ("We used temperature 0.6 for
  sampling to cover all k in pass@k, so raw pass@1 results could be improved
  with lower temperature", Table 2 caption), and Codex-D's hand-graded docstrings,
  sampled "at temperature 0.8" (§5). The randomness of the
  programs makes c random: "c, the number of correct samples that pass the
  unit tests, is distributed Binom(n, p), where p is pass@1" (Appendix A).
  Binom(n, p) is the distribution of the number of successes in n
  independent tries that each succeed with probability p. "Unbiased" is a
  statement about this second average.

## Check with small numbers

With n = 5 and k = 2, the inputs c = 0, c = 1 and c = 4 cover the empty product,
an ordinary case and the $n-c<k$ guard. Separate calls cover k = 1 and k = n.
Listing every pick of 2 programs checks the probability reading, and three
problems check the average. For unbiasedness, n = 4, k = 2 and p = 1/3 give exact
fractions. The paper's own n = 200 shows the size of the integers. Item 8
compares how much one problem's score spreads under Kulal's exactly-k method
and under equation (1) with n = 200.

```python
"""Check pass@k, equation (1) of arXiv 2107.03374v2, with small numbers."""

import itertools
import math
from fractions import Fraction

import numpy as np


# Figure 3 of the paper, copied as printed.
def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k /
        np.arange(n - c + 1, n + 1))


def direct(n, c, k):  # the bracket of equation (1), exact integers
    return 1 - Fraction(math.comb(n - c, k), math.comb(n, k))


# 1. Figure 3's product equals the bracket of (1) for every n <= 12, c <= n, 1 <= k <= n.
worst = max(
    abs(pass_at_k(n, c, k) - float(direct(n, c, k)))
    for n in range(1, 13)
    for c in range(n + 1)
    for k in range(1, n + 1)
)
assert worst < 1e-12
print("max |Figure 3 - equation (1)| over n<=12:", worst)

# 2. Branches, n = 5 samples, k = 2.
assert pass_at_k(5, 0, 2) == 0.0  # c = 0: empty product, nothing passes
assert direct(5, 1, 2) == Fraction(2, 5)  # 1 - C(4,2)/C(5,2) = 1 - 6/10
assert abs(pass_at_k(5, 1, 2) - 0.4) < 1e-12
assert pass_at_k(5, 4, 2) == 1.0  # n - c = 1 < k = 2: guard branch
assert abs(pass_at_k(5, 2, 1) - 2 / 5) < 1e-12  # k = 1 gives c / n
assert pass_at_k(5, 1, 5) == 1.0 and pass_at_k(5, 0, 5) == 0.0  # k = n: "any passes"
print("n=5,k=2: c=0 ->", pass_at_k(5, 0, 2), " c=1 ->", round(pass_at_k(5, 1, 2), 12),
      " c=4 ->", pass_at_k(5, 4, 2), " | k=1,c=2 ->", round(pass_at_k(5, 2, 1), 12))

# 3. Meaning of the bracket: draw k of the n samples without replacement;
#    the bracket is the fraction of draws that contain at least one correct sample.
n, c, k = 5, 2, 2
pool = [True] * c + [False] * (n - c)
draws = list(itertools.combinations(range(n), k))
hit = sum(any(pool[i] for i in d) for d in draws)
assert Fraction(hit, len(draws)) == direct(n, c, k) == Fraction(7, 10)
print(f"n=5,c=2,k=2: {hit} of {len(draws)} two-sample draws contain a correct sample")

# 4. E_Problems: an unweighted mean of one bracket per problem.
per_problem = [pass_at_k(5, c, 2) for c in (1, 0, 4)]
assert abs(np.mean(per_problem) - (0.4 + 0.0 + 1.0) / 3) < 1e-12
print("three problems, c = 1, 0, 4:", [round(float(x), 12) for x in per_problem],
      "-> pass@2 =", round(float(np.mean(per_problem)), 12))

# 5. Unbiased (Appendix A): with c ~ Binom(n, p), E_c[bracket] = 1 - (1 - p)^k.
#    The tempting 1 - (1 - c/n)^k comes out lower.
n, k, p = 4, 2, Fraction(1, 3)
pmf = {c: math.comb(n, c) * p**c * (1 - p) ** (n - c) for c in range(n + 1)}
e_eq1 = sum(w * direct(n, c, k) for c, w in pmf.items())
e_naive = sum(w * (1 - (1 - Fraction(c, n)) ** k) for c, w in pmf.items())
assert e_eq1 == 1 - (1 - p) ** k == Fraction(5, 9)
assert e_naive == Fraction(1, 2) < e_eq1
print("n=4,k=2,p=1/3: target", 1 - (1 - p) ** k, " E[eq (1)]", e_eq1, " E[1-(1-c/n)^k]", e_naive)

# 6. Why Figure 3 avoids the binomials: the sizes involved at the paper's n = 200.
print("C(200,100) =", f"{math.comb(200, 100):.3e}")
try:
    float(math.factorial(200))
except OverflowError as e:
    print("float(200!) ->", "OverflowError:", e)
#    The binomials themselves fit in float64 up to n = 1029, just past the paper's n = 1000 (§3.5).
fmax = float(np.finfo(np.float64).max)
last = max(m for m in range(1, 1100) if math.comb(m, m // 2) <= fmax)  # C(m, m//2) is row m's largest
assert last >= 1000
print(f"float64 max {fmax:.3e}; every C(m, k) fits in float64 for m <= {last}")
ratio = Fraction(math.comb(200 - 5, 100), math.comb(200, 100))
assert abs(pass_at_k(200, 5, 100) - float(1 - ratio)) < 1e-12
assert type(pass_at_k(200, 5, 100)) is np.float64  # Figure 3 computes in float64
print("n=200,c=5,k=100:", pass_at_k(200, 5, 100))

# 7. Guard caveat: called with k > n, Figure 3 returns 1.0 even with no correct sample.
assert pass_at_k(3, 0, 5) == 1.0
print("k > n, c = 0:", pass_at_k(3, 0, 5))

# 8. Spread of one problem's estimate, p = 0.1, k = 10 (my own illustration):
#    Kulal-style (generate exactly k samples, score 1 if any passes) versus (1) with n = 200.
def sd(n, k, p):
    pmf = [math.comb(n, c) * p**c * (1 - p) ** (n - c) for c in range(n + 1)]
    vals = [pass_at_k(n, c, k) for c in range(n + 1)]
    mean = sum(w * v for w, v in zip(pmf, vals))
    return mean, math.sqrt(sum(w * (v - mean) ** 2 for w, v in zip(pmf, vals)))

m10, s10 = sd(10, 10, 0.1)
m200, s200 = sd(200, 10, 0.1)
assert abs(m10 - (1 - 0.9**10)) < 1e-12 and abs(m200 - (1 - 0.9**10)) < 1e-12
assert abs(s10 - math.sqrt(m10 * (1 - m10))) < 1e-12  # a 0/1 score: sd = sqrt(q(1 - q))
assert s200 < s10
print(f"p=0.1,k=10: mean {m10:.4f}; sd with n=k=10: {s10:.4f}; sd with n=200: {s200:.4f}")
```

Run with `timeout 60 uv run --no-project --with "numpy>=2,<3" python -I check.py` (here Python 3.13.3
and NumPy 2.5.4). This reading quotes the Python 3.13 and NumPy 2.5.4
documentation because the check, Figure 3's code included, runs under these
versions, and neither Figure 3 nor the repository's `requirements.txt` (which
lists `numpy`, L3, commit d321ec0) pins a NumPy version. Output:

```text
max |Figure 3 - equation (1)| over n<=12: 1.6653345369377348e-16
n=5,k=2: c=0 -> 0.0  c=1 -> 0.4  c=4 -> 1.0  | k=1,c=2 -> 0.4
n=5,c=2,k=2: 7 of 10 two-sample draws contain a correct sample
three problems, c = 1, 0, 4: [0.4, 0.0, 1.0] -> pass@2 = 0.466666666667
n=4,k=2,p=1/3: target 5/9  E[eq (1)] 5/9  E[1-(1-c/n)^k] 1/2
C(200,100) = 9.055e+58
float(200!) -> OverflowError: int too large to convert to float
float64 max 1.798e+308; every C(m, k) fits in float64 for m <= 1029
n=200,c=5,k=100: 0.9703083947657067
k > n, c = 0: 1.0
p=0.1,k=10: mean 0.6513; sd with n=k=10: 0.4766; sd with n=200: 0.0831
```

Figure 3's product matches the binomial form in every case with n ≤ 12. With
5 programs of which 2 pass, 7 of the 10 possible pairs contain a passing one,
which is the bracket's 0.7. Averaged over c drawn from Binom(4, 1/3), the
bracket gives exactly $5/9=1-(2/3)^2$, while $1-(1-c/n)^k$ gives $1/2$.

Item 8 compares the spread of one problem's score, measured by the sd (see
the first section). Scoring exactly 10
programs has sd 0.477 around 0.651. Equation (1) with 200 programs has sd 0.083
around the same average.

## What the code must decide

- **Avoiding huge integers:** the product form, in float64 NumPy (Figure 3, and
  `estimator` inside `estimate_pass_at_k` in `human_eval/evaluation.py` L22–28,
  commit d321ec0, which uses the same expression; check item 6 asserts the result is
  a `np.float64`). My own note: Python's exact integers (`math.comb`,
  `Fraction`) compute the direct form exactly as well, as the check does:
  "Integers have unlimited precision" (Python 3.13 documentation, Built-in
  Types), and "The fractions module provides support for rational number
  arithmetic" (Python 3.13 documentation, `fractions`). Converting factorials
  to floats is what overflows: 200! is larger than float64's
  maximum of 1.798e+308 (check item 6). Every binomial with n ≤ 1029 fits in
  float64 (check item 6), so at the paper's n = 200 and n = 1000 the binomials
  do not overflow.
- **The $n-c<k$ case:** return 1.0 before the product (Figure 3;
  `human_eval/evaluation.py` L26–27, commit d321ec0).
- **k larger than n:** the same guard then returns 1.0 even when no program
  passed (check item 7), so the caller must refuse such k. In
  `evaluate_functional_correctness`, pass@k is reported only
  `for k in ks if (total >= k).all()` (`human_eval/evaluation.py` L89–90,
  commit d321ec0), and the `README.md` (L84–85, commit d321ec0) says: "Because there
  is no unbiased way of estimating pass@k when there are fewer samples than k,
  the script does not evaluate pass@k for these cases."
- **Weighting across problems:** a plain mean (see "Not defined in the paper").
- **The same n for every problem:** §2.1 states n = 200, and §3.5 and Table 3
  use other n (see the $n$ row). `estimate_pass_at_k` accepts one int or one n
  per problem (`num_samples: Union[int, List[int], np.ndarray]`,
  `human_eval/evaluation.py` L14 and L30–34, commit d321ec0).
- **Which k to report:** `k: List[int] = [1, 10, 100]` by default
  (`human_eval/evaluation.py` L41, commit d321ec0).
- **What "passes" means:** in `check_correctness` (`human_eval/execution.py`
  L38–42 and L44–63, commit d321ec0), a program passes when
  `problem["prompt"] + completion + "\n" + problem["test"] + "\n" + f"check({problem['entry_point']})"`
  runs without raising an exception within `timeout`. The default,
  `timeout: float = 3.0`, is set in `evaluate_functional_correctness`
  and passed to `check_correctness` (`human_eval/evaluation.py` L43 and L65–66,
  commit d321ec0), which hands it to `time_limit(seconds: float)`, and that
  calls `signal.setitimer(signal.ITIMER_REAL, seconds)`
  (`human_eval/execution.py` L13, L47 and L91–94, commit d321ec0). In Python 3.7,
  the minimum the README asks for ("Make sure to use python 3.7 or later",
  `README.md` L9, commit d321ec0), this sets a timer "to fire after seconds"
  (Python 3.7.0 documentation, `signal.setitimer`), so the default is 3
  seconds. A timeout counts as a failure
  (`human_eval/execution.py` L60–61, L79–80 and L84, commit d321ec0). The paper gives no time limit for HumanEval (I searched the whole
  text for "timeout", "time limit" and "seconds"); for APPS, §3.5 states "We use
  a timeout of 3 seconds in our evaluation", which matches the code default.
  In the released file (`human_eval/execution.py` L47–59, commit d321ec0) the
  `exec(check_program, exec_globals)` line is
  commented out behind a safety warning, which leaves
  `with time_limit(timeout):` with no body. Until a user uncomments it, the
  module cannot be imported: Python 3.7 reports a missing indented block as
  `IndentationError`, "expected an indented block" (CPython v3.7.0,
  `Python/pythonrun.c` L1339–1342), so nothing is evaluated.
- **Sampling settings** (temperature, top p, stop sequences): these change c, not
  the formula. The paper's choices are under "Where the samples come from"; the
  evaluation code takes the programs as given: it reads each sample's
  `completion` from the sample file (`human_eval/evaluation.py` L62–64, commit
  d321ec0).

## What the paper claims and what it shows

- **Equation (1) is unbiased:** derived in Appendix A. It assumes
  c ~ Binom(n, p), which requires the n programs for a problem to be
  independent, each passing with the same probability p. The check confirms the
  identity in one small case.
- **Kulal's k-program method is unbiased:** stated, "only the empirical
  estimate used by Kulal et al. (2019), and (1) are unbiased" (Appendix A), and
  derived only for (1). That derivation covers Kulal's score too (my reading):
  Kulal's score is (1) with n = k (step 4), and Appendix A's derivation needs
  only n ≥ k, since its sum runs over $i=0,\dots,n-k$; with n = k it has the
  single term $(1-p)^k$. Check item 8 confirms it at n = k = 10, p = 0.1: mean
  0.6513, which its assert equates with $1-0.9^{10}$.
- **Kulal's method "can have high variance"** (§2.1): stated in words, with no
  number. Figure 13's lower panel may show it, although the paper does not
  label it that way (Appendix A names Kulal's estimate only to call it
  unbiased; neither the sentence that cites Figure 13, nor the caption, nor
  the image's panel titles says that any curve shows Kulal's score): if its x-axis is n (see the Figure 13 bullet), each
  curve's left end, x = k, is the case n = k, Kulal's score, and in the image
  (`figs/estimators.png`) the shaded bands for k = 5, 10, 20 and 100 are
  widest at x = 5, 10, 20 and 100. Item 8 of the check is my illustration, not
  the paper's evidence.
- **"All estimators mentioned previously are consistent"** (Appendix A): stated
  without proof, and the paper never defines "consistent" (I searched the whole
  text for it; its other uses, such as "a consistent underestimate", are
  everyday ones). My reading: for (1) and $1-(1-\hat p)^k$, the estimate
  approaches the true value as n grows; for Kulal's score, which uses exactly k
  programs, the paper does not say what grows.
- **$1-(1-\hat p)^k$ is biased:** here $\hat p$ is "the empirical estimate of
  pass@1" (§2.1), that is $c/n$. The paper calls the result "a consistent
  underestimate as shown in Figure 13" and says "The gap doesn't fully close
  even when n > 5k" (Appendix A). This is measured in Figure 13's top panel;
  check item 5 shows one exact case (1/2 against 5/9). Why it comes out low:
  $(1-c/n)^k$ is the chance that k draws from the n programs, each put back
  before the next, all fail, and "The interpretation of this estimator is that
  we draw k samples with replacement from a pool of n candidates, but the k
  samples are not independent" (Appendix A), unlike (1)'s draws without
  replacement (step 3). With replacement, every draw fails with chance
  $(n-c)/n$. Without replacement, after j failed draws ($j=0,\dots,k-1$) the
  next fails with chance $(n-c-j)/(n-j)$, at most $(n-c)/n$ because the gap
  is $cj/(n(n-j))\ge0$ (my derivation). Multiplying the k chances,
  $(1-c/n)^k\ge\binom{n-c}{k}/\binom nk$ for every c, so the shortcut is never
  above the bracket of (1), whose average is the target. My addition: for k ≥ 2 the curve $1-(1-x)^k$ bends
  downward (it is _concave_), and $c/n$ averages to $p$, because Binom(n, p)
  has mean $np$; so the curve's average over the noisy $c/n$ is below its
  value at $p$ whenever 0 < p < 1 (Jensen's inequality).
- **Figure 13 leaves its setup out:** neither the caption nor Appendix A's text
  says which model, problems or temperature were used, how many reruns there
  were, or what the shaded bands and dashed lines show. The x-axis reads
  "Number of samples (k)" while the legend fixes k. The lower panel's curves start at x = k, so I read
  the axis as n. The caption's remark on variance is hedged: the unbiased
  estimator "may have a slightly higher variance initially".
- **pass@k as an oracle's pick (an interpretation):** "Pass@k can also be
  interpreted as the result of evaluating the best out of k samples, where the
  best sample is picked by an oracle with prior knowledge of the unit tests"
  (§3.3). So a deployed system reaches pass@k only if something, such as
  running the unit tests, picks out the passing program (my inference). For a
  system that "must select a single sample from k samples without having access
  to an oracle", the paper finds that "choosing the sample with the highest mean
  token log probability outperforms evaluating a random sample" (§3.3, measured
  in Figure 7).
- **Direct computation is unstable:** "Calculating this estimator directly
  results in very large numbers and numerical instability" (§2.1), stated.
  Check item 6 shows the sizes involved at n = 200.
- **Uncertainty of reported results:** the pass@k tables give single numbers
  with no ± or interval (`main.tex` contains no `\pm`; the archive's other TeX
  file, `icml-example.tex`, is the ICML template, which `main.tex` never
  inputs; and the PDF text has
  no "±", "confidence interval", "standard error", "standard deviation" or
  "error bar"). So a difference between two models' pass@k comes without a
  stated uncertainty.

## Sources

- Chen et al., [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374v2),
  arXiv 2107.03374v2: TeX source (`main.tex`), PDF, its extracted text, and the
  Figure 13 image `figs/estimators.png`.
- `openai/human-eval` at commit d321ec0b6c23dec317337be99f6d0c45ca73f3d5:
  [`human_eval/evaluation.py`](https://github.com/openai/human-eval/blob/d321ec0b6c23dec317337be99f6d0c45ca73f3d5/human_eval/evaluation.py),
  [`human_eval/execution.py`](https://github.com/openai/human-eval/blob/d321ec0b6c23dec317337be99f6d0c45ca73f3d5/human_eval/execution.py),
  [`README.md`](https://github.com/openai/human-eval/blob/d321ec0b6c23dec317337be99f6d0c45ca73f3d5/README.md),
  [`requirements.txt`](https://github.com/openai/human-eval/blob/d321ec0b6c23dec317337be99f6d0c45ca73f3d5/requirements.txt);
  the first three files at commit 463c980b59e818ace59f6f9803cd92c749ceae61
  (identical); and, from the GitHub API, the commit list for
  `human_eval/evaluation.py`, the repository's commits from June to August
  2021, and its (empty) tag list.
- Holtzman et al., [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751v2),
  arXiv 1904.09751v2 (the ICLR 2020 version): TeX source (`final.tex`), PDF and
  its extracted text, §3.1, equations (2)–(3), and §3.3, equation (4).
- Python 3.13 documentation:
  [`math.comb`](https://docs.python.org/3.13/library/math.html#math.comb),
  [Built-in Types, numeric types](https://docs.python.org/3.13/library/stdtypes.html#numeric-types-int-float-complex)
  and [`fractions`](https://docs.python.org/3.13/library/fractions.html).
- CPython at tag v3.7.0:
  [`Doc/library/signal.rst`](https://github.com/python/cpython/blob/v3.7.0/Doc/library/signal.rst)
  (`setitimer`) and
  [`Python/pythonrun.c`](https://github.com/python/cpython/blob/v3.7.0/Python/pythonrun.c).
- NumPy v2.5.4 docstrings: `arange` in
  [`numpy/_core/_add_newdocs.py`](https://github.com/numpy/numpy/blob/v2.5.4/numpy/_core/_add_newdocs.py)
  and `prod` in
  [`numpy/_core/fromnumeric.py`](https://github.com/numpy/numpy/blob/v2.5.4/numpy/_core/fromnumeric.py).
- Named, not opened: Kulal et al. (2019), "SPoC: Search-based Pseudocode to
  Code".
