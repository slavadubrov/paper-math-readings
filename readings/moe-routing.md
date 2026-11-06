# DeepSeek-V3: how each token picks 8 of 256 experts and how their outputs are weighted

**Source:** DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437v2),
v2 (18 Feb 2025), §2.1.2, equations (12)–(16). Other locations used: §2; §2.1,
Figure 2; §2.1.1, equation (11); the §2.1.2 paragraphs "Complementary
Sequence-Wise Auxiliary Loss" (equations (17)–(20)), "Node-Limited Routing" and
"No Token-Dropping"; §3.2; §3.3.3; §3.4.1; §3.4.2; §4.2; §4.5.2 and Table 5;
§4.5.3 and Figure 9; Appendix C. For comparison, DeepSeek-V2
([arXiv 2405.04434v5](https://arxiv.org/abs/2405.04434v5)), §2.2.1, equations
(20)–(22); DeepSeekMoE ([arXiv 2401.06066v1](https://arxiv.org/abs/2401.06066v1)),
§3.1, §3.2, §4.1.3, §4.4 and §4.5; Shazeer et al.
([arXiv 1701.06538v1](https://arxiv.org/abs/1701.06538v1)), §4; and the Qwen3
Technical Report ([arXiv 2505.09388v1](https://arxiv.org/abs/2505.09388v1)), §2.
Official code: `inference/model.py`, `inference/configs/config_671B.json`
(with `config_16B.json` and `config_236B.json` beside it), `inference/generate.py`
and `inference/requirements.txt` in the repository the paper links in its
abstract, at commit `1398800e` (14 Feb 2025); line numbers are of these files at
that commit.

## What it computes and what it changes

A Transformer block ends with a feed-forward network (FFN): a small MLP applied to
each token on its own. DeepSeek-V3 replaces that one MLP with many smaller MLPs,
called _experts_; the result is a _mixture-of-experts_ (MoE) layer. Not every
block has one: "We substitute all FFNs except for the first three layers with MoE
layers" (§4.2), so the first 3 of the 61 layers keep a dense FFN. One expert,
the _shared_ expert, runs on every token. For the other 256, the _routed_ experts, a scoring step (the _router_) scores each expert
against the token, the 8 best-scoring experts run (more only on a tie at the
cut-off; see "Ties at the cut-off"), and their outputs are added
with weights. Equation (12) is that weighted sum, (13)–(15) produce the weights,
and (16) replaces (14) by adding a per-expert number $b_i$ that changes which
experts are chosen. Each weight is still built from the unbiased score, but (13)
rescales over the chosen set, so a different set also changes the weights of
the experts that stay chosen (shown in the check below).

The earlier version is DeepSeek-V2 §2.2.1, equations (20)–(22). V3's (12), (14)
and (15) repeat V2's (20), (21) and (22) with two changes. V2 scores experts with
$s_{i,t}=\mathrm{Softmax}_i(\mathbf u_t^T\mathbf e_i)$ (V2 equation (22)), a
softmax across all routed experts, and uses the chosen scores directly as weights.
V3 scores each expert on its own with a sigmoid (15) and rescales the chosen
scores to sum to 1 (13): "Slightly different from DeepSeek-V2, DeepSeek-V3 uses
the sigmoid function to compute the affinity scores, and applies a normalization
among all selected affinity scores to produce the gating values" (§2.1.2, after
(15)). Equation (16) is new in V3. It serves _load balance_, keeping the experts
evenly used. "Conventional solutions usually rely on the auxiliary loss" (§2.1.2,
"Auxiliary-Loss-Free Load Balancing"), an extra
penalty term added to the training loss; V3 relies mainly on adjusting $b_i$
between training steps instead.

## The equation

$$
\mathbf{h}_{t}' = \mathbf{u}_{t} + \sum_{i=1}^{N_{s}} \operatorname{FFN}^{(s)}_{i}\left( \mathbf{u}_{t} \right) + \sum_{i=1}^{N_r} g_{i,t} \operatorname{FFN}^{(r)}_{i}\left( \mathbf{u}_{t} \right), \tag{12}
$$

$$
g_{i,t} = \frac{g'_{i,t}}{\sum_{j=1}^{N_r} g'_{j,t}}, \tag{13}
$$

$$
g'_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \operatorname{Topk} (\{ s_{j,t} | 1 \leqslant j \leqslant N_r \}, K_{r}), \\ 0, & \text{otherwise}, \end{cases} \tag{14}
$$

$$
s_{i,t} = \operatorname{Sigmoid} \left( {\mathbf{u}_{t}}^{T} \mathbf{e}_{i} \right), \tag{15}
$$

and, in the §2.1.2 paragraph "Auxiliary-Loss-Free Load Balancing", the version of
(14) that DeepSeek-V3 uses:

$$
g'_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} + b_i \in \operatorname{Topk} (\{ s_{j,t} + b_j | 1 \leqslant j \leqslant N_r \}, K_{r}), \\ 0, & \text{otherwise}. \end{cases} \tag{16}
$$

## Symbols

All quotes are from §2.1.2 unless another location is given: the sentence before
(12) and the "where" clause after (15), and, for $b_i$ and $\gamma$, the paragraph
"Auxiliary-Loss-Free Load Balancing".

| Symbol | Shape or type | Meaning | Role | Defined at |
| ------ | ------------- | ------- | ---- | ---------- |
| $\mathbf h'_t$ | vector in $\mathbb R^d$ (same shape as $\mathbf u_t$, because (12) adds them) | the layer's output for token $t$; the prime is a label, not a derivative | computed intermediate | "we compute the FFN output $\mathbf h'_t$ as follows" |
| $\mathbf u_t$ | column vector in $\mathbb R^d$, $d=7168$ | the input to this layer for token $t$ | function input; in the code it is `self.ffn_norm(x)` (`model.py` lines 730–731): the running hidden state `x`, after the attention output has been added to it, passed through RMSNorm (see "What the code must decide") | "Let $\mathbf u_t$ denote the FFN input of the $t$-th token". Shape: (11) gives $\mathbf u_t=W^O[\dots]$ with "$W^O\in\mathbb R^{d\times d_hn_h}$ denotes the output projection matrix" (§2.1.1); "the hidden dimension to 7168" (§4.2). (11) uses the same letter for the attention output, a different vector in the code (`model.py` line 730); both have width $d$, so the shape carries over |
| $t$ | integer | which token | index | "the $t$-th token" |
| $i$, $j$ | integers | which expert. In (12) the first sum's $i$ runs over the $N_s$ shared experts and the second sum's $i$ over the $N_r$ routed experts: two separate loops. $j$ is the loop variable inside (13), (14) and (16) | index | the limits printed on each $\sum$ and in each set |
| $N_s$ | integer; 1 in V3 | number of shared experts | fixed setting | "where $N_s$ and $N_r$ denote the numbers of shared experts and routed experts, respectively"; "Each MoE layer consists of 1 shared expert and 256 routed experts" (§4.2) |
| $N_r$ | integer; 256 | number of routed experts | fixed setting | same two quotes |
| $\mathrm{FFN}^{(s)}_i(\cdot)$ | function from $\mathbb R^d$ to $\mathbb R^d$ (output shape inferred from (12)), hidden width 2048 | shared expert $i$, run on every token with weight 1. The superscript $(s)$ is a label | trained weight | "$\mathrm{FFN}^{(s)}_i(\cdot)$ and $\mathrm{FFN}^{(r)}_i(\cdot)$ denote the $i$-th shared expert and the $i$-th routed expert, respectively"; "the intermediate hidden dimension of each expert is 2048" (§4.2) |
| $\mathrm{FFN}^{(r)}_i(\cdot)$ | same as above | routed expert $i$, run only for tokens that choose it. $(r)$ is a label | trained weight | same two quotes |
| $g_{i,t}$ | number $\ge0$; for each $t$, $K_r$ of the $N_r$ values are nonzero (more only on a tie at the cut-off; see "Ties at the cut-off"), and they sum to 1 (my reading of (13)) | weight on routed expert $i$'s output for token $t$ | computed intermediate | "$g_{i,t}$ is the gating value for the $i$-th expert" |
| $g'_{i,t}$ | number $\ge0$ | the weight before rescaling: the expert's score if it was chosen, else 0. The prime is a label | computed intermediate | (14) and (16) themselves; the prose calls the step after it "a normalization among all selected affinity scores" |
| $s_{i,t}$ | number in $(0,1)$ (the range of a sigmoid) | how well token $t$ matches expert $i$ | computed intermediate | "$s_{i,t}$ is the token-to-expert affinity" |
| $\mathbf e_i$ | vector of length $d$ (inferred; see "Not defined in the paper"). In the code, row $i$ of `Gate.weight` (`model.py` line 560), shape `[256, 7168]` (`n_routed_experts` and `dim`, `config_671B.json` lines 9 and 3) | one vector per routed expert; the router's raw score is its dot product with the token | trained weight (`nn.Parameter` in the code; the paper does not say how it is trained) | "$\mathbf e_i$ is the centroid vector of the $i$-th routed expert" |
| $K_r$ | integer; 8 | how many routed experts each token uses (more only on a tie at the cut-off; see "Ties at the cut-off") | fixed setting | "$K_r$ denotes the number of activated routed experts"; "Among the routed experts, 8 experts will be activated for each token" (§4.2) |
| $\mathrm{Topk}(\cdot,K)$ | takes a set of numbers and $K$; returns a set of $K$ numbers | the $K$ largest values. It returns score values, not expert numbers | operator defined by the paper | "$\mathrm{Topk}(\cdot, K)$ denotes the set comprising $K$ highest scores among the affinity scores calculated for the $t$-th token and all routed experts" |
| $\{s_{j,t}\mid1\leqslant j\leqslant N_r\}$ | set of $N_r$ numbers | all routed experts' scores for token $t$. The bar means "such that": the set of $s_{j,t}$ such that $1\le j\le N_r$. In Python: `{s[j] for j in range(N_r)}` | computed intermediate | written inside (14) |
| $b_i$ | one number per routed expert, the same for every token | an offset added to expert $i$'s score only when choosing experts | state updated by a rule rather than by gradients: read but not changed during a training step, then moved by $\pm\gamma$ at its end (the rule is quoted under "Kind of statement"); no gradient reaches it (my derivation, under "Where the samples come from"). In the code it is a stored `nn.Parameter`, `Gate.bias` (`model.py` line 561) | "we introduce a bias term $b_i$ for each expert and add it to the corresponding affinity scores $s_{i,t}$ to determine the top-K routing"; "Note that the bias term is only used for routing." |
| $\gamma$ (in the sentence after (16)) | number; 0.001, then 0.0 | step size of the change to $b_i$ | hyperparameter | "where $\gamma$ is a hyper-parameter called bias update speed"; "we set the bias update speed $\gamma$ to 0.001 for the first 14.3T tokens, and to 0.0 for the remaining 500B tokens" (§4.2) |

**Not defined in the paper:**

- **The formula inside an expert.** The paper names the expert's operator only
  in passing, "the SwiGLU operator in MoE" (§3.3.3, "Low-Precision Activation"),
  and prints no formula; §2.1.2, Figure 2 and §4.2 give only the hidden width
  (2048). §2 says "For other minor details not explicitly mentioned, DeepSeek-V3
  adheres to the settings of DeepSeek-V2"; I did not look for the expert's form
  in DeepSeek-V2. The code computes `self.w2(F.silu(self.w1(x)) * self.w3(x))`
  (`model.py` lines 630 and 529, in `Expert.forward` and `MLP.forward`).
- **The length of $\mathbf e_i$.** It is not printed: $\mathbf e_i$ appears only
  in (15) and in its definition after (15) (searched the TeX source and the PDF
  text). The product $\mathbf u_t^T\mathbf e_i$ requires length $d$.
- **"Overloaded", "underloaded" and "expert load", which drive the change to
  $b_i$.** The paper does not say what load counts as overloaded, and gives no
  initial value for $b_i$ (searched the whole PDF text, appendices included, for
  "overload", "underload" and "bias"). Its one initialization sentence, "All
  learnable parameters are randomly initialized with a standard deviation of
  0.006" (§4.2), does not say whether it covers $b_i$. The paper cites Wang et
  al. (2024a) for the method; I did not open it.
- **The superscript $T$ in $\mathbf u_t^T\mathbf e_i$ (15).** The paper does not
  say what it means. Words starting "transpos" occur only in hardware and
  quantization passages (§3.5.2, Appendix B.2; searched the PDF
  text), and §2.1.2 itself uses the letter for a count: "$T$ denotes the number
  of tokens in a sequence" (after (20)). I read it as a transpose: $\mathbf u_t$
  is a column of length $d$ (its row in the table), so $\mathbf u_t^T$ is a
  $1\times d$ row, and the product is one number, as $s_{i,t}$ is.

## Read from the inside out

1. **$\mathbf u_t^T\mathbf e_i$ (15).** With $T$ read as a transpose (see "Not
   defined in the paper"), $\mathbf u_t^T\mathbf e_i$ is one number,
   $\sum_{k=1}^{d}u_{t,k}e_{i,k}$. The sum runs over the $d$ coordinates. It is
   computed for every routed expert $i=1,\dots,N_r$, so each token gets $N_r$
   numbers; the script computes them as `E @ u`. The official code computes them
   for all tokens at once with `linear(x, self.weight)` (`model.py` line 573),
   whose docstring reads "Applies a linear transformation to the incoming data:
   y = xA^T + b." (line 131), with tokens as the rows of `x` (made by
   `x.view(-1, self.dim)`, line 677; an assumption about element order, stated
   under "Node-limited routing"); so row $i$ of `self.weight` is $\mathbf e_i$ and
   the result has shape `[n, N_r]`.
2. **Sigmoid (15).** $\mathrm{Sigmoid}(x)=1/(1+e^{-x})$, applied to each number on
   its own, gives a value in $(0,1)$. Expert $i$'s score depends only on its own
   dot product. V2's softmax divides by a sum over all experts, so raising one
   expert's score lowers all the others. Shape stays `[n, N_r]`.
3. **The selection test (16).** For one token $t$, add $b_j$ to each of the $N_r$
   scores, keep the $K_r=8$ largest of these sums, and ask whether expert $i$'s
   sum is one of them. The test runs over the experts $j$ for one token at a time,
   so each token chooses its own experts. The $\in$ here tests membership in a
   set; it is not a shape statement. The script does this with
   `np.argsort(-(s + b), kind="stable")[:K]`, which takes exactly $K_r$ and so
   matches the test only when no two sums tie at the cut-off, as in its inputs;
   the official code's version is under "Ties at the cut-off". Equation (14) is the same test with every $b_j=0$.
4. **$g'_{i,t}$ (16).** This is a two-case expression: if expert $i$ was chosen,
   $g'_{i,t}=s_{i,t}$, the score _without_ the bias. Otherwise it is 0. Each
   token gets $N_r$ values, $K_r$ of them nonzero (more only on a tie at the
   cut-off; see "Ties at the cut-off"). The official code keeps only the chosen
   ones, `original_scores.gather(1, indices)` (`model.py` line 591): in each
   token's row, the unbiased scores at the chosen indices, shape `[n, K_r]`
   ([`torch.gather`](https://docs.pytorch.org/docs/2.4/generated/torch.gather.html):
   "out will have the same shape as index").
5. **$g_{i,t}$ (13).** Divide each $g'_{i,t}$ by the sum of $g'_{j,t}$ over all
   $N_r$ experts $j$ for the same token. The unchosen experts add 0, so this is
   the sum over the chosen ones, and the chosen weights then sum to 1.
   In the script: `g_prime / g_prime.sum()`.
6. **The output (12).** Three vectors in $\mathbb R^d$ are added: the input
   $\mathbf u_t$ itself (a _residual connection_: the layer outputs its input plus
   a correction), the outputs of all $N_s$ shared experts, and the weighted sum of
   routed-expert outputs. That last sum is written over all $N_r$ experts, but
   only $K_r$ of its terms have a nonzero weight (more only on a tie at the
   cut-off), so only those experts need to run. That is the computational
   saving: per token, 8 of the 256 routed experts compute, apart from ties. The
   result $\mathbf h'_t$ is in $\mathbb R^d$.

**Kind of statement:** a computation, the forward pass of one layer for one token.
Lines (13)–(16) each define a named intermediate that can be substituted back
into (12); (16) is a replacement for (14), not an extra step. The shapes per
token are: $\mathbf u_t$ $[d]$ → scores $[N_r]$ → chosen experts $[K_r]$ →
weights $[N_r]$ with $K_r$ nonzero → $\mathbf h'_t$ $[d]$; on a tie at the
cut-off, more than $K_r$ experts are chosen and have nonzero weights (see "Ties
at the cut-off"). Every "$=$" in
(12)–(16) gives the left side its value. The $b_i$ are changed by an update rule
the paper writes only in prose: "At the end of each step, we will decrease the
bias term by $\gamma$ if its corresponding expert is overloaded, and increase it
by $\gamma$ if its corresponding expert is underloaded" (§2.1.2, after (16)).

## Where the samples come from

Equations (12)–(16) contain no expectation, average or random draw. Each token is
routed on its own and the top-$K$ choice is deterministic. The only quantity
computed over many tokens is the expert load that drives the change to $b_i$:
"During training, we keep monitoring the expert load on the whole batch of each
training step" (§2.1.2, after (16)). That batch is training data, and the load comes from the
routing decisions the current model made on it. $b_i$ is held fixed during the
step and changed only at its end; once $\gamma$ drops to 0.0 (the schedule is
in the table), it stays fixed.

Two consequences for gradients, which are my derivation from (13)–(20), not
statements in the paper. First, $b_i$ appears only inside the top-$K$ test, whose
outcome does not change under small changes of $b_i$, so backpropagation gives
$b_i$ a gradient of zero. This is why it needs a separate update rule. Second,
through (12)–(16), gradient reaches $\mathbf e_i$ only if expert $i$ was chosen,
through $s_{i,t}$ in $g_{i,t}$. But V3's training loss also contains a small
sequence-wise balance loss, (17)–(20), added "to prevent extreme imbalance within
any single sequence" (§2.1.2), with $\alpha=0.0001$ (§4.2). Its $P_i$ (20)
averages $s'_{i,t}=s_{i,t}/\sum_{j=1}^{N_r}s_{j,t}$ (19), which uses every
routed expert's score, so this loss can send a small gradient to every
$\mathbf e_j$. Its other factor, $f_i$ (18), is a count from a top-$K$ test
without $b$, so it gives $b_i$ no gradient either. The paper does not say
whether the path through $s'$ is detached ("detach" and "stop-gradient" do not
occur in it), and the released code cannot show it (see the start of "What the
code must decide").

## Check with small numbers

I chose $d=2$, $N_r=4$, $K_r=2$ and $N_s=2$ (V3 uses $N_s=1$; two shared experts
exercise that sum). $\mathbf u=(1,2)$, and the $\mathbf e_i$ give dot products
$(0,\log3,-\log3,\log7)$, so the sigmoids are exactly $1/2,3/4,1/4,7/8$. The
experts are linear maps, $\mathrm{FFN}^{(r)}_i(\mathbf u)=i\,\mathbf u$ and
$\mathrm{FFN}^{(s)}_k(\mathbf u)=10k\,\mathbf u$, so $\mathbf h'$ can be checked by
hand. Case A ($b=0$, equation (14)) chooses experts 2 and 4. In case B, a negative
bias on expert 2 and a positive bias on expert 3 swap expert 3 in for expert 2.
Across the two cases the test returns 0 and 1, both bias signs occur, and the
bias update moves in both directions. The script also re-types the official
`Gate.forward` in NumPy and compares it with the equations.

```python
"""Small-number check of DeepSeek-V3 (arXiv 2412.19437v2) equations (12)-(16)."""

from math import comb

import numpy as np

sigmoid = lambda z: 1 / (1 + np.exp(-z))
l3, l7 = np.log(3), np.log(7)


def moe_layer(u, E, b, K, shared, routed):
    """Equations (15), (16), (13), (12) for one token. u: [d], E: [N_r, d] (row i is e_i), b: [N_r]."""
    s = sigmoid(E @ u)                                  # (15) s_i = Sigmoid(u^T e_i), shape [N_r]
    chosen = np.argsort(-(s + b), kind="stable")[:K]    # (16) Topk over s_j + b_j -> K expert indices (no ties in these inputs)
    g_prime = np.zeros_like(s)
    g_prime[chosen] = s[chosen]                         # (16) keeps s_i, not s_i + b_i
    g = g_prime / g_prime.sum()                         # (13) sum runs over all N_r; unchosen add 0
    h = u + sum(f(u) for f in shared) + sum(g[i] * routed[i](u) for i in range(len(routed)))  # (12)
    return h, s, sorted(chosen.tolist()), g_prime, g


# d = 2, N_r = 4 routed experts, K_r = 2, N_s = 2 shared experts (V3 itself uses N_s = 1).
u = np.array([1.0, 2.0])
z = np.array([0.0, l3, -l3, l7])                 # wanted dot products u^T e_i
E = np.outer(z, [1 / 3, 1 / 3])                   # e_i = z_i * (1/3, 1/3), so u^T e_i = z_i * (1/3 + 2/3) = z_i
shared = [lambda u: 10 * u, lambda u: 20 * u]     # FFN^(s)_1, FFN^(s)_2
routed = [lambda u, k=k: k * u for k in (1, 2, 3, 4)]  # FFN^(r)_i(u) = i * u (1-based i)
K = 2

# Case A: b = 0, which is equation (14).
hA, s, chosenA, gpA, gA = moe_layer(u, E, np.zeros(4), K, shared, routed)
assert np.allclose(E @ u, z)
assert np.allclose(s, [1 / 2, 3 / 4, 1 / 4, 7 / 8])
assert chosenA == [1, 3]                          # 0-based: experts 2 and 4 (scores 3/4 and 7/8)
assert np.allclose(gpA, [0, 3 / 4, 0, 7 / 8])
assert np.allclose(gA, [0, 6 / 13, 0, 7 / 13]) and np.isclose(gA.sum(), 1)
assert np.allclose(hA, u * (1 + 10 + 20 + 2 * 6 / 13 + 4 * 7 / 13))
print("s =", s.round(4), "| sum of s =", s.sum())
print("A (b=0):     chosen", [i + 1 for i in chosenA], "g' =", gpA.round(4), "g =", gA.round(4), "h' =", hA.round(4))

# Case B: a bias that pushes expert 2 out and expert 3 in.
b = np.array([0.0, -0.5, 0.5, 0.0])
hB, _, chosenB, gpB, gB = moe_layer(u, E, b, K, shared, routed)
assert np.allclose(s + b, [0.5, 0.25, 0.75, 0.875])
assert chosenB == [2, 3]                          # experts 3 and 4: expert 3 (s = 1/4) beats expert 2 (s = 3/4)
assert np.allclose(gpB, [0, 0, 1 / 4, 7 / 8])     # weights use s, not s + b (that would give 0.75, 0.875)
assert np.allclose(gB, [0, 0, 2 / 9, 7 / 9])
assert np.allclose(hB, u * (1 + 10 + 20 + 3 * 2 / 9 + 4 * 7 / 9))
print("B (b=", b, "): chosen", [i + 1 for i in chosenB], "g' =", gpB.round(4), "g =", gB.round(4), "h' =", hB.round(4))

# Sigmoid scores are independent per expert; DeepSeek-V2's softmax scores are not, and sum to 1.
z2 = z.copy(); z2[0] = 5.0
assert np.isclose(sigmoid(E @ u)[1], sigmoid(z2)[1])
softmax = lambda v: np.exp(v - v.max()) / np.exp(v - v.max()).sum()
p = softmax(z)
assert np.allclose(p, np.array([1, 3, 1 / 3, 7]) / (34 / 3)) and not np.isclose(softmax(z2)[1], p[1])
v2_gate_sum = np.sort(p)[-K:].sum()               # V2 eq (20)-(22): top-K softmax values, no renormalization
assert np.isclose(v2_gate_sum, 15 / 17)
print("V2-style softmax s =", p.round(4), "| its top-2 gates sum to", round(v2_gate_sum, 4), "(15/17); V3 gates sum to 1")


# Official inference code re-typed in NumPy: inference/model.py lines 563-595 (Gate.forward) at commit 1398800e,
# with "score_func": "sigmoid" (inference/configs/config_671B.json line 15).
def gate_code(x, weight, bias, topk, n_groups, topk_groups, route_scale):
    scores = sigmoid(x @ weight.T)
    original_scores = scores
    scores = scores + bias
    if n_groups > 1:
        scores = scores.reshape(n_groups, -1)
        group_scores = np.sort(scores, axis=-1)[:, -2:].sum(-1)          # sum of top-2 per group
        keep = np.argsort(-group_scores, kind="stable")[:topk_groups]
        mask = np.ones(n_groups, bool); mask[keep] = False
        scores = np.where(mask[:, None], -np.inf, scores).ravel()
    indices = np.sort(np.argsort(-scores, kind="stable")[:topk])   # sorted only to compare with g
    weights = original_scores[indices]
    weights = weights / weights.sum()
    return weights * route_scale, sorted(indices.tolist())

for bias, g_eq in [(np.zeros(4), gA), (b, gB)]:
    w, idx = gate_code(u, E, bias, K, 1, 1, 2.5)
    assert np.allclose(w * 1 / 2.5, g_eq[idx]) and np.isclose(w.sum(), 2.5)
order = np.arange(4).reshape(2, -1)   # NumPy groups consecutive experts; torch view/flatten assumed to do the same
assert order.tolist() == [[0, 1], [2, 3]] and order.ravel().tolist() == [0, 1, 2, 3]
w, idx = gate_code(u, E, np.zeros(4), K, n_groups=2, topk_groups=1, route_scale=1.0)
assert idx == [0, 1]   # groups {1,2},{3,4}: top-2 sums 1.25 vs 1.125, so group 1 wins and expert 4 (s = 7/8) is excluded
print("code path, 2 groups keep 1, b=0: chosen", [i + 1 for i in idx], "weights", w.round(4))


# Bias update after one step (paper: -gamma if overloaded, +gamma if underloaded).
# "Overloaded" is not defined in the paper; here it means load above the mean load K*T/N_r (my stand-in).
T, gamma = 6, 0.001
loads = np.bincount(np.repeat(chosenA, T), minlength=4)   # T identical tokens, all routed as in case A
mean = K * T / 4
new_b = np.zeros(4) - gamma * np.sign(loads - mean)
assert loads.tolist() == [0, 6, 0, 6] and mean == 3
assert np.allclose(new_b, [gamma, -gamma, gamma, -gamma])
print("loads", loads.tolist(), "mean", mean, "-> b after one step", new_b)

# Numbers printed in the paper (section 4.2, hyper-parameters).
assert 8 // 4 == 2 and 256 // 8 == 32 and 256 // 64 == 4      # K_r/M; experts per node; experts per GPU
assert np.isclose(14.3 + 0.5, 14.8)                         # gamma schedule covers all 14.8T tokens
# Recount from inference/model.py at 1398800e: Transformer.__init__ lines 760-765 (embed, layers, norm, head);
# Block.__init__ 712-715 (MLA; dense MLP if layer_id < n_dense_layers, else MoE; two RMSNorms); MLA.__init__ 424-430;
# MLP.__init__ 515-517 and Expert.__init__ 616-618 (w1, w2, w3); MoE.__init__ 662-664; Gate.__init__ 560-561;
# RMSNorm.__init__ 279; ParallelEmbedding.__init__ 103; Linear.__init__ 176-190 (weight [out, in], no bias; FP8 scales not counted).
# Sizes from inference/configs/config_671B.json lines 2-20.
V, d, L, dense = 129280, 7168, 61, 3   # vocab_size, dim, n_layers, n_dense_layers (config lines 2, 3, 6, 7)
# q_lora_rank 1536, n_heads 128, qk_nope_head_dim 128, qk_rope_head_dim 64, kv_lora_rank 512, v_head_dim 128
mla = d * 1536 + 1536 + 1536 * 128 * (128 + 64) + d * (512 + 64) + 512 + 512 * 128 * (128 + 128) + 128 * 128 * d
expert = 3 * d * 2048                                       # w1, w2, w3 of one expert, moe_inter_dim 2048
gate = 256 * d + 256                                        # n_routed_experts 256; 257 and 9 below add the 1 shared expert
per_layer = mla + 2 * d
# 18432 = inter_dim (config line 4), the hidden width of the dense FFN in the first 3 layers
total = 2 * V * d + d + L * per_layer + dense * 3 * d * 18432 + (L - dense) * (257 * expert + gate)
active = 2 * V * d + d + L * per_layer + dense * 3 * d * 18432 + (L - dense) * (9 * expert + gate)
print(f"one expert {expert/1e6:.2f}M | total {total/1e9:.1f}B | activated {active/1e9:.1f}B, "
      f"{(active - V*d)/1e9:.1f}B without the input embedding table | routed share {(L-dense)*256*expert/total:.3f}")
assert round(total / 1e9) == 671 and int(active / 1e9) == 37 and round((active - V * d) / 1e9) == 37

# Table 5 (section 4.5.2): aux-loss-based vs aux-loss-free; Pile-test BPB is lower-is-better, the rest higher-is-better.
small = [(0.727, 0.724), (37.3, 39.3), (51.0, 51.8), (38.1, 39.0), (58.3, 58.5), (23.2, 23.4), (22.0, 22.6), (36.6, 35.8), (27.1, 29.6), (10.9, 11.1)]
large = [(0.656, 0.652), (66.7, 67.9), (68.3, 67.2), (67.1, 67.1), (66.7, 67.7), (27.1, 28.1), (40.2, 46.3), (59.2, 61.2), (70.7, 74.5), (37.2, 39.6)]
def tally(rows):
    sign = [np.sign(base - free) if k == 0 else np.sign(free - base) for k, (base, free) in enumerate(rows)]
    return {"free better": sign.count(1), "free worse": sign.count(-1), "tie": sign.count(0)}
print("Table 5 small:", tally(small), "| large:", tally(large))
assert tally(small) == {"free better": 9, "free worse": 1, "tie": 0}
assert tally(large) == {"free better": 8, "free worse": 1, "tie": 1}
gaps = [round(2.258 - 2.253, 3), round(2.085 - 2.080, 3)]   # section 4.5.3: sequence-wise minus aux-loss-free validation loss
print("validation-loss gap, 1B and 3B:", gaps)
assert gaps == [0.005, 0.005]

# DeepSeekMoE (arXiv 2401.06066v1) section 3.1: possible expert sets, top-2 of 16 vs top-8 of 64.
assert comb(16, 2) == 120 and comb(64, 8) == 4426165368
print("expert sets: C(16,2) =", comb(16, 2), "| C(64,8) =", comb(64, 8))
print("all checks passed")
```

Run with `timeout 60 uv run --no-project --with "numpy>=2,<3" python -I check.py`. Output:

```text
s = [0.5   0.75  0.25  0.875] | sum of s = 2.375
A (b=0):     chosen [2, 4] g' = [0.    0.75  0.    0.875] g = [0.     0.4615 0.     0.5385] h' = [34.0769 68.1538]
B (b= [ 0.  -0.5  0.5  0. ] ): chosen [3, 4] g' = [0.    0.    0.25  0.875] g = [0.     0.     0.2222 0.7778] h' = [34.7778 69.5556]
V2-style softmax s = [0.0882 0.2647 0.0294 0.6176] | its top-2 gates sum to 0.8824 (15/17); V3 gates sum to 1
code path, 2 groups keep 1, b=0: chosen [1, 2] weights [0.4 0.6]
loads [0, 6, 0, 6] mean 3.0 -> b after one step [ 0.001 -0.001  0.001 -0.001]
one expert 44.04M | total 671.0B | activated 37.6B, 36.6B without the input embedding table | routed share 0.974
Table 5 small: {'free better': 9, 'free worse': 1, 'tie': 0} | large: {'free better': 8, 'free worse': 1, 'tie': 1}
validation-loss gap, 1B and 3B: [0.005, 0.005]
expert sets: C(16,2) = 120 | C(64,8) = 4426165368
all checks passed
```

In case B, expert 3 is chosen with a score of $1/4$ over expert 2 with $3/4$, but
its weight is still built from $1/4$ (giving $2/9$ after rescaling), not from
the biased $0.75$, as the paper says: "The gating value, which will be multiplied
with the FFN output, is still derived from the original affinity score
$s_{i,t}$" (§2.1.2, after (16)). The bias still changes the weights through the
rescaling in (13): swapping expert 3 in for expert 2 changes the sum it divides
by, so expert 4's weight moves from $7/13$ (0.5385) to $7/9$ (0.7778) although
its score and its bias are unchanged. Sigmoid scores do not sum to 1 (here 2.375); V2's softmax top-2
weights sum to $15/17$; V3's rescaled weights sum to 1. The official code
(`Gate.forward`, `model.py` lines 563–595, re-typed in the script) agrees
with (13)–(16) up to a factor of 2.5. Its group restriction can change which
experts are chosen: with 2 groups and only 1 kept, case A picks experts 1 and 2
instead of 2 and 4. Both points are covered in the next section. The group
result assumes that each group is a run of consecutive experts, $\{1,2\}$ and
$\{3,4\}$: the script's `reshape` gives that (an assert checks it), and for the
official code's `view` it is my assumption, stated under "Node-limited routing".

## What the code must decide

Citations are to lines of `inference/model.py`, the three files in
`inference/configs/`, `inference/generate.py` and `inference/requirements.txt`
at commit `1398800e`. A configuration's values become the model's arguments
through `args = ModelArgs(**json.load(f))` (`generate.py` lines 112–113). The released code only runs the model: `model.py` contains
no loss, `backward` or optimizer (searched all 804 lines for those words), and
`Transformer.forward` is decorated with `@torch.inference_mode()` (line 768), a mode "to be
used when you are certain your operations will have no interactions with
autograd (e.g., model training)"
([PyTorch 2.4](https://docs.pytorch.org/docs/2.4/generated/torch.autograd.grad_mode.inference_mode.html)).
So it shows neither the bias update nor what training detaches. PyTorch links
below are to version 2.4, the version `inference/requirements.txt` pins
(`torch==2.4.1`, line 1).

- **Scale of the routed weights.** (13) makes the chosen weights sum to 1. The
  code then does `weights *= self.route_scale` (`model.py` line 594), with
  `"route_scale": 2.5` in `config_671B.json` (line 14), so the routed term in the code is 2.5 times the routed term of (12). §4.2
  mentions that V3 "multiplies additional scaling factors at the width
  bottlenecks" without values; the paper does not say whether this factor is
  one of them (the phrase occurs only in §4.2).
- **Normalization and the residual.** (12) uses the same $\mathbf u_t$ as the
  residual and as the experts' input. The code separates them:
  `x = x + self.ffn(self.ffn_norm(x))` (`Block.forward`, line 731), so the router and
  experts see the RMSNorm of `x` (`F.rms_norm(x, (self.dim,), self.weight, self.eps)`
  in `RMSNorm.forward`, line 291: each token vector normalized by its root mean square over
  its last dimension, then multiplied entrywise by the learned `weight`;
  [`torch.nn.RMSNorm`](https://docs.pytorch.org/docs/2.4/generated/torch.nn.RMSNorm.html),
  which the `F.rms_norm` page points to) while the residual adds `x` itself, and
  `MoE.forward` returns only `y + z` (routed plus shared; line 690). Figure 2 also draws an
  RMSNorm before the FFN. Equation (11) uses the same letter $\mathbf u_t$ for the
  attention output; in (12) it means "the FFN input" as quoted in the table.
- **Node-limited routing.** (16) chooses among all 256 experts. A _node_ is one
  machine; §4.2 places each layer's routed experts on "64 GPUs belonging to 8
  nodes", 32 experts per node. The paper adds a restriction in prose: "each token will be sent to at most $M$ nodes, which are
  selected according to the sum of the highest $\frac{K_r}{M}$ affinity scores of
  the experts distributed on each node" (§2.1.2, "Node-Limited Routing"), with
  $M=4$ (§4.2). The code
  implements this as 8 groups (`"n_expert_groups": 8`, `config_671B.json`
  line 12), made by `scores.view(x.size(0), self.n_groups, -1)` (`model.py`
  line 582). I assume that this `view` splits each token's 256 scores into 8
  runs of 32 consecutive experts (experts 1–32, 33–64, …), that the later
  `.flatten(1)` (line 589) joins them back in expert order, and that
  `x.view(-1, self.dim)` (line 677) makes each row of `x` one token; none of the
  PyTorch pages cited here shows the element order of `view` or `flatten`.
  Which experts share a group, the check's group result (experts 1 and 2), and
  the token rows in inside-out step 1 and under "Running only the chosen
  experts" rest on this assumption. Each
  group's score is the sum of its top 2 _biased_ scores (computed
  `scores.topk(2, dim=-1)[0].sum(dim=-1)` (line 586) after `scores = scores + self.bias` (line 580);
  [`torch.topk`](https://docs.pytorch.org/docs/2.4/generated/torch.topk.html)
  returns "the k largest elements" as "A namedtuple of (values, indices)", so
  `[0]` is the values and `[1]` the indices).
  It keeps 4 groups (`"n_limited_groups": 4`, `config_671B.json` line 13;
  `model.py` line 587): a mask of ones gets `False` at the
  kept groups' indices (`scatter_(1, indices, False)`, line 588;
  [PyTorch 2.4](https://docs.pytorch.org/docs/2.4/generated/torch.Tensor.scatter_.html)
  lists "value (Scalar) – the value to scatter"), and the other groups' scores are
  set to `-inf` (`masked_fill_`, line 589, which "Fills elements of self tensor with value where
  mask is True",
  [PyTorch 2.4](https://docs.pytorch.org/docs/2.4/generated/torch.Tensor.masked_fill_.html)),
  and only then takes the top 8 (line 590). The paper says "affinity scores" and does
  not say whether the bias is included there, nor do its other mentions of the
  4-node limit (§3.2.2, §4.2); the code includes it (lines 580 and 586). The check
  above shows, under the element-order assumption, that the group step can
  change the chosen set.
- **Ties at the cut-off.** The paper's $\mathrm{Topk}$ returns a set of score
  _values_, and (14)/(16) test membership by value, so two experts with equal
  scores at the boundary would both pass and more than $K_r$ experts would be
  chosen. The code takes indices, `torch.topk(scores, self.topk, dim=-1)[1]`
  (line 590), which returns exactly $K_r$ of them per token. Which of two tied experts is
  kept is open: the `torch.topk` page does not say.
- **Which model configurations have the bias.** `self.bias = nn.Parameter(…) if self.dim == 7168 else None`
  (`Gate.__init__`, `model.py` line 561), where `self.dim = args.dim` (line 554)
  is one width for the whole model. Line 3 of each file in `inference/configs/`
  sets it: `"dim": 7168` in `config_671B.json`, `"dim": 2048` in
  `config_16B.json` and `"dim": 5120` in `config_236B.json`, the only three
  files there. So only the full-size configuration has $b_i$, and
  then all 58 MoE layers of the 671B model have it (61 layers minus the 3 dense
  ones: `"n_layers": 61` and `"n_dense_layers": 3`, `config_671B.json` lines 6–7;
  `model.py` line 713 builds a dense MLP when `layer_id < args.n_dense_layers`).
  When the bias is absent, the group score is the group's maximum score (`scores.amax(dim=-1)`, line 584;
  [`torch.amax`](https://docs.pytorch.org/docs/2.4/generated/torch.amax.html)
  "Returns the maximum value of each slice of the input tensor in the given
  dimension(s) dim").
- **The bias update.** What counts as overloaded or underloaded, what happens at
  exactly average load, how load is counted, and the starting value of $b_i$:
  open. My check uses "load above the batch mean" only as a stand-in.
- **Running only the chosen experts.** (12) is written as a sum over all $N_r$
  experts. The code runs each local expert only on its tokens (`MoE.forward`,
  lines 681–686): `idx, top = torch.where(indices == i)` (line 685) gives the token rows and top-$K$ slots
  where expert `i` was chosen (one-argument `torch.where` "is identical to
  torch.nonzero(condition, as_tuple=True)",
  [PyTorch 2.4](https://docs.pytorch.org/docs/2.4/generated/torch.where.html),
  which returns "a tuple of 1-D tensors, one for each dimension in input",
  [PyTorch 2.4](https://docs.pytorch.org/docs/2.4/generated/torch.nonzero.html)),
  then `y[idx] += expert(x[idx]) * weights[idx, top, None]` (line 686).
- **Shared experts.** The code builds all shared experts as one MLP of width
  $N_s\cdot2048$ (`MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)`,
  `model.py` line 664, with `"moe_inter_dim": 2048` in `config_671B.json` line 5).
  This equals the sum in (12) for any $N_s$, because this MLP's output is a sum
  over its hidden units (my derivation).

## What the paper claims and what it shows

- **Sigmoid scores with rescaling of the chosen scores (13), (15):** a design
  change stated as a definition. The paper contains no sigmoid versus softmax
  comparison: "sigmoid" occurs only in (15), the §2.1.2 sentence quoted above,
  and §4.5.2. Both Table 5 baselines "use the sigmoid gating function with top-K
  affinity normalization" (§4.5.2), so Table 5 does not test it.
- **"an unbalanced expert load will lead to routing collapse (Shazeer et al.,
  2017) and diminish computational efficiency in scenarios with expert
  parallelism"** and **"too large an auxiliary loss will impair the model
  performance (Wang et al., 2024a)"** (§2.1.2): cited to other work, not tested in
  the sections read here. The paper defines neither "routing collapse" nor
  "expert parallelism" (whole-paper search; §3.2 names "64-way Expert
  Parallelism (EP)" without defining it). In my words: _routing collapse_ is the
  router settling on a few experts that then get most of the training. The
  cited paper, Shazeer et al. (2017), §4, describes it without that name: "the
  gating network tends to converge to a state where it always produces large
  weights for the same few experts. This imbalance is self-reinforcing, as the
  favored experts are trained more rapidly and thus are selected even more by
  the gating network." _Expert parallelism_ puts different experts on different
  GPUs and sends each token to the GPUs that hold its chosen experts; V3's
  placement is quoted under "Node-limited routing" above. A GPU whose experts
  receive more tokens then has more work than the others (my reasoning); for
  inference the paper states the goal as "each GPU processes approximately the
  same number of tokens" (§3.4.1, prefilling).
- **The bias method "achieves better performance than models that encourage load
  balance through pure auxiliary losses"** (§2.1.2): measured in Table 5 (§4.5.2).
  It compares a 15.7B-parameter model trained on 1.33T tokens and a
  228.7B-parameter model trained on 578B tokens, each with and without auxiliary
  losses. The script's tally: at the small scale the bias method is better on 9 of
  10 benchmarks and worse on MBPP; at the large scale it is better on 8, worse on
  MMLU and tied on DROP. The paper's own wording is "consistently achieves better
  model performance on most of the evaluation benchmarks". No run count is given and
  no $\pm$ appears anywhere in the paper, so the table does not show whether the differences exceed
  run-to-run variation. The ablated models "remove all auxiliary losses", while
  V3 itself keeps the small sequence-wise loss (see "Where the samples come
  from"), so the tested configuration is not exactly V3's.
- **"DeepSeek-V3 keeps balanced expert load during training"** (§2.1.2) and "keeps
  a good load balance during its full training" ("No Token-Dropping"): stated. The
  only expert-load plots in the paper (Figure 9, Appendix C) are for 16B models on Pile domains, not for
  the 671B training run.
- **When it reaches a similar level of batch-wise load balance, a batch-wise
  auxiliary loss performs about as well as the bias method** (§4.5.3):
  measured. Batch-wise balancing "does not enforce in-domain balance on each
  sequence", and the paper presents this experiment as an investigation of "the
  correlation between this flexibility and the advantage in model performance".
  Validation losses for 1B models are 2.258 (sequence-wise loss), 2.253 (bias method) and 2.253 (batch-wise loss); for 3B models, 2.085,
  2.080 and 2.080. The gap is 0.005 at both sizes (from the script). No variation
  across runs is reported. "This flexibility allows experts to better specialize
  in different domains" is supported by an observation, not a number: in the
  16B expert-load plots (see the previous bullet), which compare the bias method
  with an auxiliary-loss baseline and include no batch-wise-loss model, "the
  auxiliary-loss-free model demonstrates greater expert specialization patterns"
  (§4.5.3, Figure 9).
- **Node-limited routing means "our MoE training framework can nearly achieve
  full computation-communication overlap"** (§2.1.2), that is, GPUs send tokens
  to each other while they compute, so the transfers add little waiting time (my
  gloss): stated; §2.1.2 contains no measurement of it.
- **"DeepSeek-V3 does not drop any tokens during training"** (§2.1.2): a design
  decision, justified by "the effective load balancing strategy".
- **"671B total parameters, of which 37B are activated for each token"** (§4.2):
  printed. My recount from `config_671B.json` and the layer shapes in `model.py`
  (the script's comments give the lines), without the multi-token
  prediction module (an extra training-time module, §2.2), gives 671.0B total and 37.6B activated, or 36.6B if the
  input embedding table is not counted as activated. The paper does not state
  its counting rule (searched the whole paper: the abstract, §1, §4.2, §6 and
  the parameter rows of its tables give only the totals). Routed experts are 97.4% of all parameters, which is why
  choosing 8 of 256 per token makes the activated count small.
- **At decoding time** (when the model generates its output one token at a
  time, as opposed to prefilling, which processes the prompt; my gloss) the
  paper treats "the shared expert as a routed one",
  so that "each token will select 9 experts during routing" (§3.4.2). This
  is a deployment arrangement; (12) is unchanged, because the shared expert
  always runs with weight 1.
- **Many small experts**, which V3 takes from DeepSeekMoE:
  "DeepSeekMoE uses finer-grained experts and isolates some experts as shared
  ones" (§2.1.2). DeepSeekMoE argues for small experts by counting the possible
  sets of chosen experts: top-2 of 16 experts gives $\binom{16}{2}=120$ sets,
  and splitting each expert into 4 and choosing 8 of 64 gives
  $\binom{64}{8}=4{,}426{,}165{,}368$ (recomputed in the script), which "enhances
  the potential for achieving more accurate and targeted knowledge acquisition"
  (DeepSeekMoE §3.1): an argument. Its ablation (§4.4, Figure 3) is a
  measurement on its validation models, which have "approximately 2B total
  parameters" (§4.1.3): at equal total and activated parameters, splitting each
  expert into 2 or 4, for 32 or 64 experts in all, "corresponds to a
  continuous enhancement in overall model performance".
- **Shared experts**, the second half of the §2.1.2 sentence quoted above, are
  a design choice, not part of every MoE layer. DeepSeekMoE argues that without
  them "multiple experts may converge in acquiring shared knowledge in their
  respective parameters, thereby resulting in redundancy in expert parameters"
  (§3.2): an argument. It measures the effect on the same validation models:
  isolating one shared expert "yields improved performance across a majority of
  benchmarks" over GShard, the conventional design it starts from (§4.4,
  Figure 3); and in DeepSeekMoE 2B, disabling the
  shared expert and activating one more routed expert raises the Pile loss
  "from 1.808 to 2.414, even though we maintain the same computational cost"
  (§4.5). Qwen3 leaves them out: "Unlike Qwen2.5-MoE, the Qwen3-MoE design
  excludes shared experts" (Qwen3 Technical Report, §2), a sentence that gives no
  reason; it is the only occurrence of "shared" in that report's PDF text.

## Sources

- DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437v2),
  v2, submitted 18 Feb 2025: TeX source, PDF and PDF text.
- DeepSeek-AI, [DeepSeek-V2](https://arxiv.org/abs/2405.04434v5), v5, §2.2.1,
  equations (20)–(22).
- Dai et al., [DeepSeekMoE](https://arxiv.org/abs/2401.06066v1), v1 (the only
  version), §3.1, §3.2, §4.1.3, §4.4 with Figure 3, and §4.5: TeX source, PDF
  and PDF text.
- Shazeer et al., [Outrageously Large Neural Networks: The Sparsely-Gated
  Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538v1), v1, §4
  "Balancing Expert Utilization": PDF text.
- Qwen Team, [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388v1), v1
  (the only version), §2 "Architecture": PDF text.
- deepseek-ai/DeepSeek-V3 at commit
  [`1398800ebfcd49c048737e8b1aae69dee46ffefc`](https://github.com/deepseek-ai/DeepSeek-V3/tree/1398800ebfcd49c048737e8b1aae69dee46ffefc)
  (14 Feb 2025, the last commit to `inference/model.py` before v2; the next is
  26 Feb 2025): `inference/model.py` (`linear`, `RMSNorm`, `Gate`, `Expert`,
  `MLP`, `MoE`, `Block`, `Transformer`, and for the parameter recount `Linear`,
  `MLA` and `ParallelEmbedding`), `inference/configs/config_671B.json`,
  `config_16B.json` and `config_236B.json` (the three files in that directory
  at the commit), `inference/generate.py` and `inference/requirements.txt`.
- PyTorch 2.4 documentation, the pages linked in the text:
  [`torch.gather`](https://docs.pytorch.org/docs/2.4/generated/torch.gather.html),
  [`inference_mode`](https://docs.pytorch.org/docs/2.4/generated/torch.autograd.grad_mode.inference_mode.html),
  [`torch.nn.RMSNorm`](https://docs.pytorch.org/docs/2.4/generated/torch.nn.RMSNorm.html)
  (and [`torch.nn.functional.rms_norm`](https://docs.pytorch.org/docs/2.4/generated/torch.nn.functional.rms_norm.html)),
  [`torch.topk`](https://docs.pytorch.org/docs/2.4/generated/torch.topk.html),
  [`Tensor.scatter_`](https://docs.pytorch.org/docs/2.4/generated/torch.Tensor.scatter_.html),
  [`Tensor.masked_fill_`](https://docs.pytorch.org/docs/2.4/generated/torch.Tensor.masked_fill_.html),
  [`torch.amax`](https://docs.pytorch.org/docs/2.4/generated/torch.amax.html),
  [`torch.where`](https://docs.pytorch.org/docs/2.4/generated/torch.where.html)
  and [`torch.nonzero`](https://docs.pytorch.org/docs/2.4/generated/torch.nonzero.html).
- Named but not opened: Wang et al. (2024a), the auxiliary-loss-free load
  balancing paper; Lepikhin et al. (2021), GShard; Fedus et al. (2021).
