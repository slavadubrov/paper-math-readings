# DeepSeek-V3.2 DSA: score every earlier token cheaply, attend to at most 2048 of them, and train the scorer to copy the main attention

**Source:** DeepSeek-AI,
[DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556v1),
v1 (submitted 2 Dec 2025), §2.1, equations (1)–(2), and §2.1.1, equations
(3)–(4). Other locations used: footnotes 1–2, Figure 2, §2.2, §2.3, Figure 3,
§3 and §3.1, Appendix A and Figure 7. The official
code linked in footnote 2, `deepseek-ai/DeepSeek-V3.2-Exp` on Hugging Face, at
commit `194c67e12b1b0d6df0ef373ddcf215bc84027409` (18 Nov 2025, the newest
commit, made before v1 was submitted): `inference/model.py`,
`inference/kernel.py`, `inference/config_671B_v3.2.json`,
`inference/requirements.txt`; "L…" gives line numbers in these files at this
commit. MLA's own equations
come from [DeepSeek-V2](https://arxiv.org/abs/2405.04434v5), v5, §2.1.2–2.1.3.

## What it computes and what it changes

For each token, the model normally looks back at every earlier token. DSA puts
a small extra network in front of that, the _lightning indexer_: for the
current token $t$ it gives every earlier token $s$ one relevance number
$I_{t,s}$ (equation 1). The model's real attention then reads only the
earlier tokens with the $k=2048$ highest numbers, which is all of them until
there are more than 2048 (equation 2). Equations (3)
and (4) are the loss that trains the indexer: they push a probability
distribution made from the indexer's numbers toward the main attention's own
distribution over positions, first over all earlier tokens (equation 3), later
over the selected ones only (equation 4).

The paper says the change is relative to DeepSeek-V3.1-Terminus: "the only
architectural modification of DeepSeek-V3.2 is the introduction of DeepSeek
Sparse Attention (DSA) through continued training" (§2.1). That model uses MLA
(Multi-head Latent Attention, from DeepSeek-V2, which caches one compressed
vector per token instead of per-head keys and values), whose softmax for each
query head runs over every position from 1 to $t$ (DeepSeek-V2 v5, §2.1.3,
equation (18)). DSA keeps the same cached entry per token and the same
attention, restricts that softmax to the selected positions (at most $k$,
step 7), caches one extra small indexer
key per token (`k_cache`, `model.py` L453, L476), and adds the indexer's own
loss.

## The equation

$$
I_{t,s}=\sum_{j=1}^{H^I} w^I_{t,j}\cdot\mathrm{ReLU}\left(\mathbf q^I_{t,j}\cdot\mathbf k^I_s\right),
\tag{1}
$$

$$
\mathbf u_t=\mathrm{Attn}\left(\mathbf h_t,\left\{\mathbf c_s\,\middle|\,I_{t,s}\in\text{Top-k}\left(I_{t,:}\right)\right\}\right).
\tag{2}
$$

$$
\mathcal L^I=\sum_t\mathbb D_{\mathrm{KL}}\left(p_{t,:}\,\middle\|\,\mathrm{Softmax}\left(I_{t,:}\right)\right).
\tag{3}
$$

with the selected set
$\mathcal S_t=\left\{s\,\middle|\,I_{t,s}\in\text{Top-k}\left(I_{t,:}\right)\right\}$:

$$
\mathcal L^I=\sum_t\mathbb D_{\mathrm{KL}}\left(p_{t,\mathcal S_t}\,\middle\|\,\mathrm{Softmax}\left(I_{t,\mathcal S_t}\right)\right).
\tag{4}
$$

## Symbols

Values marked "config" come from `inference/config_671B_v3.2.json` at the
commit above; the paper does not print them.

| Symbol | Shape or type | Meaning | Role | Defined at |
| ------ | ------------- | ------- | ---- | ---------- |
| $I_{t,s}$ | one number, any sign; over a sequence of length $L$, an $L\times L$ table of which only $s\le t$ is used | index score: how much query token $t$ should look at token $s$ | computed intermediate | "The lightning indexer computes the index score $I_{t,s}$ between the query token $\mathbf h_t\in\mathbb R^d$ and a preceding token $\mathbf h_s\in\mathbb R^d$, determining which tokens to be selected by the query token" (§2.1, before eq. 1) |
| $t$ | position, counted from 1 | the query token, whose output is being computed | index | same sentence; "for the $t$-th query token" (§2.1.1, Dense Warm-up Stage) |
| $s$ | position | an earlier token being scored | index | "a preceding token $\mathbf h_s$" (§2.1). Whether $s=t$ is included is not said; see "What the code must decide" |
| $\mathbf h_t,\mathbf h_s$ | vectors in $\mathbb R^d$; $d=7168$ (config `dim`, L3) | the hidden state of token $t$ or $s$ entering the attention layer | function input; in the code, the _residual stream_ (the token's embedding plus the outputs of all earlier attention and feed-forward layers, added up) after RMS normalization, which divides the vector by the square root of (the mean of its squared coordinates $+10^{-6}$) and multiplies each coordinate by a trained weight (`attn_norm` in `Block.forward`, `model.py` L844–850, starting from the embedding, L904–906; `RMSNorm`, L280–306) | same sentence (§2.1) |
| $H^I$ | integer; 64 (config `index_n_heads`, L23) | number of indexer heads | fixed setting | "where $H^I$ denotes the number of indexer heads" (§2.1, after eq. 1) |
| $j$ | 1 to $H^I$ | which indexer head | index | the bounds of the sum in eq. (1). DeepSeek-V2's equation (18), cited above, uses $j$ for a position instead |
| $\mathbf q^I_{t,j}$ | vector in $\mathbb R^{d^I}$, one per head | the indexer's query for token $t$, head $j$ | computed intermediate, from trained indexer weights | "$\mathbf q^I_{t,j}\in\mathbb R^{d^I}$ and $w^I_{t,j}\in\mathbb R$ are derived from the query token $\mathbf h_t$" (§2.1, after eq. 1) |
| $d^I$ | integer; 128 (config `index_head_dim`, L24) | length of each indexer query and key | fixed setting | only as the shape $\mathbb R^{d^I}$ in that sentence (§2.1) |
| $w^I_{t,j}$ | one number per head | how much head $j$'s match counts for token $t$ | computed intermediate, from trained weights | same sentence (§2.1) |
| $\mathbf k^I_s$ | vector in $\mathbb R^{d^I}$; one per token, shared by all heads (it has no $j$) | the indexer's key for token $s$ | computed intermediate, cached per token (`k_cache`, `model.py` L453, L476) | "$\mathbf k^I_s\in\mathbb R^{d^I}$ is derived from the preceding token $\mathbf h_s$" (§2.1, after eq. 1) |
| $\mathbf u_t$ | vector in $\mathbb R^d$ (in the code, the output of `wo`, `model.py` L607; the docstring gives the input as `(batch_size, seq_len, dim)` and the output "the same shape as the input", L550–556) | the attention layer's output for token $t$ | computed intermediate | "the attention output $\mathbf u_t$ is computed by applying the attention mechanism between the query token $\mathbf h_t$ and the sparsely selected key-value entries $\{\mathbf c_s\}$" (§2.1, before eq. 2); Figure 2 labels it "Output Hidden $\mathbf u_t$" |
| $\mathrm{Attn}(\cdot,\cdot)$ | (query token, set of cached entries) → vector | MLA attention in its MQA mode: all 128 query heads (config `n_heads`, L8) of token $t$ read the same cached entries | function containing MLA's trained weights | same sentence, and "we implement DSA based on the MQA (Shazeer, 2019) mode of MLA, where each latent vector (the key-value entry of MLA) will be shared across all query heads of the query token" (§2.1, Instantiate DSA Under MLA; footnote 1 points to Appendix A) |
| $\mathbf c_s$ | vector; 576 numbers in the code (512 in `kv_cache`, 64 in `pe_cache`: `model.py` L541–542, config `kv_lora_rank` L17 and `qk_rope_head_dim` L19) | the cached key-value entry of token $s$ | computed intermediate (the KV cache) | "retrieves only the key-value entries $\{\mathbf c_s\}$ corresponding to the top-k index scores" (§2.1); "each latent vector (the key-value entry of MLA)" (§2.1). Figure 2 feeds $\{[\mathbf c^{KV}_t;\mathbf k^R_t]\}$ into the Top-k Selector; DeepSeek-V2 defines "$\mathbf c^{KV}_t\in\mathbb R^{d_c}$ is the compressed latent vector for keys and values" (§2.1.2) and "a shared key $\mathbf k^R_t\in\mathbb R^{d^R_h}$ to carry RoPE" (§2.1.3); RoPE is a position-dependent rotation of coordinate pairs |
| $I_{t,:}$ | vector of $t$ scores | row $t$ of the score table: all positions | computed intermediate | "Given the index scores $\{I_{t,s}\}$ for each query token $\mathbf h_t$" (§2.1) |
| $\text{Top-k}(\cdot)$ and $k$ | the $k$ largest entries of a row, or the whole row while it has at most $k$ (step 6); $k=2048$ | which scores survive | $k$: fixed setting | "the top-k index scores" (§2.1); "select 2048 key-value tokens for each query token" (§2.1.1, Sparse Training Stage); "where $k$ ($\ll L$) is the number of selected tokens" (§2.3) |
| $\mid$ inside $\{\,\}$ | – | "such that", as in a list comprehension's `if`; not a conditional probability | operator | the set notation of eq. (2) and of $\mathcal S_t$ (§2.1.1) |
| $\mathcal S_t$ | set of positions: $\min(t,k)$ of them in the code (step 7) | the positions selected for token $t$ | computed intermediate | "considering only the selected token set $\mathcal S_t=\{s\mid I_{t,s}\in\text{Top-k}(I_{t,:})\}$" (§2.1.1, before eq. 4) |
| $\mathcal L^I$ | one number | the indexer's loss | computed (the objective) | "we set a KL-divergence loss as the training objective of the indexer" (§2.1.1, before eq. 3) |
| $p_{t,:}$ | vector in $\mathbb R^t$, entries $\ge0$ summing to 1 | where the main attention looks from token $t$, all heads combined | computed intermediate, used as a fixed target (see "Where the samples come from") | "for the $t$-th query token, we first aggregate the main attention scores by summing across all attention heads. This sum is then L1-normalized along the sequence dimension to produce a target distribution $p_{t,:}\in\mathbb R^t$." (§2.1.1, Dense Warm-up Stage) |
| $p_{t,\mathcal S_t}$, $I_{t,\mathcal S_t}$ | vectors of $\vert\mathcal S_t\vert$ entries | the entries of $p_{t,:}$ and $I_{t,:}$ at the selected positions | computed intermediate | used in eq. (4) without a separate definition (§2.1.1) |
| $\mathbb D_{\mathrm{KL}}(\cdot\,\Vert\,\cdot)$ | (distribution, distribution) → number $\ge0$ | a number that measures how far the second distribution is from the first, and is 0 when they match: the KL divergence (formula in step 12). The order of the arguments matters; the target $p$ comes first | operator | named "a KL-divergence loss" (§2.1.1) |

**Not defined in the paper:** I searched the whole paper, including footnotes,
figure captions and Appendices A–E; the same search backs every other "the
paper does not …" in this reading.

- $p_{t,\mathcal S_t}$: whether it is renormalized to sum to 1 over
  $\mathcal S_t$, and which attention produces it in the sparse stage.
- "main attention scores" inside $p_{t,:}$: whether they are the attention
  weights after the softmax or the scores before it.
- $\mathbb D_{\mathrm{KL}}$: no formula for the divergence in (3)–(4), and no
  log base. The paper's only $\mathbb D_{\mathrm{KL}}$ formula, equation (7) in
  §3.1, is an "unbiased KL estimate" for reinforcement learning, computed from
  sampled tokens and their probabilities under the current, old and reference
  models ($\pi_\theta$, $\pi_{old}$, $\pi_{ref}$); it does not apply here.
  The name refers to the standard definition used in step 12; the log base
  only scales $\mathcal L^I$ by a constant.
- $\mathrm{Attn}$: no formula in this paper. It is given by reference to MLA
  (DeepSeek-V2 v5, equations (14)–(19)), Figures 2 and 7, and the code
  (`MLA.forward`, `model.py` L545–608).
- How $\mathbf q^I_{t,j}$, $w^I_{t,j}$ and $\mathbf k^I_s$ are "derived": the
  text says only "derived from". Figure 2 and the code (`Indexer.forward`,
  `model.py` L457–479) give the computation.

## Read from the inside out

Equation (1), for one query token $t$ and one earlier token $s$:

1. $\mathbf q^I_{t,j}\cdot\mathbf k^I_s$: two vectors of $d^I=128$ numbers →
   one number. The dot between two bold vectors is a dot product; it runs over
   the 128 coordinates.
2. $\mathrm{ReLU}(x)=\max(0,x)$: one number → one number $\ge0$. A head whose
   query points away from the key contributes 0. The paper's reason: "We choose
   ReLU as the activation function for throughput consideration." (§2.1)
3. $w^I_{t,j}\cdot\mathrm{ReLU}(\dots)$: here the dot multiplies two plain
   numbers. In the code nothing constrains the sign of $w$: it is the output of
   the linear layer `weights_proj` with no activation after it (`model.py`
   L449, L478–479), so a head can count against a token.
4. $\sum_{j=1}^{H^I}$: the sum runs over the 64 heads → $I_{t,s}$, one number of
   any sign. The free indices $t$ and $s$ give the result's shape: one score per
   (query, earlier token) pair. For one sequence in PyTorch, with `q` of shape
   `[T, H, D]`, `k` `[T, D]` and `w` `[T, H]`:
   `torch.einsum("ths,th->ts", torch.einsum("thd,sd->ths", q, k).relu(), w)`
   gives `[T, T]`, and positions $s>t$ are then masked. The check below does the same
   for one query token.

Superscripts: the $I$ on $\mathbf q^I$, $w^I$, $\mathbf k^I$, $H^I$ and $d^I$ is
a label meaning "indexer", not a power; $d^I$ is not $d$ raised to $I$. The
$=$ computes the left side from the right for every pair $(t,s)$.

Equation (2):

5. $I_{t,:}$: the colon means "every position": row $t$, with $t$ scores in the
   code ($s=1,\dots,t$; the causal mask, `model.py` L903 and L481–482, see
   "What the code must decide").
6. $\text{Top-k}(I_{t,:})$: the $k$ largest scores in that row. Row $t$ has
   only $t$ scores, so while $t\le k$ every position is kept and the selection
   changes nothing.
7. $I_{t,s}\in\text{Top-k}(\dots)$: $\in$ tests membership, so this asks
   whether $s$'s score is among the $k$ largest. Read literally, it compares
   values: if several positions share the $k$-th largest score (a tie at the
   boundary), all of them pass, and more than $k$ positions get in. The code
   instead keeps the positions that `topk` returns (`model.py` L483), so row
   $t$ gets exactly $\min(t,k)$ positions (see "What the code must decide";
   the check shows both readings). Wherever this reading says "at most $k$", it means the code's
   selection.
8. $\{\mathbf c_s\mid\dots\}$: a set comprehension,
   `[c[s] for s in positions if s in selected]`: at most $k$ cached entries
   (step 7).
9. $\mathrm{Attn}(\mathbf h_t,\text{that set})$: MLA attention of token $t$
   over those entries only. The softmax inside runs over $s\in\mathcal S_t$,
   which differs from all $s\le t$ only once $t>k$. Result:
   $\mathbf u_t\in\mathbb R^d$. The selection is the same for every query head.

Equation (3):

10. $\mathrm{Softmax}(I_{t,:})$: the softmax runs over $s$, the $t$ entries of
    row $t$ → a distribution: $t$ positive numbers that sum to 1. Call it
    $r_t$. The equation has no temperature; the code folds two constant factors
    into $w$ (`model.py` L478–479; see "What the code must decide").
11. $p_{t,:}$: take the main attention's per-head weights (this reads the
    paper's "scores" as weights after the softmax; see "Not defined in the
    paper") over the $t$ positions, add them across heads, and L1-normalize,
    which means divide by the sum of absolute values (for nonnegative numbers,
    their plain sum). If
    each head's weights already sum to 1, this equals the average over heads
    (the check confirms it).
12. $\mathbb D_{\mathrm{KL}}(p\,\Vert\,r)=\sum_s p_s\log(p_s/r_s)$: two
    distributions over the same $t$ positions → one number. It is $\ge0$ and
    is 0 only when $r=p$. Target first: positions where $p_s=0$ add nothing,
    and $r_s>0$ always (softmax output), so the value is finite. My derivation,
    checked numerically below: its gradient with respect to the score
    $I_{t,s}$ is $r_s-p_s$, the same as cross-entropy with soft labels. Why:
    $\mathbb D_{\mathrm{KL}}(p\Vert r)=\sum_{s'}p_{s'}\log p_{s'}-\sum_{s'}p_{s'}\log r_{s'}$,
    the first sum does not depend on $I$, and
    $\partial\log r_{s'}/\partial I_{t,s}=\mathbf 1[s'=s]-r_s$, where
    $\mathbf 1[s'=s]$ is 1 when $s'=s$ and 0 otherwise. So the gradient is
    $r_s\sum_{s'}p_{s'}-p_s$, which is $r_s-p_s$ when $\sum_{s'}p_{s'}=1$. In
    PyTorch 2.9: `F.kl_div(F.log_softmax(I_row, -1), p_row, reduction="sum")`.
    The arguments come in the reverse of the math order: the first is the
    log-probabilities of $r$ ([`kl_div`](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.kl_div.html):
    `input` is a "Tensor of arbitrary shape in log-probabilities";
    [`log_softmax`](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.log_softmax.html)
    applies "a softmax followed by a logarithm"), and the target is $p$, as
    probabilities by default (`log_target=False`). Each entry is
    `target * (target.log() - input)`, and `"sum"` adds them over $s$
    ([`KLDivLoss`](https://docs.pytorch.org/docs/2.9/generated/torch.nn.KLDivLoss.html)).
13. $\sum_t$: the sum runs over query tokens → one number, $\mathcal L^I$. It is a
    plain sum, not a mean.

Equation (4):

14. $\mathcal S_t$: the positions picked in steps 6–7.
15. $\mathrm{Softmax}(I_{t,\mathcal S_t})$: the softmax runs over the
    $\vert\mathcal S_t\vert$ selected scores only, $k$ of them once $t>k$ and
    all $t$ before (step 7) → a distribution on $\mathcal S_t$.
16. $p_{t,\mathcal S_t}$: $p$ at those positions. If $p$ comes from dense
    attention, these entries sum to at most 1. While $t\le k$ they sum to
    exactly 1: $\mathcal S_t$ holds every position, so token $t$'s term in (4)
    equals its term in (3). Once $t>k$ they sum to less than 1, because the
    positions left out have positive softmax weights (step 11's reading).
    Whether they are renormalized is open (see "Not defined in the paper").
    Without renormalizing, a sum below 1 can make the "KL" negative. My
    derivation, checked below: by
    step 12's line with $P=\sum_{s\in\mathcal S_t}p_{t,s}$ in place of 1, its
    gradient is $Pr_s-p_s=P\,(r_s-p_s/P)$, the renormalized gradient times
    $P$, so the two readings share a target and differ only in how much each
    token $t$ weighs. If $p$ comes from the sparse main attention of this stage, its
    weights outside $\mathcal S_t$ are 0 and the two readings agree.
17. A consequence (mine): a position outside $\mathcal S_t$ does not appear in
    token $t$'s term, so its score $I_{t,s}$ gets zero gradient from that term
    (checked below). The indexer's weights are shared across tokens and still
    change, but token $t$'s term does nothing directly to promote a position
    the indexer missed.

**Kind of statement:** equations (1) and (2) are computations: (1) returns one
number per pair $(t,s)$, and (2) returns $\mathbf u_t\in\mathbb R^d$, with a
set-builder inside it naming the entries the attention may read. Equations
(3) and (4) are objectives: losses minimized over the indexer's weights only,
plain sums over the tokens of the training sequences. The training stages and
data are in the next section.

## Where the samples come from

No expectation $\mathbb E$ appears. The sums over $t$ in (3) and (4) run over
tokens of fixed training data, not text generated by the model: "For both
stages, the distribution of training data is totally aligned with the 128K
long context extension data used for DeepSeek-V3.1-Terminus." (§2.1.1)
Post-training "also employs sparse attention in the same way as the sparse
continued pre-training stage" (§3), and its reinforcement learning trains on
responses sampled from "the old policy", an earlier copy of the model (§3.1);
the paper does not say whether $\mathcal L^I$ is still applied there.

- **Dense warm-up, equation (3):** "we keep dense attention and freeze all
  model parameters except for the lightning indexer" (§2.1.1). The selection of
  equation (2) is not used yet. The frozen main model computes $p_{t,:}$ on the
  same sequence. 1000 steps of 16 sequences of 128K tokens, learning rate
  $10^{-3}$.
- **Sparse training, equation (4):** "we introduce the fine-grained token
  selection mechanism and optimize all model parameters" (§2.1.1). The main
  model, now trained by the language-modeling loss, computes $p$. 15000 steps
  of 480 sequences of 128K tokens, learning rate $7.3\times10^{-6}$.
- **What is detached:** "we detach the indexer input from the computational
  graph for separate optimization. The training signal of the indexer is from
  only $\mathcal L^I$, while the optimization of the main model is according to
  only the language modeling loss." (§2.1.1) In PyTorch terms, the tensors fed
  to the indexer get `.detach()`. By Figure 2 and the code
  (`self.indexer(x, qr, start_pos, freqs_cis, mask)`, `model.py` L583 and
  L600) these are $\mathbf h_t$ and MLA's query latent $\mathbf c^Q_t$, the
  compressed vector from which MLA makes its queries (`qr`, L560); the paper
  does not list them. The quote's second sentence also means $p_{t,:}$ must
  carry no gradient into the main model. The language-modeling loss cannot
  reach the indexer through equation (2) anyway: the scores only choose positions, an integer choice with
  no gradient, and never multiply the values (my derivation).

## Check with small numbers

One query token and four positions ($s=4$ is the token itself, as in the
code: the causal mask, `model.py` L903, keeps $s=t$), two indexer heads,
$d^I=2$. The inputs are chosen so that ReLU zeroes
one head at $s=2$ and the other at $s=3$, keeps both at $s=1$, and one head
weight is negative, so the score at $s=3$ is negative; the inputs are chosen
so that every score is an integer. The check uses $k=2$,
plus a tie at the boundary and a row with fewer than $k$ positions. A
single-head attention stands in for MLA (my simplification); its scores
$(\log3,0)$ give the exact weights $3/4,1/4$. Two heads with power-of-two
weights give $p$. A second token with $p=(1/2,1/2)$ and $I=(0,\log3)$ has the
exact loss $\tfrac12\log\tfrac43$; its row has $t=2\le k$ positions, so it
also covers equation (4) when nothing is left out. Both readings of $p_{t,\mathcal S_t}$ are
computed. The script also recomputes the token totals the paper prints and my
operation counts used in the last section.

```python
import numpy as np

ln = np.log

# ---- Equation (1): index scores I[t, s], one query token t, four positions s = 1..4
q = np.array([[1.0, 0.0],      # q^I_{t,1}  (H^I = 2 heads, d^I = 2)
              [0.0, 1.0]])     # q^I_{t,2}
w = np.array([2.0, -1.0])      # w^I_{t,1}, w^I_{t,2}  (any real number)
k = np.array([[1.0, 1.0],      # k^I_1
              [2.0, -1.0],     # k^I_2
              [-1.0, 3.0],     # k^I_3
              [1.5, 0.0]])     # k^I_4  (s = t itself, as in the official code, model.py L903)
dots = np.einsum("jd,sd->js", q, k)          # q^I_{t,j} . k^I_s   -> [H^I, s]
relu = np.maximum(dots, 0.0)
I = (w[:, None] * relu).sum(axis=0)          # sum over heads j    -> [s]
print("dots per head:\n", dots)
print("I_t,: =", I)
assert (dots < 0).any() and (dots > 0).any()  # ReLU clips some, keeps others
assert np.allclose(I, [1.0, 4.0, -3.0, 3.0])

# ---- Equation (2): top-k selection, then attention over the selected entries only
kk = 2
sel = np.sort(np.argsort(-I, kind="stable")[:kk])       # positions, 0-based
print("selected positions (1-based):", sel + 1)
assert list(sel + 1) == [2, 4]

c = np.array([[5.0, 5.0], [1.0, 0.0], [7.0, 7.0], [0.0, 1.0]])  # stand-in entries c_s
qa = np.array([ln(3.0), 0.0])                                    # stand-in query
def attn(qv, C):                      # single-head stand-in for MLA's Attn (mine)
    a = np.exp(C @ qv - (C @ qv).max()); a /= a.sum()
    return a @ C
u_gather = attn(qa, c[sel])                                      # gather k entries
scores = c @ qa
mask = np.full(4, -np.inf); mask[sel] = 0.0                       # official-code style, model.py L584-588
a = np.exp(scores + mask - (scores + mask).max()); a /= a.sum()
u_mask = a @ c
print("u_t (gather) =", u_gather, " u_t (-inf mask) =", u_mask)
assert np.allclose(u_gather, [0.75, 0.25]) and np.allclose(u_gather, u_mask)

# fewer than k candidates (t = 1): the code takes min(k, t), i.e. every position (model.py L483)
assert len(np.argsort(-I[:1])[:min(kk, 1)]) == 1
# a tie at the boundary: reading "I_ts in Top-k(values)" keeps 3 entries, index top-k keeps 2
I_tie = np.array([1.0, 4.0, -3.0, 1.0])
top_vals = np.sort(I_tie)[-kk:]
by_value = [s + 1 for s in range(4) if I_tie[s] in top_vals]
by_index = sorted(int(s) + 1 for s in np.argsort(-I_tie, kind="stable")[:kk])
print("tie: by value", by_value, " by index", by_index)
assert by_value == [1, 2, 4] and len(by_index) == 2

# ---- Equation (3): target p from two heads, KL to Softmax(I)
def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()
def kl(p, qd):
    return float(np.sum(p * ln(p / qd)))
headA = np.array([1/2, 1/4, 1/8, 1/8])       # main-attention weights, head A
headB = np.array([1/4, 1/4, 1/4, 1/4])       # head B
summed = headA + headB
p = summed / np.abs(summed).sum()            # L1-normalize along the sequence
print("p_t,: =", p)
assert np.allclose(p, [3/8, 1/4, 3/16, 3/16]) and np.allclose(p, (headA + headB) / 2)
qI = softmax(I)
loss_t4 = kl(p, qI)
print("Softmax(I_t,:) =", qI.round(4), " KL =", round(loss_t4, 4))
assert loss_t4 > 0
assert abs(kl(p, softmax(ln(p) + 7.0))) < 1e-12           # zero when Softmax(I) = p
# second query token (t = 2, two positions): p = (1/2, 1/2), I = (0, log 3)
p2, I2 = np.array([0.5, 0.5]), np.array([0.0, ln(3.0)])
loss_t2 = kl(p2, softmax(I2))
assert np.isclose(loss_t2, 0.5 * ln(4 / 3))
print("L^I over t in {2, 4} =", round(loss_t2 + loss_t4, 4))
# gradient of the KL with respect to the scores is Softmax(I) - p
eps = 1e-6
num = np.array([(kl(p, softmax(I + eps * np.eye(4)[s])) -
                 kl(p, softmax(I - eps * np.eye(4)[s]))) / (2 * eps) for s in range(4)])
print("dKL/dI numeric =", num.round(4), " Softmax(I)-p =", (qI - p).round(4))
assert np.allclose(num, qI - p, atol=1e-6)
# if "scores" meant pre-softmax logits, L1-normalizing can leave negative entries
logits = np.array([2.0, -1.0, 0.5, 0.0])
assert (logits / np.abs(logits).sum()).min() < 0

# ---- Equation (4): only the selected set S_t = {2, 4}
pS = p[sel]                                   # (1/4, 3/16), sums to 7/16
qS = softmax(I[sel])
kl_renorm = kl(pS / pS.sum(), qS)             # reading A: renormalize p over S_t
kl_raw = float(np.sum(pS * ln(pS / qS)))      # reading B: restrict p, no renormalizing
print("p_t,S =", pS, " sum =", pS.sum(), " KL renorm =", round(kl_renorm, 4),
      " raw =", round(kl_raw, 4))
assert np.isclose(pS.sum(), 7 / 16) and kl_raw < 0 < kl_renorm
# both readings have the same per-token target; reading B scales the gradient by sum(pS)
g_A = qS - pS / pS.sum()
g_B = np.array([(np.sum(pS * ln(pS / softmax(I[sel] + eps * np.eye(2)[i]))) -
                 np.sum(pS * ln(pS / softmax(I[sel] - eps * np.eye(2)[i])))) / (2 * eps)
                for i in range(2)])
assert np.allclose(g_B, pS.sum() * g_A, atol=1e-6)
# an unselected score gets no gradient from token t's term in (4)
def loss4(Irow):
    return kl(pS / pS.sum(), softmax(Irow[sel]))
for s in [0, 2]:
    d = (loss4(I + eps * np.eye(4)[s]) - loss4(I - eps * np.eye(4)[s])) / (2 * eps)
    assert d == 0.0
# a short row (t = 2, not more than k = 2 positions): S_t is every position,
# p_{t,S_t} sums to 1, and token t's term in (4) equals its term in (3)
sel2 = np.sort(np.argsort(-I2, kind="stable")[:min(kk, 2)])
print("t = 2: S_t =", sel2 + 1, " sum of p_t,S =", p2[sel2].sum())
assert list(sel2 + 1) == [1, 2] and p2[sel2].sum() == 1.0
assert np.isclose(kl(p2[sel2], softmax(I2[sel2])), loss_t2)

# ---- Numbers the paper prints (2.1B and 943.7B tokens)
for K in (131072, 128000):
    print(f"128K = {K}: warm-up {1000*16*K/1e9:.1f}B, sparse {15000*480*K/1e9:.1f}B")
assert round(1000 * 16 * 131072 / 1e9, 1) == 2.1
assert round(15000 * 480 * 131072 / 1e9, 1) == 943.7

# ---- My arithmetic: attention pairs and multiply-adds at L = 128K, config values
L, topk = 131072, 2048
dense_pairs = L * (L + 1) // 2                           # s <= t
sparse_pairs = topk * (topk + 1) // 2 + (L - topk) * topk  # min(t, k) per query
idx_mac = 64 * 128                                       # H^I * d^I per (t, s) pair
mqa_mac = 128 * ((512 + 64) + 512)                       # 128 heads: score + value
print("pairs dense/sparse =", dense_pairs, sparse_pairs, round(dense_pairs / sparse_pairs, 1))
assert L / topk == 64                                    # last query: one position in 64
print("last position reads", topk, "of", L, "= one in", L // topk)
print("MAC per pair: indexer", idx_mac, " MLA-MQA", mqa_mac, " ratio", round(mqa_mac / idx_mac, 1))
print("MACs at 128K: indexer %.2e, sparse core %.2e, dense core %.2e" %
      (idx_mac * dense_pairs, mqa_mac * sparse_pairs, mqa_mac * dense_pairs))
print("dense core / (indexer + sparse core) = %.1f" %
      (mqa_mac * dense_pairs / (idx_mac * dense_pairs + mqa_mac * sparse_pairs)))
print("all checks passed")
```

Output:

```text
dots per head:
 [[ 1.   2.  -1.   1.5]
 [ 1.  -1.   3.   0. ]]
I_t,: = [ 1.  4. -3.  3.]
selected positions (1-based): [2 4]
u_t (gather) = [0.75 0.25]  u_t (-inf mask) = [0.75 0.25]
tie: by value [1, 2, 4]  by index [1, 2]
p_t,: = [0.375  0.25   0.1875 0.1875]
Softmax(I_t,:) = [3.510e-02 7.049e-01 6.000e-04 2.593e-01]  KL = 1.6325
L^I over t in {2, 4} = 1.7764
dKL/dI numeric = [-0.3399  0.4549 -0.1869  0.0718]  Softmax(I)-p = [-0.3399  0.4549 -0.1869  0.0718]
p_t,S = [0.25   0.1875]  sum = 0.4375  KL renorm = 0.0589  raw = -0.3359
t = 2: S_t = [1 2]  sum of p_t,S = 1.0
128K = 131072: warm-up 2.1B, sparse 943.7B
128K = 128000: warm-up 2.0B, sparse 921.6B
pairs dense/sparse = 8590000128 266339328 32.3
last position reads 2048 of 131072 = one in 64
MAC per pair: indexer 8192  MLA-MQA 139264  ratio 17.0
MACs at 128K: indexer 7.04e+13, sparse core 3.71e+13, dense core 1.20e+15
dense core / (indexer + sparse core) = 11.1
all checks passed
```

The indexer ranks position 2 first and misses position 1, where the main
attention puts the most weight (3/8). Equation (3) therefore gives a large loss
(1.6325), with gradient $\mathrm{Softmax}(I)-p$, which raises position 1's
score. Equation (4) sees only positions 2 and 4: renormalized it gives 0.0589,
unrenormalized −0.3359, and position 1's score gets no gradient from it. For
the second token, $\mathcal S_t$ holds both positions, $p$ sums to 1 over it,
and its term in (4) equals its term in (3). Only
$128\text{K}=131{,}072$ tokens reproduces both totals the paper prints, 2.1B
and 943.7B.

## What the code must decide

All at commit `194c67e`. The repository has inference code only: its root
holds `inference/`, the weights, configs and tokenizer files, and no training
code. So every choice about equations (3)–(4) stays open.

- **Which positions are candidates:** $s\le t$, including the token itself.
  `Transformer.forward` builds
  `torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)`
  (`model.py` L903), and `Indexer.forward` adds it to the scores
  (`index_score += mask`, L481–482). `triu_` is the in-place `triu`; with the
  argument 1 it keeps only the entries above the main diagonal ($s>t$, which
  stay $-\infty$) and sets the rest to 0: in PyTorch 2.9 (see Sources for the
  version), "A positive value excludes just as many diagonals above the main
  diagonal" from the kept part
  ([`torch.triu` docstring](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_torch_docs.py)).
- **How $\mathbf q^I$, $\mathbf k^I$ and $w^I$ are made** (`Indexer.forward`,
  `model.py` L457–479):
  - $\mathbf q^I$: `q = self.wq_b(qr)` (L460), where `qr` is MLA's query
    latent, computed in `MLA.forward` as `qr = self.q_norm(self.wq_a(x))`
    (L560).
  - $\mathbf k^I$: `k = self.wk(x)`, then `k = self.k_norm(k)` (L466–467),
    one 128-vector per token (`self.wk = Linear(self.dim, self.head_dim)`,
    L446). `k_norm` is a LayerNorm (L447; the class, L309–321, calls
    `F.layer_norm` with a trained weight $\gamma$, a trained bias $\beta$ and
    $\epsilon=10^{-6}$): it subtracts the mean of the 128
    coordinates, divides by their standard deviation, then multiplies each
    coordinate by $\gamma$ and adds $\beta$,
    $y=\frac{x-\mathrm E[x]}{\sqrt{\mathrm{Var}[x]+\epsilon}}*\gamma+\beta$
    in the PyTorch 2.9
    [`LayerNorm` docstring](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/nn/modules/normalization.py),
    to which the `F.layer_norm` docstring refers. The `LayerNorm` docstring
    takes the mean and variance over the last dimension when the shape passed
    in has one entry, as the code's `(self.dim,)` does.
  - Both get RoPE (see the $\mathbf c_s$ row) on their first 64 coordinates
    ("rope in indexer is not interleaved", L462–465 and L468–471; 64 is config
    `qk_rope_head_dim`, L19). Then both go through a Hadamard rotation
    (L472–473): `rotate_activation` (L428–432) returns
    `hadamard_transform(x, scale=hidden_size ** -0.5)`, with
    `hidden_size = x.size(-1)`, 128 here, from the `fast_hadamard_transform`
    package (unpinned in `inference/requirements.txt`, L4). Its README at tag
    `v1.0.4.post1` (see Sources) says the call is "Equivalent to F.linear(x, torch.tensor(scipy.linalg.hadamard(dim))) * scale"
    and pads only "If dim is not a power of 2"; 128 is a power of 2, so
    nothing is padded. A Hadamard matrix $H$ of
    size $n$ has entries $\pm1$ and mutually orthogonal rows, so $HH^\top=nI$
    and $n^{-1/2}H$ is orthogonal; the same orthogonal map applied to
    $\mathbf q$ and $\mathbf k$ leaves $\mathbf q\cdot\mathbf k$ unchanged in
    exact arithmetic (my derivation). Both are then quantized to FP8 (8-bit
    floats) with one scale per 128 values (L474–475; `act_quant`,
    `kernel.py` L87–111, with `block_size = 128`, `model.py` L15).
  - $w$: `self.weights_proj(x.float())` in FP32 (32-bit floats; L449, L478),
    multiplied by $1/\sqrt{H^I}$ (`self.n_heads ** -0.5`, L478), by
    $1/\sqrt{d^I}$ (`self.softmax_scale`, L450 and L479) and by the FP8 scale
    of $\mathbf q$ (`q_scale`, L479).
- **The order inside equation (1):** the kernel (a function that runs on the
  GPU) `fp8_index_kernel` in `inference/kernel.py` computes
  `T.max(logits[i3_n, i_h], 0) * q_s_frag[i_h]` (L241: ReLU, then the head
  weight, which `Indexer.forward` passes in as `weights`, `model.py` L480),
  sums over heads (`T.reduce_sum`, `kernel.py` L244), then multiplies by
  $\mathbf k$'s scale (L247). This matches equation (1). Keys are cached in
  FP8 (`k_cache`, `float8_e4m3fn`, `model.py` L453), and the scores are FP32
  (`kernel.py` L214, L230).
- **Scale in the softmax of (3)–(4):** the constants folded into $w$ do not
  change which positions are top-k, but they act as a temperature if the same
  $I$ goes into $\mathrm{Softmax}(I)$ during training. Open.
- **Fewer than $k$ candidates:**
  `index_score.topk(min(self.index_topk, end_pos), dim=-1)` (`model.py` L483)
  returns up to 2048 positions (`index_topk`, config L25). For one new token
  against the cache, row $t$ has only $t$ entries (the keys
  `self.k_cache[:bsz, :end_pos]`, L480), so `min` alone keeps all of them.
  When a whole prompt is processed at once, a row with $t<2048$ can also get
  positions after $t$, whose scores the causal mask set to $-\infty$; the mask
  is added back after the selection (`index_mask += mask`, L585, see "Applying
  the selection"), so the row keeps exactly its $t$ positions. Hence
  $\vert\mathcal S_t\vert=\min(t,k)$, under the assumption stated in
  "Applying the selection".
- **Ties at the boundary:** the code uses the `topk` call above; neither the
  paper nor the code states which of two equal scores wins, and the PyTorch 2.9
  [`torch.topk` docstring](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_torch_docs.py)
  says "the indices of tied elements are not guaranteed to be stable and may
  vary across different invocations". Open.
- **Applying the selection:** the code builds an additive mask,
  `torch.full(…, float("-inf"), …).scatter_(-1, topk_indices, 0)`, adds the
  causal mask back in prefill, the pass that processes the whole prompt at
  once (`index_mask += mask`), gives it a head axis of size 1
  (`index_mask.unsqueeze(2)`), adds it to the scores of every head and applies
  `softmax(dim=-1)` (`model.py` L584–588; decode, L601–604). PyTorch 2.9
  broadcasts the size-1 axis over all heads: tensors "can be automatically
  expanded to be of equal sizes (without making copies of the data)" when,
  along each dimension, the sizes are equal or one of them is 1
  ([broadcasting semantics](https://github.com/pytorch/pytorch/blob/v2.9.1/docs/source/notes/broadcasting.rst)).
  My assumption, which I could not source in the PyTorch 2.9 documentation:
  `topk` ranks a $-\infty$ score below every finite one, and `softmax` gives a
  $-\infty$ score the weight 0, as $e^{-\infty}=0$. The two statements that
  depend on it are that this mask equals gathering the selected entries (the
  check shows the equality with NumPy) and that
  $\vert\mathcal S_t\vert=\min(t,k)$ (above).
  This reference code still computes all $t$ scores (L480, L580), so it pins
  down the math, not the speed.
- **MHA or MQA mode:** the prefill branch of `MLA.forward` (`model.py`
  L574–589) uses MHA-mode einsums (multi-head attention: each head gets its
  own keys and values, expanded from the cached entry, L576–579), and the
  decode branch (L590–606), which generates one token at a time against the
  cache, uses MQA mode, with `wkv_b` absorbed into the query and the output
  (L594–595, L605–606). These are two orders of the same matrix products:
  DeepSeek-V2 says "$W^{UK}$ can be absorbed into $W^Q$, and $W^{UV}$ can be
  absorbed into $W^O$" (§2.1.2).
- **For the losses, open:** the points under "Not defined in the paper", and
  also:
  - sum or mean over sequences;
  - how the per-layer losses combine (each of the 61 layers, config
    `n_layers` L6, has its own indexer: `self.indexer = Indexer(args)` in
    `MLA.__init__`, `model.py` L539, and every `Block` holds an `MLA`, L826
    and L882–883);
  - exactly which tensors are detached.

## What the paper claims and what it shows

- **DSA is the only architectural change from V3.1-Terminus** (§2.1): a
  description of the design.
- **ReLU "for throughput consideration"; the indexer's "computational
  efficiency is remarkable"** because it "has a small number of heads and can be
  implemented in FP8" (§2.1): stated reasons. The paper shows no separate
  measurement of the indexer.
- **Core attention drops from $O(L^2)$ to $O(Lk)$** (§2.3): this follows from
  counting, since each query reads at most $k$ entries (step 7). At
  $L=128\text{K}$ my
  count is $8.59\times10^9$ pairs dense against $2.66\times10^8$ selected,
  32.3 times fewer. At the last position of a 131,072-token sequence, the main
  attention reads 2,048 of 131,072 positions: one in 64.
- **The indexer is still $O(L^2)$** but "it requires much less computation
  compared with MLA in DeepSeek-V3.1-Terminus" (§2.3): asserted without a
  number. My count from the config (`n_heads` L8, `kv_lora_rank` L17,
  `qk_rope_head_dim` L19, `index_n_heads` L23, `index_head_dim` L24),
  multiply-adds only:
  - 8,192 per pair for the indexer against 139,264 for MLA in MQA mode, 17.0
    times fewer;
  - at 128K, by this count, the indexer ($7.04\times10^{13}$) costs more than
    the sparse core attention ($3.71\times10^{13}$);
  - the two together are 11.1 times below dense MQA-mode attention, which
    V3.1-Terminus uses for decoding (Figure 7 caption).

  This count ignores projections, the speed of FP8 against BF16 (a 16-bit
  float format) arithmetic, memory reads and kernel efficiency.
- **End-to-end speedup:** estimated from benchmarks. Figure 3 plots cost in
  dollars per million tokens against token position, separately for
  prefilling and decoding. "These costs are estimated from benchmarking the
  actual service deployed on H800 GPUs, at a rental price of 2 USD per GPU
  hour." (§2.3) It is not an operation count. By eye, V3.1-Terminus's cost
  rises steeply with position in both panels, and V3.2's rises far more slowly.
  The figure shows no error bars and no run counts. For short prefills, "we
  specially implement a masked MHA mode to simulate DSA" (§2.3).
- **Quality is kept** (§2.2): measured on DeepSeek-V3.2-Exp, which the paper
  says has "exactly the same architecture" as V3.2 (§2.1). The results:
  - benchmarks: "we do not observe substantial performance degradation compared
    with DeepSeek-V3.1-Terminus, on both short- and long-context tasks";
  - ChatbotArena Elo scores (ratings computed from head-to-head comparisons)
    "are closely matched";
  - on AA-LCR, V3.2-Exp "scores four points higher" in reasoning mode;
  - on Fiction.liveBench, V3.2-Exp "consistently outperforms".

  §2.2 gives no run counts or intervals, so the paper does not show whether
  these differences exceed run-to-run noise.
- **The training recipe** (dense warm-up, then sparse training with the KL
  alignment) is described, not tested. The paper reports no ablation of it
  (an experiment that removes or changes one part to measure its effect); its
  ablation section, §4.3, concerns synthetic agentic tasks.

## Sources

- DeepSeek-AI, [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556v1),
  v1: TeX source (`sections/01-method.tex`, `sections/02-post.tex`,
  `sections/appendix.tex`, `main.tex` macros) and the PDF and its text.
- DeepSeek-AI, [DeepSeek-V2](https://arxiv.org/abs/2405.04434v5), v5:
  §2.1.1–2.1.3, equations (9)–(19).
- Official code linked in footnote 2, at a fixed commit:
  [`deepseek-ai/DeepSeek-V3.2-Exp` inference/ at `194c67e`](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/194c67e12b1b0d6df0ef373ddcf215bc84027409/inference):
  `model.py` (`Indexer`, `MLA`, `Block`, `Transformer`), `kernel.py`
  (`fp8_index_kernel`, `fp8_index`), `config_671B_v3.2.json`,
  `requirements.txt`. Commit list from the Hugging Face API.
- `fast_hadamard_transform`, which `rotate_activation` imports:
  [README at tag `v1.0.4.post1`](https://github.com/Dao-AILab/fast-hadamard-transform/blob/v1.0.4.post1/README.md),
  the latest tag before commit `194c67e`.
- PyTorch 2.9. Nothing pins `torch`: `inference/requirements.txt` lists it
  with no version (L1), and its one pinned package, `tilelang==0.1.6` (L5),
  requires `torch` with no version in its PyPI metadata. I took 2.9 because
  2.9.1, uploaded to PyPI on 12 Nov 2025, is the latest release before the
  commit, and the PyTorch calls this reading relies on (`triu_`, `topk`,
  `scatter_`, `softmax`, `F.layer_norm`, `float8_e4m3fn`) are documented in
  it. Documentation linked in step 12: `torch.nn.functional.kl_div`,
  `torch.nn.KLDivLoss`, `torch.nn.functional.log_softmax`. Docstrings in the
  source at tag `v2.9.1`:
  [`torch/_torch_docs.py`](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_torch_docs.py)
  (`torch.triu`, `torch.topk`),
  [`torch/_tensor_docs.py`](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_tensor_docs.py)
  (`Tensor.triu_`, `Tensor.scatter_`),
  [`torch/nn/modules/normalization.py`](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/nn/modules/normalization.py)
  (`LayerNorm`),
  [`torch/nn/functional.py`](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/nn/functional.py)
  (`softmax`, `layer_norm`),
  [`torch/nn/modules/activation.py`](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/nn/modules/activation.py)
  (`Softmax`),
  [`docs/source/notes/broadcasting.rst`](https://github.com/pytorch/pytorch/blob/v2.9.1/docs/source/notes/broadcasting.rst)
  and
  [`docs/source/tensor_attributes.rst`](https://github.com/pytorch/pytorch/blob/v2.9.1/docs/source/tensor_attributes.rst)
  (`float8_e4m3fn`).
- PyPI metadata: [`torch`](https://pypi.org/pypi/torch/json) (release upload
  dates) and [`tilelang` 0.1.6](https://pypi.org/pypi/tilelang/0.1.6/json)
  (its requirements).
- Cited by the paper, not opened: Yuan et al. (2025), Native Sparse Attention;
  Shazeer (2019), Multi-Query Attention.
