# Gated DeltaNet: how a layer updates its memory matrix token by token, and how training computes the same thing chunk by chunk

**Source:** Songlin Yang, Jan Kautz, Ali Hatamizadeh,
[Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464v3),
v3 (6 Mar 2025, the ICLR 2025 version), §3.1, equation (10). Other locations
used: §1; §2.1 (linear attention, Mamba2, chunk notation, equations (1)–(2),
footnote 1); §2.2 (delta rule, equations (3)–(9), footnote 2); §3.3 (chunkwise
algorithm, unnumbered equations); §3.4, footnote 4 and Figure 1 (projections,
normalization, hybrid models); §4 and Figure 3 (plotted coordinates in the TeX
source); §5; Appendix A (proof); Appendix B.2 and Table S.1
(ablations). Official code, linked from the abstract:
[NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet) at commit
`3154b25f2361eab4831999fb8deb1641d1a41a16` (23 Feb 2025). The repository has no
tags; this is the last commit before v3's date, and the next one (13 Mar 2025)
is titled "[Readme] add some instructions for evaluation". Code paths below are
relative to that commit. Its `Dockerfile` fixes the versions of the libraries
whose behaviour the reading relies on only in part: it builds on the
`pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel` image (L2) and installs
`causal-conv1d>=1.2.0`, a minimum only (L22), einops with no version (L23) and
flash-linear-attention ("fla", which the README links as
`fla-org/flash-linear-attention`, L177) from its default branch
(`Run pip install -U git+https://github.com/sustcsonglin/flash-linear-attention`,
L32). So library behaviour is cited from the PyTorch 2.3 documentation, for the
image's PyTorch 2.3.1; from fla at commit
`326184c45f42380063fb10bdda31d394b0c8c1b7` (21 Nov 2024), which is not the last
fla commit before `3154b25`'s date ("Sources" says why); from causal-conv1d at
tag `v1.4.0`, not the Dockerfile's minimum, because fla's short convolution,
which calls it, asks for `causal-conv1d>=1.4.0` (fla's
`fla/modules/convolution.py` L107–112 and `setup.py` L138); and from einops at tag `v0.8.1`, its last release
before `3154b25`'s date, which has the `rearrange` and `repeat` that the
official code imports (`lit_gpt/gated_delta_net.py` L17; `einops/einops.py`
L545, L603).

## What it computes and what it changes

A Transformer's softmax attention keeps every past key and value, so its KV cache
grows by one key and one value per token, and compares each new query with all of
them; §1: "the self-attention component scales quadratically with sequence
length". Linear attention drops the softmax. §2.1 writes the linear Transformer,
"when excluding normalization and query/key activations", as
$\mathbf S_t=\mathbf S_{t-1}+\mathbf v_t\mathbf k_t^\intercal$,
$\mathbf o_t=\mathbf S_t\mathbf q_t$ (the matrix $\mathbf S_t$ is the memory
after token $t$; $\mathbf k_t$, $\mathbf v_t$, $\mathbf q_t$ are the key, value
and query computed from token $t$, as column vectors; $\mathbf o_t$ is its
output; $^\intercal$ turns a column into a row), which it expands to
$\mathbf o_t=\sum_{i=1}^t\mathbf v_i(\mathbf k_i^\intercal\mathbf q_t)$: attention
whose weights are raw dot products, with a memory that only adds. §1 credits this
form with "substantially reducing memory requirements during inference by
reframing as a linear RNN with matrix-valued states".

Gated DeltaNet keeps that design: each head of a layer keeps one fixed-size
matrix as its memory, $d_v\times d_k$ ($d_v$ numbers per value, $d_k$ per key)
at any sequence length, and stores value
vectors in it, so that multiplying the matrix by a key vector returns the value
stored under that key. Every stored value reads back exactly only when the keys
are perpendicular to each other (orthogonal), and a key of $d_k$ numbers has room
for at most $d_k$ of them; after that, writes interfere. §1 says this of linear
Transformers: "the number of orthogonal key-value pairs they can store is bounded
by the model’s dimensionality. When the sequence length exceeds this dimension,
“memory collisions“ become inevitable, hindering exact retrieval" ($d_k$ per head
is my reading of "the model’s dimensionality").

Equation (10) is the per-token update of that memory matrix: shrink the whole
memory by a factor $\alpha_t$ between 0 and 1, then move what is stored under the current key a fraction $\beta_t$ of the way toward the
new value; the layer then reads the memory with a query vector. The paper builds it
from two updates printed in §2: Mamba2 (§2.1,
$\mathbf S_t=\alpha_t\mathbf S_{t-1}+\mathbf v_t\mathbf k_t^\intercal$) shrinks
everything but only ever adds, while DeltaNet (§2.2,
$\mathbf S_t=\mathbf S_{t-1}(\mathbf I-\beta_t\mathbf k_t\mathbf k_t^\intercal)+\beta_t\mathbf v_t\mathbf k_t^\intercal$,
with $\mathbf I$ the identity matrix) moves the entry under one key toward the
new value, all the way only at $\beta_t=1$, but never shrinks the rest; the
paper calls this update the _delta rule_. For example, if a
key of length 1 holds the value 2 and a token writes 5 under the same key, the additive
rule of linear attention reads back 7, while the delta rule with $\beta_t=1$ reads
back 5 (check, part 1). Equation (10)
multiplies DeltaNet's transition matrix $\mathbf I-\beta_t\mathbf k_t\mathbf k_t^\intercal$,
the matrix that multiplies the old memory, by Mamba2's $\alpha_t$, so what is stored
(one $d_v\times d_k$ matrix per head) stays the same and only the way it is
updated changes.

§3.3 computes the same states as equation (10) in blocks of $C$ tokens with
matrix multiplications so that training is fast on GPUs, and the same outputs
once one mask in its output formula is replaced (step C7): the paper prints a
plain 0/1 mask that lets each token see itself and the chunk's earlier tokens,
where the recurrence needs a mask that also shrinks each earlier token's write
by the $\alpha$'s of the tokens after it, up to the current one. §3.4
says how the layer makes $\mathbf q,\mathbf k,\mathbf v,\alpha,\beta$ from its
input and what it does to the output.

## The equation

$$
\mathbf S_t=\mathbf S_{t-1}\left(\alpha_t(\mathbf I-\beta_t\mathbf k_t\mathbf k_t^\intercal)\right)+\beta_t\mathbf v_t\mathbf k_t^\intercal
\qquad (10)
$$

The read-out, printed in §2.1 for linear attention and Mamba2 and used by the §3.3
output formula below: $\mathbf o_t=\mathbf S_t\mathbf q_t$.

The chunkwise training form, §3.3 (unnumbered), as printed:

$$
\widetilde{\mathbf U_{[t]}}=\left[\mathbf I+\operatorname{strictLower}\left(\operatorname{diag}(\beta_{[t]})(\Gamma_{[t]}\odot\mathbf K_{[t]}\mathbf K_{[t]}^\intercal)\right)\right]^{-1}\operatorname{diag}(\beta_{[t]})\mathbf V_{[t]}\in\mathbb R^{C\times d_v}
$$

$$
\mathbf S_{[t+1]}=\overrightarrow{\mathbf S_{[t]}}+\left(\widetilde{\mathbf U_{[t]}}-\overleftarrow{\mathbf W_{[t]}}\mathbf S_{[t]}^\intercal\right)^\intercal\overrightarrow{\mathbf K_{[t]}}\in\mathbb R^{d_v\times d_k}
$$

$$
\mathbf O_{[t]}=\overleftarrow{\mathbf Q_{[t]}}\mathbf S_{[t]}^\intercal+(\mathbf Q_{[t]}\mathbf K_{[t]}^\intercal\odot\mathbf M)\left(\widetilde{\mathbf U_{[t]}}-\overleftarrow{\mathbf W_{[t]}}\mathbf S_{[t]}^\intercal\right)\in\mathbb R^{C\times d_v}
$$

"where $\overleftarrow{\mathbf q_{[t]}^r}=\gamma_{[t]}^r\mathbf q_{[t]}^r$,
$\overleftarrow{\mathbf w_{[t]}^r}=\gamma_{[t]}^r\mathbf w_{[t]}^r$,
$\overrightarrow{\mathbf k_{[t]}^r}=\frac{\gamma_{[t]}^C}{\gamma_{[t]}^r}\mathbf k_{[t]}^r$,
and $\overrightarrow{\mathbf S_{[t]}}=\gamma_{[t]}^C\mathbf S_{[t]}$ like we defined
in Eq. 2." The $\mathbf O_{[t]}$ line has the wrong mask; see step C7 below.

## Symbols

How to read the notation. $\mathbb R^{a\times b}$ is a matrix with $a$ rows and
$b$ columns, a tensor of shape `[a, b]`. The paper does not say whether its
vectors are columns or rows (equation (2) calls $\mathbf q$ and $\mathbf k$
vectors and $\mathbf S$ "the state matrix"), but its shapes settle it:
$\mathbf S_t\in\mathbb R^{d_v\times d_k}$ times $\mathbf q_t$ gives
$\mathbf o_t\in\mathbb R^{d_v}$, and $\mathbf v_t\mathbf k_t^\intercal$ is
$d_v\times d_k$, so $\mathbf q_t$,
$\mathbf k_t$, $\mathbf v_t$ are column vectors and bold capitals are matrices.
The superscript $\intercal$ is a transpose. In the chunk notation
a superscript such as the $r$ in $\mathbf q^r_{[t]}$ or $\gamma^r_{[t]}$ is a
position label, not a power, and the subscript $[t]$ numbers the chunk.

| Symbol | Shape or type | Meaning | Role | Defined at |
| ------ | ------------- | ------- | ---- | ---------- |
| $\mathbf S_t$ | $d_v\times d_k$ matrix, one per head | the memory after token $t$ | computed intermediate, carried from token to token (not a trained weight) | "$\mathbf S_t=\mathbf S_{t-1}+\mathbf v_t\mathbf k_t^\intercal\in\mathbb R^{d_v\times d_k}$" (§2.1, linear attention; equation (10) keeps the shape, as §3.3's "$\in\mathbb R^{d_v\times d_k}$" shows); "the hidden state S can be interpreted as a (fast) weight matrix" (§3.1) |
| $\mathbf S_{t-1}$ | $d_v\times d_k$ | the memory before token $t$ | computed intermediate | as above; the starting value $\mathbf S_0$ is not defined (see below) |
| $t$ | integer $1,\dots,L$ | token position | index | "decays all key-value associations at each time step" (§1); "$L$ is the sequence length" (§2.1) |
| $\alpha_t$ | one number in $(0,1)$ per token; per head in the code (`lit_gpt/gated_delta_net.py` L92) | how much of the whole memory survives this token | computed intermediate, made from the token by a trained linear projection | "where the data-dependent gating term $\alpha_t\in(0,1)$ controls state decay" (§3.1, after equation (10)); "$\alpha,\beta$ use linear projection only" (§3.4) |
| $\beta_t$ | one number in $(0,1)$ | how far the entry under $\mathbf k_t$ moves toward $\mathbf v_t$ | computed intermediate from a trained linear projection | the delta rule "writes a new value ($\mathbf v_t^{\text{new}}$), which is a linear combination of the current input value and the old value based on the “writing strength” $\beta_t\in(0,1)$" (§2.2) |
| $\mathbf k_t$ | column of $d_k$ numbers, length 1 | key: where the token writes | computed intermediate from the token | "queries, keys and values $\{\mathbf q,\mathbf k,\mathbf v\}$ are generated through linear projection, short convolution and SiLU, with L2 normalization applied to $\mathbf q,\mathbf k$ for training stability" (§3.4) |
| $\mathbf v_t$ | column of $d_v$ numbers | value written under $\mathbf k_t$ | computed intermediate | the same §3.4 sentence |
| $\mathbf q_t$ | column of $d_k$ numbers, length 1 | query: where the token reads | computed intermediate | the same §3.4 sentence |
| $\mathbf o_t$ | column of $d_v$ numbers | what the head reads out at token $t$ | computed intermediate | "$\mathbf o_t=\mathbf S_t\mathbf q_t\in\mathbb R^{d_v}$" (§2.1) |
| $\mathbf I$ | $d_k\times d_k$ in equation (10), equation (3) and $\mathbf F^r$; $C\times C$ in equation (6) and $\widetilde{\mathbf U_{[t]}}$ | identity matrix: ones on the diagonal, zeros elsewhere | fixed | "DeltaNet’s data-dependent identity-plus-low-rank structure ($\mathbf I-\beta_t\mathbf k_t\mathbf k_t^\intercal$) offers greater flexibility than Mamba2’s data-dependent diagonal matrices ($\alpha_t\mathbf I$)" (§5); the $C\times C$ use from equation (6), "$\mathbf T_{[t]}=\left[\mathbf I+\dots\right]^{-1}\operatorname{diag}(\beta_{[t]})\in\mathbb R^{C\times C}$" (§2.2) |
| $d_k$, $d_v$ | integers | per-head sizes of keys/queries and of values | fixed setting | "where $d_k$ and $d_v$ represent the (head) dimensions for query/key and value, respectively" (§2.1) |
| $C$ | integer | tokens per chunk | fixed setting | "the chunkwise parallel form splits inputs and outputs into several chunks of size $C$" (§2.1) |
| $[t]$ (subscript), $r$ (superscript) | integers | chunk number; position $1,\dots,C$ inside the chunk. In §3.3, $t$ counts chunks, not tokens | index | "We denote $\mathbf Q_{[t]}:=\mathbf q_{tC+1:(t+1)C+1}$ as the query block for chunk $t$, and $\mathbf q_{[t]}^r:=\mathbf q_{tC+r}$ as the $r$-th query within chunk $t$." (§2.1) |
| $\mathbf Q_{[t]},\mathbf K_{[t]},\mathbf V_{[t]}$; $\beta_{[t]}$ | $C\times d_k$, $C\times d_k$, $C\times d_v$, one token per row; $C$ numbers | the chunk's queries, keys, values and writing strengths | computed intermediate | the quote above, which "take[s] the query block, $\mathbf q$, as an example" for all of them (§2.1) |
| $\mathbf O_{[t]}$ | $C\times d_v$, one token per row | the chunk's outputs: row $r$ is $\mathbf o^r_{[t]}=\mathbf S^r_{[t]}\mathbf q^r_{[t]}$ | computed intermediate, the result of the §3.3 output equation with $\Gamma_{[t]}$ in place of $\mathbf M$ (step C7) | §2.1, chunkwise linear attention: "$\mathbf o_{[t]}^r=\mathbf S_{[t]}^r\mathbf q_{[t]}^r=\dots\in\mathbb R^{d_v}$" and "$\mathbf O_{[t]}=\mathbf Q_{[t]}\mathbf S_{[t]}^\intercal+\left(\mathbf Q_{[t]}\mathbf K_{[t]}^\intercal\odot\mathbf M\right)\mathbf V_{[t]}\in\mathbb R^{C\times d_v}$", with the block notation of the $[t]$ row; the whole-sequence version is "$\mathbf O=(\mathbf Q\mathbf K^\intercal\odot\mathbf M)\mathbf V\in\mathbb R^{L\times d_v}$" (§2.1) |
| $\mathbf S_{[t]}$, $\mathbf S^r_{[t]}$ | $d_v\times d_k$ | memory at the start of chunk $t$; after its $r$-th token | computed intermediate | "The initial state of chunk $t$ is defined as $\mathbf S_{[t]}:=\mathbf S_{[t]}^0=\mathbf S_{[t-1]}^C$" (§2.1) |
| $\gamma^r_{[t]}$ | number in $(0,1)$ | product of the chunk's $\alpha$'s at positions $1,\dots,r$: total shrinking from chunk start through position $r$ | computed intermediate | "$(\Gamma_{[t]})_{ij}=\frac{\gamma_{[t]}^i}{\gamma_{[t]}^j},\ \gamma_{[t]}^j=\prod_{j=tC+1}^{tC+j}\alpha_j$" (§2.1, after equation (1)), and footnote 1: "Here we slightly abuse the notation of $\gamma$ to denote the cumulative product for each chunk (starting with the first position of each chunk separately) instead of the entire sequence." |
| $\Gamma_{[t]}$ | $C\times C$ | entry $(i,j)$: how much position $j$'s write has shrunk by position $i$ | computed intermediate | the equation (1) quote above. Zeros above the diagonal are stated only for the whole-sequence version: "$\Gamma\in\mathbb R^{L\times L}$ is a decay-aware causal mask where $\Gamma_{ij}=\frac{\gamma_i}{\gamma_j}$ if $i\ge j$ and $\Gamma_{ij}=0$ otherwise" (§2.1) |
| $\mathbf M$ | $C\times C$ of 0s and 1s | lets position $i$ see only positions $j\le i$ | fixed | "where $\mathbf M\in\mathbb R^{C\times C}$ is the causal mask" and, for the $L\times L$ version, "$\mathbf M_{ij}=0$ when $i<j$, and 1 otherwise" (§2.1) |
| $\overleftarrow{\cdot}$, $\overrightarrow{\cdot}$ | same shape as the variable | the variable rescaled to the chunk's first or last position | computed intermediate | "we use the left arrow ($\overleftarrow{\cdot}$) or the right arrow ($\overrightarrow{\cdot}$) to denote a variable decaying to the first position and the last position of each chunk, respectively" (§2.1, equation (2)); the factors are in the §3.3 "where" clause quoted under the equation |
| $\mathbf P^r_{[t]},\mathbf H^r_{[t]}$; $\mathbf F^r_{[t]},\mathbf G^r_{[t]}$ | $d_k\times d_k$, $d_v\times d_k$ | the multiplier on the chunk-start memory and the sum of the chunk's own writes, without gates ($\mathbf P,\mathbf H$) and with gates ($\mathbf F,\mathbf G$): $\mathbf S^r_{[t]}=\mathbf S_{[t]}\mathbf F^r_{[t]}+\mathbf G^r_{[t]}$ | computed intermediate | the underbraces "$:=\mathbf P_{[t]}^r$", "$:=\mathbf H_{[t]}^r$" (equation (3)) and "$:=\mathbf F_{[t]}^r$", "$:=\mathbf G_{[t]}^r$" (§3.3) |
| $\mathbf w^r_{[t]}$, $\mathbf T_{[t]}$, $\mathbf W_{[t]}$ | $d_k$; $C\times C$; $C\times d_k$ | DeltaNet's compact form of $\mathbf P$: $\mathbf P^r_{[t]}=\mathbf I-\sum_{i\le r}\mathbf w^i_{[t]}\mathbf k^{i\intercal}_{[t]}$; no gates | computed intermediate | equation (4), equation (6), and "$\mathbf W_{[t]}=\mathbf T_{[t]}\mathbf K_{[t]}\in\mathbb R^{C\times d_k}$" (equation (7)) |
| $\tilde{\mathbf u}^r_{[t]}$, $\widetilde{\mathbf U_{[t]}}$ | $d_v$; $C\times d_v$ | the gated version of DeltaNet's $\mathbf u$ (equation (5)): $\mathbf G^r_{[t]}=\sum_{i\le r}\frac{\gamma^r_{[t]}}{\gamma^i_{[t]}}\tilde{\mathbf u}^i_{[t]}\mathbf k^{i\intercal}_{[t]}$ | computed intermediate | the §3.3 display that precedes "(see §A for a proof)", and the matrix form printed above |

**Not defined in the paper** (I searched the whole paper: §1–§6, footnotes 1–6,
every figure and table caption, Appendix A and Appendix B):

- $\mathbf S_0$, the memory before the first token. Appendix A, which "only
  consider[s] the first chunk", writes $\mathbf S_t$ as a sum of writes with no
  starting term, which implies zero. The code's choice: "What the code must
  decide".
- How $\alpha_t$ is computed: footnote 4 says "We use Mamba2’s parameterization for
  $\alpha$ but omit it for brevity." Step B4 gives the code's formula.
- How $\beta_t$ is kept in $(0,1)$; §3.4 says only "linear projection". Step B5.
- SiLU, "L2 normalization", "short convolution", the output "normalization" and
  "gating" (§3.4, Figure 1): named, never defined. Steps B2, B3 and B7 give what
  the code does.
- $\odot$, $\operatorname{diag}(\cdot)$, $\operatorname{strictLower}(\cdot)$. The
  code multiplies entry by entry, builds a diagonal matrix from a vector (as a
  row scaling, `wy_fast.py` L57, L79), and keeps only entries with row index
  greater than column index (`wy_fast.py` L60, L130).
- The factor $1/\sqrt{d_k}$ by which the code multiplies $\mathbf q$ ("What the
  code must decide"). The paper never mentions it.
- The value of $C$ (the code's: "What the code must decide").
- The order of the factors in the products $\prod$ of equation (3) and §3.3. The
  matrices do not commute, so it matters; expanding equation (10) puts $i=1$ on the
  left (my derivation; the check below confirms it).
- The block end $(t+1)C+1$ in $\mathbf Q_{[t]}:=\mathbf q_{tC+1:(t+1)C+1}$. Read
  with both ends included, it holds $C+1$ tokens. I read it as positions $tC+1$ to
  $(t+1)C$, with $t$ counting from 0, because the paper says the chunks have "size
  $C$". The check uses that reading.

## Read from the inside out

**A. Equation (10), one head, one token, column vectors.**

1. $\mathbf k_t\mathbf k_t^\intercal$: a $d_k\times1$ column times a $1\times d_k$
   row gives a $d_k\times d_k$ matrix with entries $k_ik_j$ (an _outer product_).
   Applied to any key $\mathbf x$,
   $(\mathbf k_t\mathbf k_t^\intercal)\mathbf x=\mathbf k_t(\mathbf k_t^\intercal\mathbf x)$
   is the part of $\mathbf x$ that points along $\mathbf k_t$, because §3.4 makes
   $\mathbf k_t$ length 1.
2. $\mathbf I-\beta_t\mathbf k_t\mathbf k_t^\intercal$: $d_k\times d_k$. It leaves
   the part of a key perpendicular to $\mathbf k_t$ unchanged and multiplies the
   part along $\mathbf k_t$ by $1-\beta_t$. §2.2 calls matrices of this form
   "generalized Householder transition matrices".
3. $\alpha_t(\cdot)$: every entry times the same number. Since $\alpha_t$ is one
   number, it shrinks every direction equally, which is multiplying by
   $\alpha_t\mathbf I$, the form §5 gives Mamba2's transition (quoted in the
   $\mathbf I$ row).
4. $\mathbf S_{t-1}(\cdot)$: $(d_v\times d_k)(d_k\times d_k)\to d_v\times d_k$.
   Multiplying on the right acts on the key side: reading the new first term at a
   key $\mathbf x$ is reading the old memory at
   $\alpha_t(\mathbf x-\beta_t\mathbf k_t(\mathbf k_t^\intercal\mathbf x))$.
5. $\beta_t\mathbf v_t\mathbf k_t^\intercal$: $(d_v\times1)(1\times d_k)\to d_v\times d_k$,
   one outer product. Read at $\mathbf k_t$ it returns $\beta_t\mathbf v_t$; read at
   a key perpendicular to $\mathbf k_t$ it returns zero.
6. $+$ adds two $d_v\times d_k$ matrices entry by entry. The $=$ computes
   $\mathbf S_t$ from $\mathbf S_{t-1}$; the rule runs once per token, in order
   $t=1,\dots,L$.
7. $\mathbf o_t=\mathbf S_t\mathbf q_t$: $(d_v\times d_k)(d_k)\to d_v$, read after
   the update at $t$.

Put together (my derivation, checked in parts 1 and 2 of the check):

- At the current key: $\mathbf S_t\mathbf k_t=\alpha_t(1-\beta_t)\mathbf S_{t-1}\mathbf k_t+\beta_t\mathbf v_t$.
  The stored value becomes a blend: $\beta_t$ of the new value and
  $\alpha_t(1-\beta_t)$ of the old one.
- At any key $\mathbf x$ perpendicular to $\mathbf k_t$:
  $\mathbf S_t\mathbf x=\alpha_t\mathbf S_{t-1}\mathbf x$. Everything else only
  shrinks.
- The same update in two steps: $\mathbf S'=\alpha_t\mathbf S_{t-1}$, then
  $\mathbf S_t=\mathbf S'-\beta_t(\mathbf S'\mathbf k_t-\mathbf v_t)\mathbf k_t^\intercal$.
  The correction $\mathbf v_t-\mathbf S'\mathbf k_t=\mathbf v_t-\alpha_t\mathbf S_{t-1}\mathbf k_t$,
  the new value minus what the decayed memory returns at $\mathbf k_t$, is the
  difference ("delta") the delta rule is named for (my reading: the paper
  credits the rule to Widrow et al. (1960) and never says what "delta" refers
  to; I searched every "delta" in its text). The second step is one step of
  gradient descent with step size $\beta_t$ on the squared error
  $\frac12\Vert\mathbf S\mathbf k_t-\mathbf v_t\Vert^2$ between what the memory
  returns at $\mathbf k_t$ and $\mathbf v_t$; its gradient with respect
  to $\mathbf S$ is $(\mathbf S\mathbf k_t-\mathbf v_t)\mathbf k_t^\intercal$.
  That gradient is the matrix whose entry $(i,j)$ is the rate at which the error
  changes when only the entry $S_{ij}$ of $\mathbf S$ moves. With $k_j$ and $v_i$ the entries of
  $\mathbf k_t$ and $\mathbf v_t$, entry $i$ of $\mathbf S\mathbf k_t$ is
  $\sum_jS_{ij}k_j$, so moving $S_{ij}$ by $h$ moves only that entry, by $hk_j$.
  Since $\frac12(x-v_i)^2$ changes at the rate $x-v_i$ as $x$ moves, the error
  $\frac12\sum_i((\mathbf S\mathbf k_t)_i-v_i)^2$ changes at the rate
  $((\mathbf S\mathbf k_t)_i-v_i)\,k_j$: entry $(i,j)$ of
  $(\mathbf S\mathbf k_t-\mathbf v_t)\mathbf k_t^\intercal$. The
  paper states this for DeltaNet, "the delta rule optimizing the online regression
  objective $\mathcal L(\mathbf S_t)=\frac12\Vert\mathbf S_t\mathbf k_t-\mathbf v_t\Vert^2$
  via test-time stochastic gradient descent (SGD)" (§3.1), and says "the gated delta
  rule can be viewed as incorporating an adaptive weight decay term $\alpha_t$ into
  the SGD update" (§3.1). The two-step form is my exact version of that sentence.
- The two limits: $\alpha_t\to1$ gives DeltaNet, and $\alpha_t\to0$ leaves only
  $\beta_t\mathbf v_t\mathbf k_t^\intercal$, as §1 says: "it can promptly clear
  memory by setting $\alpha_t\to0$, while selectively updating specific content
  without affecting other information by setting $\alpha_t\to1$ (effectively
  switching to the pure delta rule)."

In PyTorch the official code stores the transpose of $\mathbf S$, one
$d_k\times d_v$ matrix per sequence and head
(`S = torch.zeros(b, h, d_k, d_v).to(v)`, `chunk.py` L699), and passes
$g=\log\alpha$. Its reference loop
(`chunk.py` L705–710; the comments are mine):

```python
S = S.clone() * g[:, :, i].exp()[..., None, None]     # S' = alpha_t * S_{t-1}
beta_i = beta[:, :, i]                               # beta_t
_v = _v - (S.clone() * _k[..., None]).sum(-2)        # v_t - S' k_t
_v = _v * beta_i[..., None]                          # beta_t * (v_t - S' k_t)
S = S.clone() + _k.unsqueeze(-1) * _v.unsqueeze(-2)  # add the outer product
o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q, S)    # o_t = S_t q_t, with q already scaled by d_k**-0.5 (L700)
```

**B. Where the inputs come from and where the output goes (§3.4, Figure 1, code).**
The layer takes one vector $\mathbf x_t$ of size $d_{model}$ per token, the block
input after a normalization (`lit_gpt/model.py` L285, L307, L310). Line numbers are in
`lit_gpt/gated_delta_net.py`.

1. Linear projections, for $H$ heads: $\mathbf q$ and $\mathbf k$ map
   $d_{model}\to Hd_k$ and $\mathbf v$ maps $d_{model}\to Hd_v$, all without
   bias; $\alpha$ and $\beta$ each get one number per head (L83–93). Each is
   PyTorch's `nn.Linear`, which computes $xA^\intercal+b$ for a row $x$ with a
   trained matrix $A$; `bias=False` drops $b$
   ([PyTorch 2.3 `nn.Linear`](https://docs.pytorch.org/docs/2.3/generated/torch.nn.Linear.html)).
2. For $\mathbf q$, $\mathbf k$, $\mathbf v$: mix each channel with the same
   channel at the current and the three previous tokens through a learned filter
   of length 4 (the paper's "short convolution"), then apply
   $x\cdot\mathrm{sigmoid}(x)$ to every entry, where
   $\mathrm{sigmoid}(x)=1/(1+e^{-x})$ lies in $(0,1)$; this is a smooth version of
   $\max(0,x)$ called SiLU. The layer builds fla's `ShortConvolution` with kernel
   size 4 and `activation='silu'` (L43, L89–91, L150–152). That class is an
   `nn.Conv1d` with `groups=hidden_size`, which gives each channel its own
   filter, and no bias by default (`fla/modules/convolution.py` L79–99;
   [PyTorch 2.3 `nn.Conv1d`](https://docs.pytorch.org/docs/2.3/generated/torch.nn.Conv1d.html)).
   By default it calls `causal_conv1d_fn` from causal-conv1d (`convolution.py`
   L18, L90, L168–174), whose CUDA kernel I did not read. That package's reference version,
   `causal_conv1d_ref`, pads 3 positions on both sides of the sequence
   (`padding=width - 1`), keeps the first $L$ outputs and applies SiLU
   (`causal_conv1d/causal_conv1d_interface.py` L133, L158, L162, L171;
   [PyTorch 2.3 `F.conv1d`](https://docs.pytorch.org/docs/2.3/generated/torch.nn.functional.conv1d.html),
   [`F.silu`](https://docs.pytorch.org/docs/2.3/generated/torch.nn.functional.silu.html)),
   so output $t$ uses tokens $t-3,\dots,t$ only (my reading of these lines). The
   package's test runs `causal_conv1d_fn` and `causal_conv1d_ref` on the same
   random inputs, for widths 2, 3 and 4, with and without bias and SiLU, and
   asserts that their outputs agree within a small tolerance
   (`tests/test_causal_conv1d.py` L25–29, L42–72, L82): the fast path is tested
   to compute the same window.
3. Split into heads, then divide each head's $\mathbf q_t$ and $\mathbf k_t$ by
   the square root of the sum of its $d_k$ squared entries plus $10^{-6}$, so that
   it has length 1 up to that $10^{-6}$: this is the paper's "L2 normalization"
   (L29, L166–175). The split is einops' `rearrange`; in a group of axes such as
   `(h d)` the last name varies fastest (einops `v0.8.1`,
   `docs/1-einops-basics.ipynb`, "Order of axes matters"), so each head gets
   $d_k$ adjacent entries. fla's `l2_norm` normalizes each slice along the last
   axis, here one head's $d_k$ entries, with that $10^{-6}$
   (`fla/modules/l2norm.py` L178–181, L201): it reshapes the input into rows of
   $d_k$ entries (L102), normalizes each row (L38–43) and reshapes the result
   back (L133). A PyTorch `reshape` keeps the elements in order, last index
   fastest: the example in
   [PyTorch 2.3 `torch.reshape`](https://docs.pytorch.org/docs/2.3/generated/torch.reshape.html)
   turns `torch.arange(4.)`, the numbers $0,1,2,3$, into the rows $(0,1)$ and
   $(2,3)$, and [`Tensor.reshape`](https://docs.pytorch.org/docs/2.3/generated/torch.Tensor.reshape.html)
   refers to that page. So each row is one head of one token. $\mathbf v_t$ is not normalized.
4. $\alpha_t$: $\log\alpha_t=-\exp(A_{\log})\cdot\mathrm{softplus}(z_t+b_{dt})$,
   where $z_t$ is the head's linear projection, $A_{\log}$ and $b_{dt}$ are
   trained per-head numbers, and $\mathrm{softplus}(x)=\log(1+e^x)>0$ (PyTorch's
   `F.softplus` returns $x$ itself above 20, also positive;
   [PyTorch 2.3 `F.softplus`](https://docs.pytorch.org/docs/2.3/generated/torch.nn.functional.softplus.html)),
   so $\alpha_t\in(0,1)$ (L157–161). Larger $z_t$ means more forgetting.
5. $\beta_t=\mathrm{sigmoid}(z'_t)$, from a linear projection with bias (L93, L164;
   `.sigmoid()` is $1/(1+e^{-x})$:
   [PyTorch 2.3 `torch.sigmoid`](https://docs.pytorch.org/docs/2.3/generated/torch.sigmoid.html),
   an alias of [`torch.special.expit`](https://docs.pytorch.org/docs/2.3/special.html)).
6. Equation (10) and the read-out for every head, computed by the chunked kernel
   of part C (L190).
7. "the output is processed through normalization and gating before applying
   output projection" (§3.4), and Figure 1: "output gate applies linear proj. with
   SiLU". In the code each head's $\mathbf o_t$ is divided by the square root of
   the mean of its $d_v$ squared entries plus $10^{-5}$ (RMSNorm), multiplied
   entry by entry by a learned scale of $d_v$ numbers that starts at 1 and is
   shared by the heads, then multiplied entry by entry by SiLU of another
   projection of $\mathbf x_t$ ($d_{model}\to Hd_v$), split into heads the same
   way; the heads are concatenated in order, by the einops rule of step B3,
   and a last linear map takes $Hd_v\to d_{model}$ (L45–49, L87, L95–102,
   L203–213). On the
   default path fla's `FusedRMSNormSwishGate` (`fla/modules/fused_norm_gate.py`
   L765–803) does the normalization, scale and gate: it calls
   `rms_norm_swish_gate_fn`, which passes `True` as `is_rms_norm` to
   `LayerNormSwishGateFn` (L439–459, L647–667). `LayerNormSwishGateFn`
   reshapes $\mathbf o$ and the gate input into rows of $d_v$ entries, one head of one token
   per row (L458–459), and the result back (L479), in the order shown for
   `reshape` in step B3; the kernel then works on each row (L119–121,
   L129–130, L136).

In the code's 0.4B configuration, $d_{model}=1536$ (`lit_gpt/config.py`
L104–111) and $H=9$, the layer's default (`gated_delta_net.py` L40), not the
configuration's `n_head=12` (`config.py` L110), which `model.py` L288 does not
pass; so $d_k=128$ and $d_v=256$, and each head's memory holds 32,768 numbers
whatever the sequence length (check, part 4).

**C. The chunked algorithm (§3.3).** Running equation (10) token by token is a loop
of $L$ dependent small steps; §2.1: "However, both the recurrent and parallel forms
are not ideal for efficient training". Chunking leaves only the $L/C$ chunk
boundaries sequential. Inside a chunk, everything is a product of $C\times C$,
$C\times d_k$ and $C\times d_v$ matrices, which "are rich in matrix multiplications
(matmuls), allowing for tensor-core-based hardware optimization" (§2.1). For one
chunk, with one token per row:

1. $\gamma^r=\alpha^1\cdots\alpha^r$ (the chunk's own $\alpha$'s, restarting at
   each chunk): $C$ numbers. $(\Gamma_{[t]})_{ij}=\gamma^i/\gamma^j$ for $i\ge j$
   and 0 above the diagonal, which is the product of $\alpha$ over positions
   $j+1,\dots,i$.
2. Expanding equation (10) over the first $r$ tokens of the chunk gives
   $\mathbf S^r_{[t]}=\mathbf S_{[t]}\mathbf F^r_{[t]}+\mathbf G^r_{[t]}$, with
   $\mathbf F^r_{[t]}=\prod_{i=1}^r\alpha^i_{[t]}(\mathbf I-\beta^i_{[t]}\mathbf k^i_{[t]}\mathbf k^{i\intercal}_{[t]})$.
   The $\alpha$'s are numbers, so they factor out: "It is easy to see that
   $\mathbf F_{[t]}^r=\gamma_{[t]}^r\mathbf P_{[t]}^r$" (§3.3).
3. $\mathbf T_{[t]}$ (equation (6)), $C\times C$, and
   $\mathbf W_{[t]}=\mathbf T_{[t]}\mathbf K_{[t]}$ (equation (7)), $C\times d_k$,
   are DeltaNet's, with no $\alpha$ in them. Their rows $\mathbf w^r$ satisfy
   equation (4) and give
   $\mathbf P^r=\mathbf I-\sum_{i\le r}\mathbf w^i\mathbf k^{i\intercal}$. The
   matrix inside the brackets has ones on the diagonal and zeros above it, so it
   always has an inverse, and the code solves it one row at a time
   (`wy_fast.py` L65–71) rather than calling a general inverse.
4. $\widetilde{\mathbf U_{[t]}}$: the same pattern with
   $\Gamma_{[t]}\odot\mathbf K_{[t]}\mathbf K_{[t]}^\intercal$, where $\odot$
   multiplies entry by entry; $C\times d_v$. Row $r$ is
   $\tilde{\mathbf u}^r=\beta^r\big(\mathbf v^r-\sum_{i<r}\frac{\gamma^r}{\gamma^i}(\mathbf k^{i\intercal}\mathbf k^r)\tilde{\mathbf u}^i\big)$:
   if the chunk had started from an empty memory, this is what token $r$ adds:
   $\beta^r$ times the gap between $\mathbf v^r$ and what the chunk's earlier
   writes, shrunk to position $r$, return at $\mathbf k^r$ (my reading of the
   recursion). Appendix A proves
   $\mathbf G^r=\sum_{i\le r}\frac{\gamma^r}{\gamma^i}\tilde{\mathbf u}^i\mathbf k^{i\intercal}$
   by induction, for the first chunk.
5. The arrows rescale rows: $\overleftarrow{\mathbf Q}$ and
   $\overleftarrow{\mathbf W}$ multiply row $r$ by $\gamma^r$,
   $\overrightarrow{\mathbf K}$ multiplies row $r$ by $\gamma^C/\gamma^r$, and
   $\overrightarrow{\mathbf S}=\gamma^C\mathbf S_{[t]}$.
6. $\widetilde{\mathbf U}-\overleftarrow{\mathbf W}\mathbf S_{[t]}^\intercal$:
   $(C\times d_k)(d_k\times d_v)\to C\times d_v$. Row $r$ is
   $\tilde{\mathbf u}^r-\gamma^r\mathbf S_{[t]}\mathbf w^r$, the empty-memory write
   corrected for what the chunk-start memory, shrunk to position $r$, returns at
   $\mathbf w^r$ (equation (4), which already includes $\beta^r$ and the effect
   of the chunk's earlier keys). The
   code computes this at `chunk.py` L156–159 and stores it as `v_new` (L152,
   L161).
7. Output:
   $\mathbf O_{[t]}=\overleftarrow{\mathbf Q}\mathbf S_{[t]}^\intercal+(\mathbf Q\mathbf K^\intercal\odot\text{mask})(\widetilde{\mathbf U}-\overleftarrow{\mathbf W}\mathbf S_{[t]}^\intercal)$,
   two $C\times d_v$ terms from $(C\times d_k)(d_k\times d_v)$ and
   $(C\times C)(C\times d_v)$; row $r$ is the chunk's $\mathbf o^r$. The paper
   prints the mask as $\mathbf M$, but it must be $\Gamma_{[t]}$ (my derivation).
   Steps 2–4 give
   $\mathbf S^r=\gamma^r\mathbf S_{[t]}(\mathbf I-\sum_{i\le r}\mathbf w^i\mathbf k^{i\intercal})+\sum_{i\le r}\frac{\gamma^r}{\gamma^i}\tilde{\mathbf u}^i\mathbf k^{i\intercal}$.
   Multiply by $\mathbf q^r$ and write
   $\gamma^r\mathbf S_{[t]}\mathbf w^i=\frac{\gamma^r}{\gamma^i}(\gamma^i\mathbf S_{[t]}\mathbf w^i)$:
   $\mathbf o^r=\gamma^r\mathbf S_{[t]}\mathbf q^r+\sum_{i\le r}\frac{\gamma^r}{\gamma^i}(\mathbf k^{i\intercal}\mathbf q^r)(\tilde{\mathbf u}^i-\gamma^i\mathbf S_{[t]}\mathbf w^i)$.
   The first term is row $r$ of $\overleftarrow{\mathbf Q}\mathbf S_{[t]}^\intercal$,
   the bracket is row $i$ of step 6, and its weight is
   $(\mathbf Q\mathbf K^\intercal)_{ri}(\Gamma_{[t]})_{ri}$. By step 1,
   $(\Gamma_{[t]})_{ri}$ for $i<r$ is the product of the $\alpha$'s at positions
   $i+1,\dots,r$, which never includes the chunk's first $\alpha$. So
   $\Gamma_{[t]}=\mathbf M$ on and below the diagonal exactly when every $\alpha$
   after the first position of the chunk is 1; then the printed line is right, and
   with every $\alpha=1$ it is DeltaNet's equation (9). When some $\alpha$ after the
   first position is below 1, the weights differ from the ones the recurrence
   needs, and for general inputs the printed line gives a wrong output. Part 3 of
   the check shows all three cases, and the official kernel multiplies
   $\mathbf Q\mathbf K^\intercal$ by $\gamma^i/\gamma^j$ before masking
   (`chunk.py` L227–232; the same in the reference at L660–661, L680).
8. Memory for the next chunk:
   $\mathbf S_{[t+1]}=\overrightarrow{\mathbf S}+(\widetilde{\mathbf U}-\overleftarrow{\mathbf W}\mathbf S_{[t]}^\intercal)^\intercal\overrightarrow{\mathbf K}$:
   $(d_v\times C)(C\times d_k)\to d_v\times d_k$. It equals the token-by-token
   memory at the end of every chunk (check, part 3).

**Kind of statement:** equation (10) is an _update rule_. At each token
$\mathbf S$ changes; the trained weights that produce
$\mathbf q,\mathbf k,\mathbf v,\alpha,\beta$ stay fixed during the pass. The §3.3
equations are a _computation_: an exact rearrangement of equation (10), not an
approximation, once $\Gamma_{[t]}$ replaces the printed $\mathbf M$ (step C7), so
they must give the same $\mathbf S$ and $\mathbf o$ up to floating-point rounding. §3.4 describes the layer's architecture.

## Where the samples come from

Equation (10), §3.3 and §3.4 contain no expectation and no sampling. The memory
$\mathbf S$ is filled from the tokens of the sequence being processed, in training
and in generation alike; for its starting value see $\mathbf S_0$ under "Not
defined in the paper". The weights that make $\mathbf q,\mathbf k,\mathbf v,\alpha,\beta$
and the output are trained by language modeling ("all models are trained under
identical conditions with 1.3B parameters on 100B tokens sampled from the
FineWeb-Edu dataset", §4), with gradients through the whole recurrence
(the `backward` method of `ChunkGatedDeltaRuleFunction` in `chunk.py`, L547 and
from L580), except into a
memory passed in from an earlier segment, which is detached ("What the code must
decide").

## Check with small numbers

Part 1 uses two perpendicular unit keys ($d_k=2$, $d_v=3$) and five tokens that
exercise each term of equation (10): a write to an empty slot, a partial overwrite
of a filled slot (so the $\alpha_t(1-\beta_t)$ term is not zero), $\alpha=1$ with
$\beta=1$ (exact overwrite), and $\alpha=0$ (full clear); it then makes one write
under the additive rule and under the delta rule. Part 2 checks the
identities of part A on one random step and the _eigenvalues_ of the transition
matrix; an eigenvalue is a factor by which a matrix stretches some direction.
Part 3 runs $L=8$ tokens in two chunks of
$C=4$ from a nonzero starting memory, with random $\alpha$, with $\alpha<1$ only
at each chunk's first position, and with every $\alpha=1$; both entries of
$\mathbf M$ (0 and 1) and both cases of $\operatorname{strictLower}$ (kept and
zeroed) occur. Part 4 checks the code's formulas for $\alpha$, $\beta$ and L2
normalization on inputs with exact answers. Part 5 recomputes the throughput gaps
of Figure 3 from its plotted coordinates. Run with
`uv run --no-project --with numpy python check.py`.

```python
"""Small-number checks for Gated DeltaNet (arXiv 2412.06464v3), Eq. (10), §3.3, §3.4.

Paper convention: column vectors, state S has shape (d_v, d_k), readout o = S q.
Chunk blocks Q, K, V have one token per row.
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def step(S, k, v, alpha, beta):
    """Eq. (10): S_t = S_{t-1} (alpha_t (I - beta_t k_t k_t^T)) + beta_t v_t k_t^T."""
    I = np.eye(k.shape[0])
    return S @ (alpha * (I - beta * np.outer(k, k))) + beta * np.outer(v, k)


# ---- 1. Hand example: d_k = 2 (two orthogonal unit keys), d_v = 3 -------------
e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
S = np.zeros((3, 2))                                   # S_0 = 0
S = step(S, e1, np.array([1.0, 2, 3]), alpha=0.5, beta=1.0)   # t=1
S = step(S, e2, np.array([4.0, 5, 6]), alpha=0.5, beta=0.5)   # t=2
print("t=2  S k1 =", S @ e1, "  S k2 =", S @ e2)
assert np.allclose(S @ e1, [0.5, 1.0, 1.5])      # pair 1 only decayed by alpha_2
assert np.allclose(S @ e2, [2.0, 2.5, 3.0])      # beta_2 * v_2 (nothing stored at k2 before)
S = step(S, e2, np.array([8.0, 8, 8]), alpha=0.8, beta=0.5)   # t=3, same key again
print("t=3  S k1 =", S @ e1, "  S k2 =", S @ e2)
assert np.allclose(S @ e1, [0.4, 0.8, 1.2])      # 0.8 * previous
assert np.allclose(S @ e2, [4.8, 5.0, 5.2])      # 0.8*(1-0.5)*old + 0.5*new
S = step(S, e1, np.array([7.0, 8, 9]), alpha=1.0, beta=1.0)   # t=4, alpha -> 1 limit
print("t=4  S k1 =", S @ e1, "  S k2 =", S @ e2)
assert np.allclose(S @ e1, [7, 8, 9])            # exact overwrite at k1
assert np.allclose(S @ e2, [4.8, 5.0, 5.2])      # k2's value untouched
S = step(S, e2, np.array([2.0, 2, 2]), alpha=0.0, beta=0.5)   # t=5, alpha -> 0 limit
print("t=5  S =\n", S)
assert np.allclose(S, 0.5 * np.outer([2, 2, 2], e2))  # everything older is gone
# the same write under the additive rule and under the delta rule: k1 holds 2, the token writes 5 under k1
S1 = np.array([[2.0, 0.0]])                                   # d_v = 1
additive = S1 + np.outer([5.0], e1)                           # linear attention: S_t = S_{t-1} + v_t k_t^T
delta = step(S1, e1, np.array([5.0]), alpha=1.0, beta=1.0)    # Eq. (10) with alpha = beta = 1
print("same write, read at k1: additive =", additive @ e1, "  delta rule =", delta @ e1)
assert np.allclose(additive @ e1, [7.0]) and np.allclose(delta @ e1, [5.0])

# ---- 2. Identities for one step, random values, d_k = 3, d_v = 2 --------------
rng = np.random.default_rng(0)
Sp = rng.normal(size=(2, 3))
k = rng.normal(size=3); k /= np.linalg.norm(k)     # unit key, as after L2 norm
kp = np.cross(k, rng.normal(size=3)); kp /= np.linalg.norm(kp)   # unit key with k^T kp = 0
v = rng.normal(size=2)
a, b = 0.7, 0.3
St = step(Sp, k, v, a, b)
# (a) read-back at k and at a perpendicular key
assert np.allclose(St @ k, a * (1 - b) * (Sp @ k) + b * v)
assert np.allclose(St @ kp, a * (Sp @ kp))
# (b) decay first, then one gradient step of size beta on 0.5*||S k - v||^2 at the decayed state
Sd = a * Sp
assert np.allclose(St, Sd - b * np.outer(Sd @ k - v, k))
sq = lambda S: 0.5 * np.sum((S @ k - v) ** 2)                 # the squared error at k
unit = lambda i, j: np.outer(np.eye(2)[i], np.eye(3)[j])      # 1 at entry (i, j), 0 elsewhere
fd = np.array([[(sq(Sd + 1e-6 * unit(i, j)) - sq(Sd - 1e-6 * unit(i, j))) / 2e-6
                for j in range(3)] for i in range(2)])
assert np.allclose(fd, np.outer(Sd @ k - v, k))               # entry (i, j) is (S k - v)_i k_j
# (c) Table 1 row "Gated DeltaNet": St minimizes ||S - a Sp||_F^2 - 2 <S k, b (v - a Sp k)>
f = lambda S: np.sum((S - a * Sp) ** 2) - 2 * (S @ k) @ (b * (v - a * Sp @ k))
h = 1e-6
grad = np.zeros_like(St)
for i in range(2):
    for j in range(3):
        E = np.zeros_like(St); E[i, j] = h
        grad[i, j] = (f(St + E) - f(St - E)) / (2 * h)
print("finite-difference gradient of the Table 1 objective at S_t:", np.abs(grad).max())
assert np.abs(grad).max() < 1e-6
# (d) eigenvalues of the transition alpha (I - beta k k^T)
ev_unit = np.sort(np.linalg.eigvalsh(a * (np.eye(3) - b * np.outer(k, k))))
k3 = 3 * k                                          # same direction, length 3 (no L2 norm)
ev_long = np.sort(np.linalg.eigvalsh(1.0 * (np.eye(3) - 0.5 * np.outer(k3, k3))))
print("eigenvalues, unit k, alpha=0.7, beta=0.3:", ev_unit)
print("eigenvalues, |k|=3, alpha=1,   beta=0.5:", ev_long)
assert np.allclose(ev_unit, [a * (1 - b), a, a])
assert np.allclose(ev_long, [1 - 0.5 * 9, 1, 1])
ev_gen = np.sort(np.linalg.eigvalsh(a * (np.eye(3) - 0.5 * np.outer(k3, k3))))   # alpha = 0.7
assert np.allclose(ev_gen, [a * (1 - 0.5 * 9), a, a])   # alpha (1 - beta |k|^2) along k, alpha elsewhere

# ---- 3. Chunked algorithm (§2.2 Eq. 3-7, §3.3) vs the token-by-token recurrence --------
L, C, dk, dv = 8, 4, 3, 2                 # two chunks of four tokens
q = rng.normal(size=(L, dk)); q /= np.linalg.norm(q, axis=1, keepdims=True)
K = rng.normal(size=(L, dk)); K /= np.linalg.norm(K, axis=1, keepdims=True)
V = rng.normal(size=(L, dv))
beta = rng.uniform(0.1, 0.9, size=L)
S_init = rng.normal(size=(dv, dk))        # nonzero start state, so the S_[t] terms matter
I_C = np.eye(C)
strictLower = lambda X: np.tril(X, -1)
M = np.tril(np.ones((C, C)))              # causal mask: 0 when i < j, 1 otherwise


def recurrent(alpha):
    """Eq. (10) token by token from S_init; o_t = S_t q_t (§2.1 readout)."""
    S, O, ends = S_init.copy(), [], []
    for t in range(L):
        S = step(S, K[t], V[t], alpha[t], beta[t])
        O.append(S @ q[t])
        if (t + 1) % C == 0:
            ends.append(S.copy())
    return np.array(O), ends


def chunked(alpha):
    """§3.3, one chunk at a time. Returns O with the printed mask M, O with Gamma, chunk-end states."""
    S, O_M, O_G, ends = S_init.copy(), [], [], []
    for c in range(L // C):
        sl = slice(c * C, (c + 1) * C)
        Qc, Kc, Vc, a_c, b_c = q[sl], K[sl], V[sl], alpha[sl], beta[sl]
        gam = np.cumprod(a_c)                                  # gamma^r_[t], restarts each chunk
        Gam = np.tril(gam[:, None] / gam[None, :])             # Gamma_[t], zero above diagonal
        if c == 0:   # per-position forms (Eq. 3-5, §3.3) for every r of the first chunk
            Sr = S.copy()
            for r in range(1, C + 1):
                Sr = step(Sr, Kc[r - 1], Vc[r - 1], a_c[r - 1], b_c[r - 1])
                P, F = np.eye(dk), np.eye(dk)                  # products, i = 1 leftmost
                for i in range(r):
                    H_i = np.eye(dk) - b_c[i] * np.outer(Kc[i], Kc[i])
                    P, F = P @ H_i, F @ (a_c[i] * H_i)
                assert np.allclose(F, gam[r - 1] * P)           # F^r = gamma^r P^r
                w = []                                          # Eq. 4 recursion
                for j in range(r):
                    w.append(b_c[j] * (Kc[j] - sum(w[i] * (Kc[i] @ Kc[j]) for i in range(j))))
                assert np.allclose(P, np.eye(dk) - sum(np.outer(w[i], Kc[i]) for i in range(r)))
                ut = []                                         # §3.3 recursion for u-tilde
                for j in range(r):
                    ut.append(b_c[j] * (Vc[j] - sum(ut[i] * (gam[j] / gam[i]) * (Kc[i] @ Kc[j])
                                                    for i in range(j))))
                G = sum((gam[r - 1] / gam[i]) * np.outer(ut[i], Kc[i]) for i in range(r))
                assert np.allclose(Sr, S @ F + G)               # S^r = S F^r + G^r
        T = np.linalg.inv(I_C + strictLower(np.diag(b_c) @ Kc @ Kc.T)) @ np.diag(b_c)   # Eq. 6
        W = T @ Kc                                                                       # Eq. 7
        Ut = np.linalg.inv(I_C + strictLower(np.diag(b_c) @ (Gam * (Kc @ Kc.T)))) @ np.diag(b_c) @ Vc
        if c == 0:
            assert np.allclose(W, np.array(w)) and np.allclose(Ut, np.array(ut))  # matrix = recursion
        Wl = gam[:, None] * W                    # <-W : row r times gamma^r
        Ql = gam[:, None] * Qc                   # <-Q
        Kr = (gam[-1] / gam)[:, None] * Kc       # ->K : row r times gamma^C / gamma^r
        Vnew = Ut - Wl @ S.T                     # C x d_v
        O_M.append(Ql @ S.T + ((Qc @ Kc.T) * M) @ Vnew)      # as printed in §3.3
        O_G.append(Ql @ S.T + ((Qc @ Kc.T) * Gam) @ Vnew)    # Gamma_[t] in place of M
        S = gam[-1] * S + Vnew.T @ Kr                         # ->S + (U~ - <-W S^T)^T ->K
        assert S.shape == (dv, dk)
        ends.append(S.copy())
    return np.vstack(O_M), np.vstack(O_G), ends


alpha_starts = np.ones(L); alpha_starts[[0, C]] = [0.5, 0.3]   # alpha < 1 only at each chunk's first position
for name, alpha, M_ok in [("random alpha in (0.3, 0.95)", rng.uniform(0.3, 0.95, size=L), False),
                          ("alpha < 1 only at chunk starts", alpha_starts, True),
                          ("every alpha = 1 (DeltaNet)", np.ones(L), True)]:
    O_rec, ends_rec = recurrent(alpha)
    O_M, O_G, ends = chunked(alpha)
    assert all(np.allclose(a_, b_) for a_, b_ in zip(ends, ends_rec))   # S_[t+1] matches
    print(f"{name}: max|O with Gamma - recurrence| = {np.abs(O_G - O_rec).max():.1e},"
          f" max|O with M - recurrence| = {np.abs(O_M - O_rec).max():.1e}")
    assert np.allclose(O_G, O_rec)
    assert np.allclose(O_M, O_rec) == M_ok

# ---- 4. Layer-level numbers from the official code (§3.4, footnote 4) -------------
softplus = lambda x: np.log1p(np.exp(x))
alpha_of = lambda z, A, dt_bias: np.exp(-A * softplus(z + dt_bias))   # g = log(alpha)
print("alpha(z=0, A=1) =", alpha_of(0.0, 1.0, 0.0),
      " alpha(z=ln 3, A=1) =", alpha_of(np.log(3), 1.0, 0.0),
      " alpha(z=0, A=2) =", alpha_of(0.0, 2.0, 0.0))
assert np.isclose(alpha_of(0.0, 1.0, 0.0), 0.5)
assert np.isclose(alpha_of(np.log(3), 1.0, 0.0), 0.25)
assert np.isclose(alpha_of(0.0, 2.0, 0.0), 0.25)
zs = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
al = alpha_of(zs, 1.0, 0.0)
print("alpha over z =", zs, "->", al)
assert np.all((al > 0) & (al < 1)) and np.all(np.diff(al) < 0)
dt = np.array([0.001, 0.1])                       # dt_bias init: dt + log(-expm1(-dt)), gated_delta_net.py L116-122
assert np.allclose(softplus(dt + np.log(-np.expm1(-dt))), dt)   # i.e. softplus(dt_bias) = dt
sigmoid = lambda x: 1 / (1 + np.exp(-x))
print("beta(z=0) =", sigmoid(0.0))
assert sigmoid(0.0) == 0.5
k_raw = np.array([3.0, 4.0])
print("L2-normalized (3, 4) =", k_raw / np.linalg.norm(k_raw))
assert np.allclose(k_raw / np.linalg.norm(k_raw), [0.6, 0.8])
# head sizes for GatedDeltaNet_0.4B at commit 3154b25: n_embd=1536 (lit_gpt/config.py L111);
# expand_k=0.75, expand_v=1.5, 9 heads (layer defaults, lit_gpt/gated_delta_net.py L38-40)
n_embd, heads = 1536, 9
d_k, d_v = int(n_embd * 0.75) // heads, int(n_embd * 1.5) // heads
print("0.4B config: d_k =", d_k, " d_v =", d_v, " state entries per head =", d_k * d_v,
      " q scale 1/sqrt(d_k) =", round(d_k ** -0.5, 4))
assert (d_k, d_v) == (128, 256)

# ---- 5. Figure 3 training throughput (Kt/s), plotted coordinates in main.tex ------
settings = ["2Kx16", "4Kx8", "8Kx4", "16Kx2"]
gdn = np.array([46.1, 45.7, 45.8, 45.9])
deltanet = np.array([45.9, 45.6, 45.6, 45.2])
mamba2 = np.array([48.1, 48.2, 48.2, 48.1])
for name, x in [("Gated DeltaNet", gdn), ("DeltaNet", deltanet)]:
    gap = [round(float(g), 1) for g in mamba2 - x]
    print(f"Mamba2 - {name} (Kt/s) at {settings}: {gap}")
    assert all(2 <= g <= 3 for g in gap)                 # the paper's "2-3K tokens/sec"
print("Gated DeltaNet - DeltaNet (Kt/s):", [round(float(g), 1) for g in gdn - deltanet])
print("all checks passed")
```

Output:

```text
t=2  S k1 = [0.5 1.  1.5]   S k2 = [2.  2.5 3. ]
t=3  S k1 = [0.4 0.8 1.2]   S k2 = [4.8 5.  5.2]
t=4  S k1 = [7. 8. 9.]   S k2 = [4.8 5.  5.2]
t=5  S =
 [[0. 1.]
 [0. 1.]
 [0. 1.]]
same write, read at k1: additive = [7.]   delta rule = [5.]
finite-difference gradient of the Table 1 objective at S_t: 1.1102230246251565e-10
eigenvalues, unit k, alpha=0.7, beta=0.3: [0.49 0.7  0.7 ]
eigenvalues, |k|=3, alpha=1,   beta=0.5: [-3.5  1.   1. ]
random alpha in (0.3, 0.95): max|O with Gamma - recurrence| = 2.2e-16, max|O with M - recurrence| = 5.8e-01
alpha < 1 only at chunk starts: max|O with Gamma - recurrence| = 4.4e-16, max|O with M - recurrence| = 4.4e-16
every alpha = 1 (DeltaNet): max|O with Gamma - recurrence| = 4.4e-16, max|O with M - recurrence| = 4.4e-16
alpha(z=0, A=1) = 0.5  alpha(z=ln 3, A=1) = 0.24999999999999994  alpha(z=0, A=2) = 0.25
alpha over z = [-3. -1.  0.  1.  3.] -> [0.9526 0.7311 0.5    0.2689 0.0474]
beta(z=0) = 0.5
L2-normalized (3, 4) = [0.6 0.8]
0.4B config: d_k = 128  d_v = 256  state entries per head = 32768  q scale 1/sqrt(d_k) = 0.0884
Mamba2 - Gated DeltaNet (Kt/s) at ['2Kx16', '4Kx8', '8Kx4', '16Kx2']: [2.0, 2.5, 2.4, 2.2]
Mamba2 - DeltaNet (Kt/s) at ['2Kx16', '4Kx8', '8Kx4', '16Kx2']: [2.2, 2.6, 2.6, 2.9]
Gated DeltaNet - DeltaNet (Kt/s): [0.2, 0.1, 0.2, 0.7]
all checks passed
```

At $t=3$ the value under $\mathbf k_2$ becomes $0.8\cdot0.5\cdot(2,2.5,3)+0.5\cdot(8,8,8)=(4.8,5,5.2)$
while the value under $\mathbf k_1$ only shrinks by 0.8. At $t=4$ ($\alpha=1$,
$\beta=1$) the value under $\mathbf k_1$ is replaced exactly and the one under
$\mathbf k_2$ is untouched; at $t=5$ ($\alpha=0$) only the new write remains.
In part 3 the chunked state matches the recurrence at every chunk end, and the
two output masks behave as step C7 says: the printed $\mathbf M$ is off by 0.58
with random $\alpha$ and agrees to rounding when $\alpha<1$ only at chunk starts.

## What the code must decide

All lines are at commit `3154b25`; `gated_delta_net.py` and `model.py` are in
`lit_gpt/`, `chunk.py` and `wy_fast.py` in `lit_gpt/gated_delta_rule_ops/`.

- **Parameterization of $\alpha$:** step B4. At
  initialization $\exp(A_{\log})$ is drawn uniformly between 0 and 16 and $b_{dt}$
  is set so that $\mathrm{softplus}(b_{dt})$ lies between 0.001 and 0.1 (L106–123;
  check, part 4;
  [PyTorch 2.3 `Tensor.uniform_`](https://docs.pytorch.org/docs/2.3/generated/torch.Tensor.uniform_.html),
  [`torch.rand`](https://docs.pytorch.org/docs/2.3/generated/torch.rand.html),
  which draws from $[0,1)$, and
  [`torch.special.expm1`](https://docs.pytorch.org/docs/2.3/special.html),
  $e^x-1$). Both are marked `_no_weight_decay`.
- **Range of $\beta$:** $(0,1)$ by the sigmoid of step B5, not the $(0,2)$ that
  footnote 2 mentions as an option: "It is possible to set $\beta_t\in(0,2)$ to
  allow negative eigenvalue …".
- **Normalization axis for $\mathbf q,\mathbf k$:** step B3. The code also
  offers, in place of L2, `softmax` and `longhorn`; the second leaves
  $\mathbf q,\mathbf k$ unnormalized and instead divides $\beta_t$ by
  $1+\beta_t\Vert\mathbf k_t\Vert^2$ (L172–182). The default is `l2` (L42).
- **Scale of $\mathbf q$:** multiplied by $d_k^{-1/2}$ inside the kernel
  (`chunk.py` L219, L515), 0.0884 for $d_k=128$ (check, part 4).
- **Short convolution:** step B2. `conv_bias: bool = False,`
  (L44) is stored (L70) but never passed to `ShortConvolution` (L89–91), so that
  class's default of no bias applies.
- **Head sizes:** $d_k=0.75\,d_{model}/H$ and $d_v=1.5\,d_{model}/H$ (L72–73, L80–81),
  where 0.75, 1.5 and $H$ are the layer's defaults (L38–40); `model.py` L288
  passes only `hidden_size`, so the defaults apply. The 0.4B values: step B, last
  paragraph.
- **Memory layout and precision:** transposed, as in part A (`chunk.py` L699;
  `gated_delta_net.py` L224). The kernel
  accumulates it in float32 (`chunk.py` L134) but converts it for the matrix
  products ("state will convert to bf16 to do matmul anyway so we don't need fp32
  state in the forward pass.", L565).
- **Starting memory and detaching:** zeros unless a state is passed in
  (`chunk.py` L134–138); a memory passed in is detached (`chunk.py` L634;
  [PyTorch 2.3 `Tensor.detach`](https://docs.pytorch.org/docs/2.3/generated/torch.Tensor.detach.html):
  "The result will never require gradient.").
- **Chunk size and padding:** $C=64$ (`chunk.py` L622). A sequence whose length is
  not a multiple of 64 is padded at the end with zeros (L628–629: `F.pad` with
  4 numbers pads the last two axes, and its third and fourth numbers pad the
  token axis before and after; the fill value is 0 by default,
  [PyTorch 2.3 `F.pad`](https://docs.pytorch.org/docs/2.3/generated/torch.nn.functional.pad.html)),
  so padded tokens have
  $g=0$ ($\alpha=1$) and $\beta=0$ and, by equation (10), leave $\mathbf S$
  unchanged; the outputs are cut back to length $L$ (L636).
- **Cumulative products:** computed as running sums of $g$ that restart at each
  chunk, as in footnote 1, converted to base 2 for `exp2` (`chunk.py` L556–559:
  einops' `rearrange` cuts the token axis into chunks of 64 adjacent tokens, by
  the order rule in step B3, and
  [PyTorch 2.3 `torch.cumsum`](https://docs.pytorch.org/docs/2.3/generated/torch.cumsum.html)
  sums within each chunk).
  In the forward kernels each ratio $\gamma^i/\gamma^j$ is computed as
  `tl.math.exp2(b_g[:, None] - b_g[None, :])` (`chunk.py` L227–230, `wy_fast.py`
  L128–129), and each
  $\gamma^C/\gamma^r$ as `tl.math.exp2(b_g_last - b_g)` (`chunk.py` L144, L151,
  L162), so the
  code never divides one small product by another.
- **Mask in the in-chunk output:** $\Gamma_{[t]}$, not the printed $\mathbf M$
  (step C7).
- **How $\overleftarrow{\mathbf W}$ is formed:** from the gate-free
  $\mathbf T_{[t]}$ of equation (6) (`wy_fast.py` L54–80), multiplied by
  $\gamma^r$ in the state kernel (`chunk.py` L151, L156), as §3.3 writes it.
  $\widetilde{\mathbf U}$ uses the $\Gamma$-weighted matrix (`wy_fast.py`
  L128–142).
- **Output normalization and gate:** step B7. The unfused branch in the same
  file (L211–212) has the same order as the fused kernel of step B7: normalize
  first, then gate.
- **Memory traffic in training:** the forward pass discards the per-chunk states
  and the backward pass recomputes them (`chunk.py` L571–572, L586–587).
- **Off by default:** a Mamba2-style skip term that adds $D\,\mathbf v_t$ to each
  head's output, with one trained number $D$ per head (`use_residual`, L53,
  L126–128, L201–202), and an input gate that multiplies $\mathbf v_t$ by
  $1-\alpha_t$ (`use_input_gate`, L54, L183–185).

## What the paper claims and what it shows

The paper reports no number of runs, seeds or error bars for any table (I searched
the PDF text for "seed", "±" and "runs"), so every compared result below is a
single number per model.

- **"This formulation unifies the advantages of both gating mechanisms and the delta
  rule: the gating term enables adaptive memory management, while the delta update
  structure facilitates effective key-value association learning."** (§3.1).
  Argued from the form of equation (10); the support offered is the experiments
  listed below.
- **The two limits** (§1, quoted in part A): they follow from equation (10); check,
  part 1.
- **Online-learning view** (Table 1): among all matrices $\mathbf S_t$, the one
  given by equation (10) makes
  $\Vert\mathbf S_t-\alpha_t\mathbf S_{t-1}\Vert_F^2-2\langle\mathbf S_t\mathbf k_t,\beta_t(\mathbf v_t-\alpha_t\mathbf S_{t-1}\mathbf k_t)\rangle$
  smallest, and the paper writes it as a formula (a _closed-form_ solution).
  Here $\Vert\cdot\Vert_F^2$ is the sum of squared entries and
  $\langle\cdot,\cdot\rangle$ a dot product: the first term keeps
  $\mathbf S_t$ near the shrunk old memory, the second rewards returning the
  corrected value at $\mathbf k_t$.
  Entry by entry, as in part A: moving only $(\mathbf S_t)_{ij}$ changes the
  first term at the rate $2(\mathbf S_t-\alpha_t\mathbf S_{t-1})_{ij}$, and,
  writing $\mathbf c=\beta_t(\mathbf v_t-\alpha_t\mathbf S_{t-1}\mathbf k_t)$,
  the second term $-2\sum_ic_i\sum_j(\mathbf S_t)_{ij}k_j$ at the rate
  $-2c_ik_j$, entry $(i,j)$ of $-2\mathbf c\mathbf k_t^\intercal$.
  Setting the gradient with respect to $\mathbf S_t$,
  $2(\mathbf S_t-\alpha_t\mathbf S_{t-1})-2\beta_t(\mathbf v_t-\alpha_t\mathbf S_{t-1}\mathbf k_t)\mathbf k_t^\intercal$,
  to zero gives
  $\mathbf S_t=\alpha_t\mathbf S_{t-1}+\beta_t(\mathbf v_t-\alpha_t\mathbf S_{t-1}\mathbf k_t)\mathbf k_t^\intercal$,
  which rearranges to equation (10) (check, part 2b). The objective is the sum of
  squared entries of $\mathbf S_t$ plus terms linear in $\mathbf S_t$, a bowl-shaped
  (convex) quadratic, so this point is its only minimum (my derivation). The paper places
  the result in the framework of Liu et al. (2024), which I did not open; the
  check (part 2c) finds the numerical gradient at equation (10)'s $\mathbf S_t$
  to be $1.1\times10^{-10}$. That the update rule of Longhorn, the linear RNN of
  Liu et al. (2024) whose framework Table 1 uses, "closely
  resembles the delta update rule" is offered as "suggesting the superiority of
  the (gated) delta rule over Mamba2 in in-context associative recall" (§3.1): a
  suggestion, not a measurement.
- **The chunked algorithm** (§3.3): a derivation. Appendix A proves the
  $\mathbf G^r$ formula by induction for the first chunk only. Its second line has
  $\mathbf k_i\mathbf k_{t+1}^\intercal$ where the step needs
  $\mathbf k_{t+1}\mathbf k_{t+1}^\intercal$; the next line uses the right term, and
  the formula holds numerically (check, part 3). This typo and the printed
  $\mathbf O_{[t]}$ mask (step C7) are my findings.
- **Speed:** "the proposed gated delta rule introduces only marginal overhead
  compared to the original delta rule, with Gated DeltaNet achieving essentially
  the same throughput as DeltaNet. Both are slightly slower than Mamba2 (2-3K
  tokens/sec) due to their more expressive transition matrices." (§4). Measured
  as training throughput in tokens per second: "Figure 3: Training throughput
  comparison of 1.3B models on a single H100 GPU", at sequence length × batch
  size from 2K×16 to 16K×2. From Figure 3's plotted coordinates, Mamba2 leads
  Gated DeltaNet by 2.0–2.5K and DeltaNet by 2.2–2.9K tokens/s at every
  setting, and Gated DeltaNet is 0.1–0.7K tokens/s above DeltaNet (check,
  part 5). No inference or decoding speed is measured. The
  cause ("more expressive transition matrices") is the authors' explanation; no
  experiment isolates it.
- **L2 normalization "for training stability"** (§3.4): the reason is stated but
  not tested; no experiment in the paper measures stability. Table S.1 (400M
  parameters, 15B tokens) measures average perplexity, a score of how well the
  model predicts text, where lower is better (the table's "Avg-PPL (↓)"): 27.35, 27.67 and 27.58 for
  the L2 rows (27.35 is the full block, which §3.4 builds with L2 and SiLU; the
  table has no separate "L2-norm & SiLU" row) against 30.79, 30.34 and 30.18 for
  the L1 rows, and Appendix B.2
  concludes "we found L2 normalization to be essential for optimal performance,
  though the choice of feature map was less influential". Here is a possible
  mechanism; it is mine, not the paper's. The transition
  $\alpha_t(\mathbf I-\beta_t\mathbf k_t\mathbf k_t^\intercal)$ has eigenvalue
  $\alpha_t(1-\beta_t\Vert\mathbf k_t\Vert^2)$ along $\mathbf k_t$ and $\alpha_t$
  in every perpendicular direction (check, part 2d). With a unit key these are
  $\alpha_t(1-\beta_t)$ and $\alpha_t$, all in $(0,1)$, so multiplying by the
  transition can only shrink the memory. With $\alpha=1$, a key of length 3 and
  $\beta=0.5$, the eigenvalue along the key is $-3.5$ (check, part 2d): each write
  along that key multiplies what the memory returns at it by $-3.5$ before adding
  the new value, so repeated writes along it can make the memory grow.
- **Block components** (Table S.1, same setting): "both the short convolution and
  output gate are crucial for model performance, while output normalization yields
  marginal improvements". Average perplexity 28.95 without short convolution,
  29.12 without the output gate and 27.55 without output normalization, against
  27.35 for the full block. The row "w. naive Delta Rule" (30.87) is not explained
  anywhere in the paper.
- **Head size:** "a head dimension of 128 provides an optimal trade-off between
  performance and computational efficiency" (Appendix B.2). Table S.1 gives
  perplexities 28.31, 27.35 and 27.13 for head dimensions 64, 128 and 256; it
  does not measure the computational cost.
- **S-NIAH case study** (Table 2, 1.3B models): measured. S-NIAH is the "Single
  Needle-In-A-Haystack (S-NIAH) benchmark suite from RULER (Hsieh et al., 2024),
  where a key-value pair acts as a needle in the haystack (context) and the model
  must recall the value when given the key" (§3.2). Table 2 calls its numbers
  "Zero-shot performance" and does not name the metric (I searched for
  "accuracy" and "metric"); higher is better, since §3.2 calls DeltaNet's
  S-NIAH-1 row "near-perfect performance". For example, on S-NIAH-1
  at 8K tokens DeltaNet scores 98.8, Mamba2 30.4 and Gated DeltaNet 91.8; on
  S-NIAH-2 at 4K tokens, 18.6, 56.2 and 92.2. The explanations, such as "Mamba2
  degrades significantly beyond 2K sequences since it decays historical
  information too quickly, while Gated DeltaNet’s degradation is less severe thanks
  to the use of delta rule" (§3.2), interpret the table; no ablation tests them.
  §3.1 also hedges that selective forgetting "could be useful in filtering out
  irrelevant information".
- **Language modeling** (Table 3; the text says it covers "models with 400M and
  1.3B parameters", but the table shows one row per model with no size label):
  Gated DeltaNet's average accuracy is 55.32, against 54.89 for Mamba2 and 52.14 for DeltaNet, and
  its Wikitext perplexity is 16.42, against 16.56 and 17.71.
- **Hybrid models** (§3.4): GatedDeltaNet-H1 combines Gated DeltaNet layers with
  attention over a sliding window of recent tokens (SWA; §4 uses "a sliding
  window size of 2K"), and GatedDeltaNet-H2 stacks Mamba2, Gated DeltaNet and
  SWA. The motivation is cited, not tested in §3.4: "Linear transformers have
  limitations in modeling local shifts and comparisons, and their fixed state
  size makes it hard for retrieval tasks (Arora et al., 2024a)".

## Sources

- Yang, Kautz, Hatamizadeh, Gated Delta Networks: Improving Mamba2 with Delta Rule,
  [arXiv 2412.06464v3](https://arxiv.org/abs/2412.06464v3): TeX source
  (`main.tex`, `figures/model.tex`) and PDF of v3; rendered images of the PDF
  from §2.2's paragraph "As shown above, DeltaNet implements …" to §4's "In-context retrieval on
  real-world data" paragraph: equations (3)–(10), the §3.3 displays, Tables 1–4
  and Figure 1.
- Official code, NVlabs/GatedDeltaNet at commit
  [`3154b25f2361eab4831999fb8deb1641d1a41a16`](https://github.com/NVlabs/GatedDeltaNet/tree/3154b25f2361eab4831999fb8deb1641d1a41a16):
  `lit_gpt/gated_delta_net.py`, `lit_gpt/gated_delta_rule_ops/chunk.py`,
  `lit_gpt/gated_delta_rule_ops/wy_fast.py`,
  `lit_gpt/gated_delta_rule_ops/__init__.py`, `lit_gpt/model.py`,
  `lit_gpt/config.py`, `scripts/tsz512x4k_15B_gated_deltanet_h1_0.4B.sh`,
  `README.md`, `Dockerfile`; the GitHub API's tag and commit listings for the
  repository, used to choose the commit.
- flash-linear-attention at commit
  [`326184c45f42380063fb10bdda31d394b0c8c1b7`](https://github.com/fla-org/flash-linear-attention/tree/326184c45f42380063fb10bdda31d394b0c8c1b7):
  `fla/modules/l2norm.py`, `fla/modules/fused_norm_gate.py`,
  `fla/modules/convolution.py`, `setup.py`, and `fla/modules/__init__.py`,
  `fla/modules/activations.py`, `fla/modules/layernorm.py`,
  `fla/ops/simple_gla/__init__.py`, `fla/ops/simple_gla/chunk.py`,
  `fla/utils.py` and `fla/models/utils.py` to confirm that the names the
  official code imports exist there. To choose the commit: its tag and commit
  listings (its first tag, `v0.1.0`, is dated 20 Mar 2025), the listing of
  commits that change `fla/modules/convolution.py`, and that file at
  [`46706afd3a77a06966ed77518efdfce012e96cc6`](https://github.com/fla-org/flash-linear-attention/blob/46706afd3a77a06966ed77518efdfce012e96cc6/fla/modules/convolution.py),
  [`a1b1719624e8e6278a280502fd794dbce30a9663`](https://github.com/fla-org/flash-linear-attention/blob/a1b1719624e8e6278a280502fd794dbce30a9663/fla/modules/convolution.py)
  and
  [`695d7eb7fe4e6b918e4f9170fcb35ad8bb8d534d`](https://github.com/fla-org/flash-linear-attention/blob/695d7eb7fe4e6b918e4f9170fcb35ad8bb8d534d/fla/modules/convolution.py).
  The commit chosen by date does not run the official code: from commit
  `46706afd` (22 Nov 2024) to `695d7eb` (23 Feb 2025, the last fla commit
  before `3154b25`), the `forward` method of `ShortConvolution`, the short
  convolution of step B2, returns a pair, the output and a cache
  (`fla/modules/convolution.py` at `695d7eb`, L79, L142, L149, L192), while
  the layer passes that result straight to einops' `rearrange`, the head split of step B3 (`lit_gpt/gated_delta_net.py`
  L150, L166), which takes a tensor or a list of tensors and raises an error
  for a tuple (`einops/einops.py` L521–527, L600; `einops/_backends.py` L59,
  L224–225). `326184c` is the parent of `46706afd`: there `forward` returns one
  tensor (`convolution.py` L142, L147, L179), and every name the official code
  imports from fla exists.
- causal-conv1d at tag
  [`v1.4.0`](https://github.com/Dao-AILab/causal-conv1d/tree/v1.4.0):
  `causal_conv1d/__init__.py`, `causal_conv1d/causal_conv1d_interface.py`,
  `tests/test_causal_conv1d.py`.
- einops at tag [`v0.8.1`](https://github.com/arogozhnikov/einops/tree/v0.8.1):
  `docs/1-einops-basics.ipynb`, `einops/einops.py`, `einops/_backends.py`;
  PyPI's release dates, used to choose the tag.
- PyTorch 2.3 documentation:
  [`nn.Linear`](https://docs.pytorch.org/docs/2.3/generated/torch.nn.Linear.html),
  [`nn.Conv1d`](https://docs.pytorch.org/docs/2.3/generated/torch.nn.Conv1d.html),
  [`F.conv1d`](https://docs.pytorch.org/docs/2.3/generated/torch.nn.functional.conv1d.html),
  [`F.silu`](https://docs.pytorch.org/docs/2.3/generated/torch.nn.functional.silu.html),
  [`F.softplus`](https://docs.pytorch.org/docs/2.3/generated/torch.nn.functional.softplus.html),
  [`nn.Softplus`](https://docs.pytorch.org/docs/2.3/generated/torch.nn.Softplus.html),
  [`nn.SiLU`](https://docs.pytorch.org/docs/2.3/generated/torch.nn.SiLU.html),
  [`torch.sigmoid`](https://docs.pytorch.org/docs/2.3/generated/torch.sigmoid.html),
  [`Tensor.sigmoid`](https://docs.pytorch.org/docs/2.3/generated/torch.Tensor.sigmoid.html),
  [`torch.expm1`](https://docs.pytorch.org/docs/2.3/generated/torch.expm1.html),
  [`torch.special`](https://docs.pytorch.org/docs/2.3/special.html),
  [`Tensor.uniform_`](https://docs.pytorch.org/docs/2.3/generated/torch.Tensor.uniform_.html),
  [`torch.rand`](https://docs.pytorch.org/docs/2.3/generated/torch.rand.html),
  [`F.pad`](https://docs.pytorch.org/docs/2.3/generated/torch.nn.functional.pad.html),
  [`Tensor.detach`](https://docs.pytorch.org/docs/2.3/generated/torch.Tensor.detach.html),
  [`torch.cumsum`](https://docs.pytorch.org/docs/2.3/generated/torch.cumsum.html),
  [`torch.reshape`](https://docs.pytorch.org/docs/2.3/generated/torch.reshape.html),
  [`Tensor.reshape`](https://docs.pytorch.org/docs/2.3/generated/torch.Tensor.reshape.html).
- Named, not opened: Widrow et al. (1960), the delta rule; Dao & Gu (2024), Mamba2; Yang et al. (2024b), the DeltaNet
  chunkwise algorithm; Liu et al. (2024), Longhorn; Sun et al. (2023a), RetNet;
  Bischof & Van Loan (1985), the WY representation; Joffrain et al. (2006), the UT
  transform; the CUDA kernel behind causal-conv1d's `causal_conv1d_fn`;
  Arora et al. (2024a); Hsieh et al. (2024), RULER.
