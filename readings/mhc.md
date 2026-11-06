# DeepSeek-V4 mHC: four residual streams per token, mixed by a matrix pushed toward rows and columns that each sum to 1

**Source:** DeepSeek-AI,
[DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348v1),
v1 (submitted 2026-04-26), §2.2 (pp. 7–8), equations (1), (2) and (8), with
equations (3)–(7) of the same subsection for the symbols they use. Other
locations in that paper: §2.3 and its footnote 1 (the link to the code), §2.4
"Basic Configurations" and Algorithm 1, §3.3, §3.4.2, §4.2.1. The method comes
from Xie et al.,
[mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880v2),
v2: §1, §3, §3.1, §4.1, §4.2, §4.3.1, §5.2, §5.3, §5.4 and Appendix A.1 (Table 5).
Code: `inference/model.py`, `inference/kernel.py`, `inference/config.json` and
`inference/requirements.txt` in the repository that
V4's footnote 1 links, at commit `a1fd202632e91bc0efe4eedb63ce554649c53997`
(2026-04-27). Of the repository's commits, this one is closest to the v1
submission date.

## What it computes and what it changes

A Transformer normally carries one vector of $d$ numbers per token from layer
to layer and adds each sublayer's output to it, $\mathbf x_{l+1}=\mathbf x_l+\mathcal F(\mathbf x_l)$
(mHC paper, eq. (1)). That pass-through path is the _residual connection_. The
vector it carries is the _residual stream_. Equation (1) of DeepSeek-V4 gives
Hyper-Connections (HC). HC carries $n_{\text{hc}}=4$ such vectors per token,
which I call _streams_. Three small arrays control the streams. They are not
trained weights: trained weights compute them afresh for every token from that
token's streams (in V4, by equations (3)–(7) below). A row $A_l$ blends the
four streams into one sublayer input. A column $C_l$ copies the sublayer's
output back onto the four streams, with a different weight for each. A
$4\times4$ matrix $B_l$ mixes the four streams on the pass-through path. With
one stream ($n_{\text{hc}}=1$) and $A_l=B_l=C_l=1$, equation (1) is the ordinary residual
$\mathbf x_{l+1}=\mathbf x_l+\mathcal F_l(\mathbf x_l)$ (my substitution).

mHC keeps equation (1) and changes what $B_l$ may be. In plain HC, $B_l$ is any
matrix. Equation (2) allows only matrices with nonnegative entries in which
every row and every column sums to 1. With one stream that leaves only $B_l=1$:
the condition "degenerates to the scalar 1" (mHC §4.1). Equation (8) is the
loop that turns a raw matrix $\tilde B_l$, computed for each token by equation
(4), into an approximation of such a matrix. The mHC paper says the loop
converges to such a matrix as the number of rounds goes to infinity, and V4
runs 20 (step 5 under "Equation (8)" in "Read from the inside out"). mHC also
passes $A_l$ and $C_l$ through a sigmoid (equations (6)–(7)).

V4 says that plain HC training is numerically unstable (quoted under
"What the paper claims and what it shows") and that mHC is meant to "enhance
the stability of signal propagation across layers" (§2.2). The mHC paper's
reason for the instability (mHC §1) is that the product of many unconstrained
$B_l$ across layers can make the signal
grow or shrink without limit. A comparison of mHC eq. (5) with mHC eq. (7) (my
comparison) shows that the raw values are also computed differently from the
original HC. Original HC applies a tanh to a
projection of each stream. mHC uses a linear projection of the whole flattened
state. DeepSeek-V4 eqs. (3)–(5) are the mHC version.

## The equation

$$
X_{l+1} = B_{l} X_l + C_{l} \mathcal{F}_{l}(A_{l} X_l), \tag{1}
$$

$$
B_l \in \mathcal{M} \coloneqq \{ M \in \mathbb{R}^{n \times n} \mid M\mathbf{1}_n = \mathbf{1}_n, \; \mathbf{1}_n^T M = \mathbf{1}_n^T, \; M \geqslant 0 \}. \tag{2}
$$

$$
M^{(t)} = \mathcal{T}_r(\mathcal{T}_c(M^{(t-1)})), \tag{8}
$$

The text around (8) gives the start and the end of the loop:
$M^{(0)} = \exp(\tilde{B}_l)$ and $B_l = M^{(t_{\text{max}})}$, with
$t_{\text{max}} = 20$.

The raw values that (1) and (8) use come from equations (3)–(7):

$$
\begin{aligned}
\tilde{A}_l &= \alpha_l^\mathrm{pre} \cdot (\hat{X}_l W^\mathrm{pre}_l) + S_l^\mathrm{pre}, & (3)\\
\tilde{B}_l &= \alpha_l^\mathrm{res} \cdot \operatorname{Mat}(\hat{X}_l W^\mathrm{res}_l) + S_l^\mathrm{res}, & (4)\\
\tilde{C}_l &= \alpha_l^\mathrm{post} \cdot (\hat{X}_l W^\mathrm{post}_l)^T + S_l^\mathrm{post}, & (5)\\
A_l &= \sigma(\tilde{A}_l), & (6)\\
C_l &= 2\sigma(\tilde{C}_l). & (7)
\end{aligned}
$$

## Symbols

All locations are in DeepSeek-V4 unless marked "mHC".

| Symbol | Shape or type | Meaning | Role | Defined at |
| ------ | ------------- | ------- | ---- | ---------- |
| $X_l$ | $n_{\text{hc}}\times d$, one per token | The residual state entering layer $l$. Each row is one stream. | Computed intermediate (the hidden state passed from layer to layer) | "Let $X_l = [\mathbf{x}_{l,1}; \ldots; \mathbf{x}_{l,n_{\text{hc}}}]^T \in \mathbb{R}^{n_{\text{hc}} \times d}$ be the residual state before the $l$-th layer." (§2.2) |
| $X_{l+1}$ | $n_{\text{hc}}\times d$ | The state after layer $l$, which is the input of layer $l+1$ | Computed intermediate | Same sentence, with $l+1$ in place of $l$ (§2.2) |
| $l$ | Integer | Which layer. Each Transformer block holds two such layers, attention and then MoE. | Index | "the residual state before the $l$-th layer" (§2.2). mHC, Fig. 3 caption: "The layer index $l$ (x-axis) unrolls each standard Transformer block into two independent layers (Attention and FFN)." |
| $n_{\text{hc}}$ | Integer, 4 | The number of streams | Fixed setting | "The standard HC expands the width of the residual stream by a factor of $n_{\text{hc}}$." (§2.2). "As for mHC, the expansion factor $n_{\text{hc}}$ is set to 4" (§4.2.1) |
| $d$ | Integer: 4096 (V4-Flash), 7168 (V4-Pro) | The size of a sublayer's input and output vectors | Fixed setting | "where $d$ is the hidden size of the actual layer input" (§2.2). "the hidden dimension $d$ to 4096" and "to 7168" (§4.2.1) |
| $\mathcal F_l$ | Function $\mathbb R^d\to\mathbb R^d$ | The sublayer itself: attention or MoE. In the code it starts with an RMSNorm that has its own learned weight (see "What $\mathcal F_l$ includes"). | Trained weight (a sublayer whose parameters are trained) | "where $\mathcal{F}_l$ denotes the $l$-th layer (e.g., an MoE layer), whose input and output shapes are both $\mathbb{R}^d$" (§2.2, after (1)) |
| $A_l$ | $1\times n_{\text{hc}}$ row, entries in $(0,1)$ | How much of each stream goes into the sublayer's input | Computed intermediate, recomputed for every token from trained weights by (3) and (6) | "an input mapping $A_l \in \mathbb{R}^{1 \times n_{\text{hc}}}$" (§2.2). Value: eq. (6) |
| $B_l$ | $n_{\text{hc}}\times n_{\text{hc}}$ | How the streams are mixed on the pass-through path | Computed intermediate for every token, by (4) and (8) | "a residual transformation $B_l \in \mathbb{R}^{n_{\text{hc}} \times n_{\text{hc}}}$" (§2.2). "This iteration converges to a constrained doubly stochastic matrix $B_l = M^{(t_{\text{max}})}$." (§2.2, after (8)) |
| $C_l$ | $n_{\text{hc}}\times 1$ column, entries in $(0,2)$ | How much of the sublayer's output is added to each stream | Computed intermediate, by (5) and (7) | "an output mapping $C_l \in \mathbb{R}^{n_{\text{hc}} \times 1}$" (§2.2). Value: eq. (7) |
| $\mathcal M$ | A set of $n\times n$ matrices | The allowed values of $B_l$. The paper calls them _doubly stochastic_ matrices: every entry is nonnegative, and every row and every column sums to 1. In a _stochastic_ matrix each row can be read as a probability distribution (nonnegative numbers that sum to 1); _doubly_ means the columns can too. Nothing is random. | Fixed setting (the constraint set, defined by (2)) | "the manifold of doubly stochastic matrices (the Birkhoff polytope) $\mathcal{M}$" (§2.2). mHC: "a doubly stochastic matrix, which has non-negative entries where both the rows and columns sum to 1" (mHC §4.1) |
| $M$ in (2) | $n\times n$ | A placeholder inside the set definition, like the loop variable in `{M for M in ... if ...}`. It is not the $M^{(t)}$ of (8). | Index (the variable of the set definition) | Eq. (2) itself |
| $n$ | Integer | The matrix size in (2). It must equal $n_{\text{hc}}$, or $B_l\in\mathcal M$ cannot hold (my inference from the shapes). | Fixed setting | V4: see "Not defined in the paper". mHC: "where $n$ is the expansion rate" (mHC §1, after its eq. (3)) |
| $\mathbf 1_n$ | $n\times 1$ column of ones | $M\mathbf 1_n$ gives the row sums of $M$, and $\mathbf 1_n^TM$ gives its column sums. It is not an indicator. | Fixed setting (a constant) | V4: see "Not defined in the paper". mHC: "where $\mathbf{1}_n$ represents the $n$-dimensional vector of all ones." (mHC §4.1, after its eq. (6)) |
| $M\geqslant 0$ | Entrywise comparison | Every entry is $\ge 0$. It does not mean positive semidefinite. | Fixed setting (part of the constraint) | V4: see "Not defined in the paper". mHC §4.1 says "non-negative entries" (quoted in the $\mathcal M$ row). |
| $\tilde A_l$, $\tilde B_l$, $\tilde C_l$ | $1\times n_{\text{hc}}$, $n_{\text{hc}}\times n_{\text{hc}}$, $n_{\text{hc}}\times1$; any real numbers | The raw values behind $A_l$ (before the sigmoid of (6)), $B_l$ (before the loop of (8)) and $C_l$ (before the sigmoid of (7)) | Computed intermediate, for every token, by (3)–(5) | "the unconstrained raw parameters $\tilde{A}_l \in \mathbb{R}^{1 \times n_{\text{hc}}}$, $\tilde{B}_l \in \mathbb{R}^{n_{\text{hc}} \times n_{\text{hc}}}$, and $\tilde{C}_l \in \mathbb{R}^{n_{\text{hc}} \times 1}$" (§2.2). Values: eqs. (3)–(5) |
| $\exp$ | Entrywise | $e^x$ of each entry. It is not the matrix exponential. | Fixed setting (an operation with no weights) | "which first applies an exponential function to $\tilde{B}_l$ to ensure positivity" (§2.2). mHC: "firstly makes all elements to be positive via an exponent operator" (mHC §4.2) |
| $M^{(t)}$ | $n_{\text{hc}}\times n_{\text{hc}}$ | The matrix after $t$ rounds of the loop. $(t)$ is a round number, not a power. | Computed intermediate | "getting $M^{(0)} = \exp(\tilde{B}_l)$, and then iteratively performs column and row normalization" (§2.2). Eq. (8) |
| $t$ | Integer, $1,\dots,t_{\text{max}}$ | The loop counter | Index | Eq. (8) |
| $t_{\text{max}}$ | Integer, 20 | The number of rounds | Fixed setting | "We choose $t_{\text{max}} = 20$ as a practical value." (§2.2). Also §4.2.1 |
| $\mathcal T_r$, $\mathcal T_c$ | $n\times n\to n\times n$ | Divide each row ($\mathcal T_r$) or each column ($\mathcal T_c$) by its own sum | Fixed setting (operations with no weights) | "where $\mathcal{T}_r$ and $\mathcal{T}_c$ denote row and column normalization, respectively" (§2.2). mHC says what the normalization does: "conducts iterative normalization process that alternately rescales rows and columns to sum to 1" (mHC §4.2) |
| $\hat X_l$ | $1\times n_{\text{hc}}d$ | The token's whole state, flattened and RMS-normalized. In the code, each entry is divided by the root mean square of all $n_{\text{hc}}d$ entries, and no learned weight multiplies the result: the mHC paper absorbs the RMSNorm weight into the projection (see "Where the RMSNorm goes"). | Computed intermediate | "it is first flattened and normalized: $\hat{X}_l = \operatorname{RMSNorm}(\operatorname{vec}(X_l)) \in \mathbb{R}^{1 \times n_{\text{hc}}d}$" (§2.2) |
| $W^{\mathrm{pre}}_l, W^{\mathrm{post}}_l, W^{\mathrm{res}}_l$ | $n_{\text{hc}}d\times n_{\text{hc}}$, $n_{\text{hc}}d\times n_{\text{hc}}$, $n_{\text{hc}}d\times n_{\text{hc}}^2$ | Linear maps from the flattened state to the input-dependent part of $\tilde A_l$, $\tilde C_l$, $\tilde B_l$ | Trained weight. §2.4 assigns the "static biases and gating factors" to AdamW and "All other modules" to Muon (another optimizer), so these use Muon. | "are learnable parameters for generating the dynamic components" (§2.2) |
| $S^{\mathrm{pre}}_l, S^{\mathrm{post}}_l, S^{\mathrm{res}}_l$ | $1\times n_{\text{hc}}$, $n_{\text{hc}}\times1$, $n_{\text{hc}}\times n_{\text{hc}}$ | The part of the raw values that does not depend on the input | Trained weight (AdamW, §2.4) | "are learnable static biases" (§2.2) |
| $\alpha^{\mathrm{pre}}_l,\alpha^{\mathrm{res}}_l,\alpha^{\mathrm{post}}_l$ | Scalars | Scale the input-dependent part | Trained weight (AdamW, §2.4) | "are learnable gating factors initialized to small values" (§2.2). mHC Table 5 (Appendix A.1) gives "mHC/HC Gating Factor Init $\alpha$" as 0.01 for the mHC paper's own models. |
| $\operatorname{Mat}(\cdot)$ | $1\times n_{\text{hc}}^2\to n_{\text{hc}}\times n_{\text{hc}}$ | A reshape | Fixed setting (an operation with no weights) | "$\operatorname{Mat}(\cdot)$ reshapes a vector of size $1 \times n_{\text{hc}}^2$ into a matrix of size $n_{\text{hc}} \times n_{\text{hc}}$" (§2.2) |
| $\sigma$ | Entrywise | The sigmoid $1/(1+e^{-z})$, which maps any number into $(0,1)$. It is not a standard deviation. | Fixed setting (an operation with no weights) | "we employ a Sigmoid function $\sigma(\cdot)$ to ensure their non-negativity and boundedness" (§2.2) |

**Not defined in the paper:**

- **$n$ and $\mathbf 1_n$ in (2).** V4 never defines $\mathbf 1_n$, and it does not define the $n$ in (2). V4 uses $n$ for the sequence length (§2.3.1, §2.3.2) and for the row count of a weight matrix $W\in\mathbb R^{n\times m}$ (Algorithm 1, §2.4). Neither is the $n$ of (2). I searched §2.2, §4.2.1 and the whole TeX source, including Algorithm 1, for `\mathbf{1}` and for every bare $n$. The definitions above come from the mHC paper.
- **Three meanings left unstated in V4:** that $\geqslant$ and $\exp$ act entry by entry, and that "normalization" in $\mathcal T_r$ and $\mathcal T_c$ means dividing by the sum. The mHC paper states all three, as quoted in the table.
- **The order of entries in $\operatorname{vec}(\cdot)$ and $\operatorname{Mat}(\cdot)$** (row-major or column-major). Neither paper states it (§2.2 here, mHC §4.2). The code decides it (see below).
- **The bar $\mid$ in (2).** Neither paper says how to read it: I searched both papers' TeX source and PDF text for a definition, "such that" and "s.t.", and found none. mHC's eq. (6) uses the same bar in the same way. Here it stands inside set braces, between the variable $M\in\mathbb R^{n\times n}$ and the conditions, and §2.2 speaks of "the set $\mathcal{M}$". So it means "such that", not conditioning as in $p(y\mid x)$: read (2) like `{M for M in R^{n×n} if …}`.
- **The bracket notation $[\mathbf x_{l,1};\ldots;\mathbf x_{l,n_{\text{hc}}}]^T$.** The paper does not say how to read it, and its stated shape $n_{\text{hc}}\times d$ settles the reading: each $\mathbf x_{l,i}$ is one row of $X_l$.

## Read from the inside out

The equations are written for one token. $X_l$ has no sequence axis. In the
code every one of these quantities has two more leading axes, for the batch and
the sequence position.

Equation (1):

1. **$A_lX_l$:** $(1\times n_{\text{hc}})(n_{\text{hc}}\times d)\to 1\times d$.
   Entry $k$ is $\sum_j A_l[j]\,X_l[j,k]$, a sum over the stream index $j$. Each
   entry of $A_l$ lies in $(0,1)$, but nothing makes the entries sum to 1. So the
   result is a nonnegative weighted sum of the streams, not an average. The paper
   writes the result as "$\in\mathbb R^d$". It is a row of length $d$.
2. **$\mathcal F_l(\cdot)$:** the sublayer takes that length-$d$ input and
   returns a length-$d$ output. For attention, the sublayer also reads other
   tokens' $A_lX_l$. The mHC parts act on each token separately.
3. **$C_l\,\mathcal F_l(\cdot)$:** $(n_{\text{hc}}\times1)(1\times d)\to n_{\text{hc}}\times d$.
   This is an outer product: stream $i$ receives $C_l[i]$ times the sublayer's
   output.
4. **$B_lX_l$:** $(n_{\text{hc}}\times n_{\text{hc}})(n_{\text{hc}}\times d)\to n_{\text{hc}}\times d$.
   Output row $i$ is $\sum_j B_l[i,j]\,X_l[j,:]$, a sum over the input streams
   $j$. The same weights apply to all $d$ features.
5. **$+$ and $=$:** the two $n_{\text{hc}}\times d$ matrices are added entry by
   entry. The $=$ defines the next layer's state. The paper says "the update of
   the residual state", but this is one step of the forward pass, from layer $l$
   to layer $l+1$, not a training step.

The superscript $T$ is a transpose in $X_l$'s definition, in $\mathbf 1_n^T$ in
(2) and in (5). In (3)–(5), "pre", "res" and "post" are labels.

Equation (2), read from the conditions outward:

1. **$M\mathbf 1_n$:** $(n\times n)(n\times1)\to n\times1$, the vector of row
   sums. "$=\mathbf 1_n$" says every row sums to 1. Together with $M\geqslant0$,
   this makes each output row of $B_lX_l$ a weighted average of the input
   streams, with weights that are $\ge0$ and sum to 1. The mHC paper calls such
   an average a _convex combination_.
2. **$\mathbf 1_n^TM$:** $(1\times n)(n\times n)\to1\times n$, the vector of
   column sums. "$=\mathbf 1_n^T$" says every column sums to 1. One consequence
   (my one-line derivation): $\mathbf 1^T(B_lX_l)=(\mathbf 1^TB_l)X_l=\mathbf 1^TX_l$.
   For every feature, the sum over the streams does not change. This is what the
   mHC paper means by "the feature mean is conserved" (mHC §1).
3. **$M\geqslant0$:** every entry is nonnegative.
4. **$\{M\in\mathbb R^{n\times n}\mid\ldots\}$:** the set of all real
   $n\times n$ matrices that meet all three conditions. The commas mean "and".
   $\coloneqq$ gives that set the name $\mathcal M$. "$B_l\in\mathcal M$" states
   that $B_l$ must be in the set.

After (2) the paper makes a claim about the _spectral norm_ $\Vert B_l\Vert_2$,
examined under "What the paper claims and what it shows". The spectral norm is the largest factor by which a matrix can stretch a vector. Here
the vector is one feature's $n_{\text{hc}}$ values across the streams. The
paper's word for "never stretches" is _non-expansive_.

Equation (8):

1. **$M^{(0)}=\exp(\tilde B_l)$:** $e^x$ of every entry. All entries are then
   $>0$.
2. **$\mathcal T_c(M^{(t-1)})$:** this is the innermost operation, so it runs
   first. It divides each column by its sum, so the columns sum to 1.
3. **$\mathcal T_r(\cdot)$:** divides each row by its sum, so the rows sum to 1.
   This usually makes the column sums differ from 1 again.
4. **Repeat** for $t=1,\dots,t_{\text{max}}$, which gives 20 column steps and 20
   row steps. Every step multiplies whole rows or whole columns by positive
   numbers. So the result is $\exp(\tilde B_l)$ with its rows and columns
   rescaled, and every entry stays positive (my observation). The paper calls
   this loop the _Sinkhorn–Knopp algorithm_.
5. **$B_l=M^{(t_{\text{max}})}$:** the last step is $\mathcal T_r$. So the rows
   of $B_l$ sum to 1 up to floating-point rounding, but the columns need not.
   On the spread $4\times4$ input in the check below, 20 rounds leave column
   sums from 0.95 to 1.05 and a spectral norm of 1.0086, slightly above 1. The mHC paper says the loop converges to a doubly stochastic matrix
   "as $t_{\text{max}}\to\infty$" (mHC §4.2).

**Kind of statement:**

- **(1) is a computation.** It is the forward pass through one layer, and the
  shapes above agree.
- **(2) is a definition used as a constraint.** $\coloneqq$ names the set
  $\mathcal M$, and $B_l\in\mathcal M$ restricts $B_l$ to it. The constraint does
  not say how to produce such a $B_l$.
- **(8) is a computation that produces $B_l$.** It is a loop whose limit meets
  (2).

## Where the samples come from

None of these equations has an expectation, a sampled value or a dataset
average. The mHC parts, (2)–(8), depend only on one token's state $X_l$ and the
trained weights. (1) also depends on other tokens, through an attention
sublayer $\mathcal F_l$ (step 2 under "Equation (1)" above).

Neither paper describes a stop-gradient here. The mHC paper says training
backpropagates through the whole loop of (8): "For the backward pass, we derive
a custom backward kernel that recomputes the intermediate results on-chip and
traverses the entire iteration." (mHC §4.3.1)

## Check with small numbers

The inputs cover each equation in turn:

- **Equation (1):** 2 streams and 3 features with answers I can compute by
  hand. Both terms, $B_lX_l$ and $C_l\mathcal F_l(A_lX_l)$, are nonzero, and every
  sum has two items. The sublayer is replaced by $\mathcal F(v)=2v$.
- **Code orientation:** a non-symmetric $3\times3$ member of $\mathcal M$ shows
  which way round the official code's `comb` array holds $B_l$.
- **Equation (2):** one matrix passes all three conditions. Three others each
  fail a different condition once.
- **Equation (8):**
  - $\exp(\tilde B)=\begin{pmatrix}1&3\\1&1\end{pmatrix}$. Its first round can be
    checked by hand, and its limit is known exactly (my derivation):
    - Every step only rescales rows or columns, so the ratio
      $M_{11}M_{22}/(M_{12}M_{21})$, which starts at $1\cdot1/(3\cdot1)=1/3$,
      never changes. Each row and each column holds one entry of the numerator
      and one of the denominator, so its factor cancels.
    - A $2\times2$ doubly stochastic matrix has the form
      $\begin{pmatrix}p&1-p\\1-p&p\end{pmatrix}$: the row sums fix each row's
      second entry, and the first column's sum makes $M_{21}=1-p$.
    - The limit is doubly stochastic (mHC §4.2) and keeps the ratio, so
      $p^2/(1-p)^2=1/3$. Both $p$ and $1-p$ are positive, so
      $p/(1-p)=1/\sqrt3$, which gives $p=1/(1+\sqrt3)$.
  - A spread $4\times4$ input, which I chose, to show what 20 rounds leave
    undone.
- **Equations (6)–(7):** the sigmoid is evaluated below, at and above 0.

```python
import numpy as np

np.set_printoptions(precision=4, suppress=True)
ones = lambda n: np.ones(n)

# ---------- Eq. (1): X_{l+1} = B X + C F(A X), n=2 streams, d=3 features ----------
X = np.array([[1.0, 2.0, 3.0],
              [3.0, 0.0, 1.0]])            # n x d, row i = stream i
A = np.array([[0.5, 0.25]])                # 1 x n
B = np.array([[0.75, 0.25],
              [0.25, 0.75]])               # n x n, doubly stochastic
C = np.array([[1.0], [0.5]])               # n x 1
F = lambda v: 2.0 * v                      # stand-in for the sublayer, R^d -> R^d

AX = A @ X                                 # 1 x d
CF = C @ F(AX)                             # (n x 1)(1 x d) = n x d, outer product
BX = B @ X                                 # n x d
X_next = BX + CF
assert AX.shape == (1, 3) and CF.shape == (2, 3) and BX.shape == (2, 3)
assert np.allclose(AX, [[1.25, 1.0, 1.75]])
assert np.allclose(BX, [[1.5, 1.5, 2.5], [2.5, 0.5, 1.5]])
assert np.allclose(CF, [[2.5, 2.0, 3.5], [1.25, 1.0, 1.75]])
assert np.allclose(X_next, [[4.0, 3.5, 6.0], [3.75, 1.5, 3.25]])
print("eq1 A X        =", AX)
print("eq1 B X        =", BX.tolist())
print("eq1 C F(A X)   =", CF.tolist())
print("eq1 X_{l+1}    =", X_next.tolist())
# columns of B sum to 1 -> sum over streams is unchanged by B X
assert np.allclose(ones(2) @ BX, ones(2) @ X)
print("sum over streams: X ->", ones(2) @ X, " B X ->", ones(2) @ BX)

# NumPy translations of the official code (model.py hc_pre / hc_post at a1fd202), with the same axes [b,s,hc,d]
# model.py line 680: y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
def code_hc_pre_sum(pre, x):               # x is passed in as [b,s,hc,d], the order x.view(shape) restores (see "What the code must decide")
    return np.sum(pre[..., None] * x, axis=2)
# model.py line 685: y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
def code_hc_post(x, residual, post, comb):
    return post[..., None] * x[..., None, :] + np.sum(comb[..., None] * residual[..., None, :], axis=2)

B3 = np.array([[0.5, 0.5, 0.0],           # non-symmetric doubly stochastic, n=3
               [0.0, 0.5, 0.5],
               [0.5, 0.0, 0.5]])
X3 = np.array([[1.0, 0.0], [0.0, 2.0], [4.0, 0.0]])
A3 = np.array([0.2, 0.4, 0.6]); C3 = np.array([1.0, 0.5, 1.5])
lay_in = code_hc_pre_sum(A3[None, None], X3[None, None])          # [1,1,d]
assert np.allclose(lay_in[0, 0], A3 @ X3)
out_T = code_hc_post(F(lay_in), X3[None, None], C3[None, None], B3.T[None, None])[0, 0]
out_N = code_hc_post(F(lay_in), X3[None, None], C3[None, None], B3[None, None])[0, 0]
paper = B3 @ X3 + np.outer(C3, F(A3 @ X3))
assert np.allclose(out_T, paper) and not np.allclose(out_N, paper)
print("code comb = B^T reproduces eq1:", np.allclose(out_T, paper), "| comb = B does:", np.allclose(out_N, paper))

# ---------- Eq. (2): membership in the set M ----------
def in_M(Mx, tol=1e-12):
    n = Mx.shape[0]
    rows = np.allclose(Mx @ ones(n), ones(n), atol=tol)
    cols = np.allclose(ones(n) @ Mx, ones(n), atol=tol)
    nonneg = bool((Mx >= 0).all())
    return rows, cols, nonneg

R   = np.array([[0.9, 0.1], [0.3, 0.7]])   # rows sum to 1, columns do not
Rt  = R.T                                  # columns sum to 1, rows do not
Neg = np.array([[1.5, -0.5], [-0.5, 1.5]]) # rows and columns sum to 1, has negatives
for name, Mx in [("B", B), ("R", R), ("R^T", Rt), ("Neg", Neg), ("B3", B3)]:
    print(f"eq2 {name:4s} (rows, cols, >=0) =", in_M(Mx), " spectral norm =", round(np.linalg.norm(Mx, 2), 4))
assert in_M(B) == (True, True, True) and in_M(B3) == (True, True, True)
assert in_M(R) == (True, False, True) and in_M(Rt) == (False, True, True) and in_M(Neg) == (True, True, False)
assert np.isclose(np.linalg.norm(B, 2), 1.0) and np.isclose(np.linalg.norm(B3, 2), 1.0)
assert np.isclose(np.linalg.norm(Neg, 2), 2.0)
# closure under multiplication, and what 30 layers of each do
P_B = np.linalg.matrix_power(B, 30)                        # 30 copies of B
P_mix = B3 @ B3.T @ B3                                     # product of different members
assert in_M(P_mix, 1e-12) == (True, True, True)
P_N = np.linalg.matrix_power(Neg, 30)
print("30 layers: norm(B^30) =", round(np.linalg.norm(P_B, 2), 4), " norm(Neg^30) =", f"{np.linalg.norm(P_N, 2):.4g}")
assert np.isclose(np.linalg.norm(P_B, 2), 1.0) and np.isclose(np.linalg.norm(P_N, 2), 2.0**30)

# ---------- Eq. (8): Sinkhorn-Knopp, M(t) = T_r(T_c(M(t-1))) ----------
T_c = lambda Mx: Mx / Mx.sum(axis=0, keepdims=True)   # each column / its column sum
T_r = lambda Mx: Mx / Mx.sum(axis=1, keepdims=True)   # each row / its row sum
def sinkhorn_paper(Bt, t_max=20):
    Mx = np.exp(Bt)                                    # entrywise exp, M(0)
    for _ in range(t_max):
        Mx = T_r(T_c(Mx))
    return Mx

Bt = np.array([[0.0, np.log(3.0)], [0.0, 0.0]])       # exp -> [[1,3],[1,1]]
M0 = np.exp(Bt)
assert np.allclose(M0, [[1, 3], [1, 1]])
M_c = T_c(M0); M1 = T_r(M_c)
assert np.allclose(M_c, [[0.5, 0.75], [0.5, 0.25]])
assert np.allclose(M1, [[0.4, 0.6], [2 / 3, 1 / 3]])
print("eq8 M(0) =", M0.tolist(), " T_c(M(0)) =", M_c.tolist(), " M(1) =", M1.round(4).tolist())
print("    after one iteration: row sums", M1.sum(1), " column sums", M1.sum(0).round(4))
B20 = sinkhorn_paper(Bt, 20)
p = 1 / (1 + np.sqrt(3))                               # exact limit: cross-ratio 1/3 is kept
limit = np.array([[p, 1 - p], [1 - p, p]])
print("    M(20) =", B20.round(6).tolist(), " exact limit =", limit.round(6).tolist())
assert np.allclose(B20, limit, atol=1e-9)
cross = lambda Mx: Mx[0, 0] * Mx[1, 1] / (Mx[0, 1] * Mx[1, 0])
assert np.isclose(cross(B20), cross(M0))               # only rows/columns were rescaled
# subtracting each column's max (the code's row softmax on comb = B~^T) cancels in the first T_c
for t in (1, 20):
    assert np.allclose(sinkhorn_paper(Bt - Bt.max(0, keepdims=True), t), sinkhorn_paper(Bt, t))

# a spread input: 20 iterations leave the columns off
Bs = np.tril(np.full((4, 4), -12.0), -1)               # 0 on/above diagonal, -12 below
for t in (1, 20, 200):
    Mt = sinkhorn_paper(Bs, t)
    print(f"    spread input, t_max={t:3d}: max|row sum-1| = {abs(Mt.sum(1)-1).max():.1e},"
          f" column sums = {Mt.sum(0).round(4)}, norm = {np.linalg.norm(Mt, 2):.4f}")
M20s = sinkhorn_paper(Bs, 20)
assert np.allclose(M20s.sum(1), 1) and not np.allclose(M20s.sum(0), 1, atol=1e-3)

# official kernel order (kernel.py hc_split_sinkhorn_kernel at a1fd202), run on comb = Bt^T
def sinkhorn_code(comb, iters=20, eps=1e-6):
    comb = np.exp(comb - comb.max(1, keepdims=True)); comb = comb / comb.sum(1, keepdims=True) + eps
    comb = comb / (comb.sum(0, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(1, keepdims=True) + eps)
        comb = comb / (comb.sum(0, keepdims=True) + eps)
    return comb
for name, Bx in (("2x2 input", Bt), ("spread input", Bs)):
    exact = abs(sinkhorn_code(Bx.T, eps=0.0).T - sinkhorn_paper(Bx)).max()
    with_eps = abs(sinkhorn_code(Bx.T).T - sinkhorn_paper(Bx)).max()
    print(f"    {name}: code order (rows first on comb, transposed) vs eq8: eps=0 -> {exact:.1e}, eps=1e-6 -> {with_eps:.1e}")
    assert exact < 1e-12

# ---------- eqs. (6)-(7) ranges, eq. (3)-(5) sizes ----------
sig = lambda z: 1 / (1 + np.exp(-z))
for z in (-10.0, 0.0, 10.0):
    print(f"    A = sigma({z:+.0f}) = {sig(z):.5f}   C = 2 sigma({z:+.0f}) = {2*sig(z):.5f}")
assert np.isclose(sig(0.0), 0.5) and np.isclose(2 * sig(0.0), 1.0)
n = 4
assert n * n + 2 * n == 24                              # "output dimension of only 24"
for name, d in (("Flash", 4096), ("Pro", 7168)):
    per = n * d * (n * n + 2 * n) + (n * n + 2 * n) + 3  # W's + S's + alphas, one sublayer
    print(f"    {name}: d={d}, residual state per token n*d={n*d}, mHC params per sublayer={per:,}")
# the code divides by the RMS after the matmul; same result as normalizing first
rng = np.random.default_rng(0); xv = rng.normal(size=8); W = rng.normal(size=(24, 8))
rms = np.sqrt((xv ** 2).mean() + 1e-6)
assert np.allclose((xv / rms) @ W.T, (xv @ W.T) / rms)
print("all checks passed")
```

Output, from
`uv run --no-project --with numpy python check.py`:

```text
eq1 A X        = [[1.25 1.   1.75]]
eq1 B X        = [[1.5, 1.5, 2.5], [2.5, 0.5, 1.5]]
eq1 C F(A X)   = [[2.5, 2.0, 3.5], [1.25, 1.0, 1.75]]
eq1 X_{l+1}    = [[4.0, 3.5, 6.0], [3.75, 1.5, 3.25]]
sum over streams: X -> [4. 2. 4.]  B X -> [4. 2. 4.]
code comb = B^T reproduces eq1: True | comb = B does: False
eq2 B    (rows, cols, >=0) = (True, True, True)  spectral norm = 1.0
eq2 R    (rows, cols, >=0) = (True, False, True)  spectral norm = 1.0298
eq2 R^T  (rows, cols, >=0) = (False, True, True)  spectral norm = 1.0298
eq2 Neg  (rows, cols, >=0) = (True, True, False)  spectral norm = 2.0
eq2 B3   (rows, cols, >=0) = (True, True, True)  spectral norm = 1.0
30 layers: norm(B^30) = 1.0  norm(Neg^30) = 1.074e+09
eq8 M(0) = [[1.0, 3.0000000000000004], [1.0, 1.0]]  T_c(M(0)) = [[0.5, 0.7500000000000001], [0.5, 0.25]]  M(1) = [[0.4, 0.6], [0.6667, 0.3333]]
    after one iteration: row sums [1. 1.]  column sums [1.0667 0.9333]
    M(20) = [[0.366025, 0.633975], [0.633975, 0.366025]]  exact limit = [[0.366025, 0.633975], [0.633975, 0.366025]]
    spread input, t_max=  1: max|row sum-1| = 0.0e+00, column sums = [0.48   0.7016 1.0391 1.7793], norm = 1.1886
    spread input, t_max= 20: max|row sum-1| = 1.1e-16, column sums = [0.9501 0.9802 1.0175 1.0522], norm = 1.0086
    spread input, t_max=200: max|row sum-1| = 2.2e-16, column sums = [1. 1. 1. 1.], norm = 1.0000
    2x2 input: code order (rows first on comb, transposed) vs eq8: eps=0 -> 0.0e+00, eps=1e-6 -> 9.4e-07
    spread input: code order (rows first on comb, transposed) vs eq8: eps=0 -> 0.0e+00, eps=1e-6 -> 2.5e-03
    A = sigma(-10) = 0.00005   C = 2 sigma(-10) = 0.00009
    A = sigma(+0) = 0.50000   C = 2 sigma(+0) = 1.00000
    A = sigma(+10) = 0.99995   C = 2 sigma(+10) = 1.99991
    Flash: d=4096, residual state per token n*d=16384, mHC params per sublayer=393,243
    Pro: d=7168, residual state per token n*d=28672, mHC params per sublayer=688,155
all checks passed
```

What the numbers show:

- **The mixing preserves the sum.** The check's `B`, an exact member of
  $\mathcal M$, mixes the streams while keeping each feature's sum over the
  streams (4, 2, 4 before and after).
- **All three conditions of (2) are needed.** Without them the norm can
  exceed 1. `Neg` has rows and columns that sum to 1, yet it doubles some
  vectors. Thirty layers of `Neg` give $2^{30}\approx1.07\times10^9$, while
  thirty layers of `B` stay at 1. `R`, whose columns do not sum to 1, has norm
  1.03.
- **The 2×2 input of (8) matches its exact limit** $p=1/(1+\sqrt3)$ to all six
  printed decimals after 20 rounds. The spread $4\times4$ input does not get
  that close (step 5 under "Equation (8)").

## What the code must decide

All citations are to the files linked from V4 §2.3, footnote 1, at commit
`a1fd202`. This is inference code. The paper links no mHC training code, so
choices that only training makes are open.

- **Which way round $B_l$ is stored:** `Block.hc_post` (`model.py` line 685)
  computes $\sum_j$`comb[j,k]`$\cdot$`residual[j]`: each `unsqueeze` inserts a
  size-1 axis, the product broadcasts as NumPy would, and `torch.sum` with
  `dim=2` adds over the input-stream axis $j$ (PyTorch 2.10 `torch.unsqueeze`,
  "Broadcasting semantics" and `torch.sum`; see Sources). So `comb` holds
  $B_l^T$. The kernel (`kernel.py` lines 401–423) normalizes `comb` row-first,
  which is column-first on $B_l$: there `T.reduce_max` and `T.reduce_sum` with
  `dim=1` remove the second axis, giving one value per row (tilelang 0.1.8,
  `tilelang/language/reduce_op.py`). That matches (8), and the check shows the
  two give identical results when the epsilons are set to 0.
- **The layout of the projected values, and $\operatorname{Mat}$ / $\operatorname{vec}$ order:**
  - One matmul produces 24 numbers, laid out as
    [pre (4), post (4), res (16)] (`kernel.py` lines 391–396).
  - The res block is read row-major, entry $4j+k$, into `comb`. Because `comb`
    is $B_l^T$, the paper's $\tilde B_l$ is the transpose of that row-major
    reshape.
  - The three $\alpha$'s are `hc_scale[0..2]`, in the order pre, post, res.
  - The flattening is `x.flatten(2)` on `[b,s,hc,d]`, stream after stream
    (`model.py` lines 673–681). PyTorch 2.10 `torch.flatten` says "The order
    of elements in input is unchanged", and its example with `start_dim=1`
    turns a block with rows (1, 2) and (3, 4) into (1, 2, 3, 4).
  - Line 680's `x.view(shape)` splits that axis back in the same order: entry
    $jd+k$ becomes stream $j$, feature $k$. PyTorch's Python reference
    implementation at tag v2.10.0, the definition it registers for
    `aten.view` (`torch/_refs/__init__.py` lines 4866–4868), shows this. For
    this shape it runs the general algorithm (line 3995), which leaves axes
    that already have the right length alone ("Skips dimensions that are
    already the correct length", line 3896) and splits the long one with
    `prims.split_dim` (line 3927). The split gives the outer new axis (streams)
    a step of `inner_length` ($=d$) entries of the old axis, and the inner one
    (features) a step of one: the strides `a.stride()[idx] * inner_length` and
    `a.stride()[idx]` (`torch/_prims/__init__.py` line 1587). Line 734 in
    `hc_head` is the same split. The `view` calls in `kernel.py` lines 436–437
    only merge the batch and sequence axes, and by the same skip the last axes
    keep their order.
- **Where the RMSNorm goes:** `mixes = F.linear(x, hc_fn) * rsqrt` (`model.py`
  line 678), with `rsqrt` = $1/\sqrt{\text{mean}(x^2)+10^{-6}}$ taken over all
  $n_{\text{hc}}d$ entries (line 677: `torch.rsqrt` takes $1/\sqrt{\cdot}$ of
  each entry, `mean(-1)` averages the last axis, of size $n_{\text{hc}}d$ after
  the flatten, and $10^{-6}$ is `norm_eps`, line 61, which
  `inference/config.json` does not set; PyTorch 2.10 `torch.rsqrt`,
  `torch.mean`).
  - The division comes after the matmul (`F.linear` multiplies `x` by the
    transposed weight, PyTorch 2.10 `torch.nn.functional.linear`), as mHC
    §4.3.1 describes: "we reorder
    the dividing-by-norm operation to follow the matrix multiplication".
  - There is no separate norm weight: "the RMSNorm weight is also absorbed in
    $\varphi_l$" (mHC §4.3.1). $\varphi_l$ is the mHC paper's name for V4's
    $W^{\mathrm{pre}}_l$, $W^{\mathrm{post}}_l$ and $W^{\mathrm{res}}_l$ joined
    into one $n_{\text{hc}}d\times24$ matrix (mHC eq. (10)). In the code it is
    `hc_fn`, stored as $24\times n_{\text{hc}}d$ (`hc_attn_fn` or `hc_ffn_fn`,
    `model.py` lines 666–667).
  - The norm and the matmul run on `x` cast to float32 (`.float()`, line 676,
    is `self.to(torch.float32)` per PyTorch 2.10 `Tensor.float`), and the
    kernel takes float32 tensors (`kernel.py` lines 379–384).
- **Overflow in $\exp$:** the first step is a softmax over each row of `comb`,
  which subtracts the row maximum first (`kernel.py` lines 401–408). In the
  paper's orientation that is each column's maximum of $\tilde B_l$. After
  $\exp$, subtracting it divides that whole column by one positive number,
  which the first $\mathcal T_c$ cancels exactly (tested in the check).
- **Epsilons:** `hc_eps: float = 1e-6` (`ModelArgs`, `model.py` line 80;
  `inference/config.json` does not set it) is added in four places inside
  `hc_split_sinkhorn_kernel`, and once more in `hc_head` (see "How the streams
  end"):
  - to every entry after the first softmax (`kernel.py` line 408);
  - to every row-sum denominator of `comb` except the first, the softmax's own
    (line 419);
  - to every column-sum denominator of `comb` (lines 413 and 423);
  - to $A_l$ (`pre[i, j] = T.sigmoid(…) + eps`, line 392), which equation (6) does
    not have.

  It is not added to $C_l$ (line 394). On the 2×2 input the epsilons change the
  result by about $10^{-6}$. On my spread input, whose smallest entries $e^{-12}\approx6\times10^{-6}$ are close
  to `eps`, they change it by $2.5\times10^{-3}$.
- **The number of rounds:** `"hc_sinkhorn_iters": 20` (`inference/config.json`) and
  `hc_sinkhorn_iters: int = 20` (`ModelArgs`).
  That gives 20 column and 20 row normalizations in the paper's orientation,
  ending with a row step.
- **What $\mathcal F_l$ includes:** `Block.forward` (`model.py` lines 688–700)
  applies `attn_norm` or `ffn_norm` to $A_lX_l$ before attention or MoE. Each is
  an RMSNorm (lines 658–659) that multiplies its output by a learned
  `self.weight` (lines 189 and 196). The block's two layers (see the $l$ row)
  have separate mHC parameters (`hc_attn_*` and `hc_ffn_*`).
- **How the streams start:** `Transformer.forward` (`model.py` line 805)
  copies the token embedding into all four streams with
  `h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)`: `unsqueeze(2)` inserts a
  size-1 stream axis, and `repeat` "copies the tensor’s data" along it four
  times (PyTorch 2.10 `torch.unsqueeze`, `Tensor.repeat`). The paper does not
  describe this step. My observation: when all rows of $X_l$ are equal, every $B_l$ whose
  rows sum to 1 leaves $X_l$ unchanged, so at the first layer the streams can
  only start to differ through $C_l$.
- **How the streams end:** `ParallelHead.hc_head` (`model.py` lines 728–735)
  collapses the four streams before the final norm and the logits. It uses
  sigmoid weights plus `self.hc_eps` from a separate projection (`hc_head_fn`), with no
  $B$ or $C$. The paper does not describe this step.
- **Initial values of $W$, $S$ and $\alpha$:** open. The inference code loads
  them from the checkpoint. What the papers say about the $\alpha$'s is in
  their row of the symbol table.
- **The backward pass through (8):** open in this code. mHC §4.3.1 describes a
  custom backward kernel (quoted in "Where the samples come from").

## What the paper claims and what it shows

- **Plain HC is unstable.** V4 says: "we find that the training will frequently
  exhibit numerical instability when stacking multiple layers, which hinders the
  scaling of HC" (§2.2). V4 shows no figure or table for this. The mHC paper
  measures it on a 27B model:
  - "HC exhibits an unexpected loss surge around the 12k step" (mHC §3.1,
    Fig. 2).
  - It defines the _Amax Gain Magnitude_ as the largest absolute row sum
    (forward) or column sum (backward) of the product of the $B_l$'s across
    layers. This gain "yields extreme values with peaks of 3000" (mHC §3.1,
    Fig. 3(b)).
- **$\Vert B_l\Vert_2\le1$, "so the residual transformation is non-expansive"
  (§2.2).** This is a mathematical property of exact members of $\mathcal M$.
  Neither paper prints a proof. My derivation:
  - Each entry of $B\mathbf v$ is a weighted average of the entries of
    $\mathbf v$, so $(B\mathbf v)_i^2\le\sum_jB_{ij}v_j^2$. With
    $m=(B\mathbf v)_i$, the right side minus the left equals
    $\sum_jB_{ij}(v_j-m)^2$ (expand the square and use that row $i$ sums to
    1), and that is $\ge0$ because every $B_{ij}\ge0$.
  - Summing over $i$ and swapping the two sums gives
    $\Vert B\mathbf v\Vert^2\le\sum_jv_j^2\sum_iB_{ij}$. Each column sum
    $\sum_iB_{ij}$ is 1, so $\Vert B\mathbf v\Vert^2\le\Vert\mathbf v\Vert^2$.
  - $B\mathbf 1=\mathbf 1$: the all-ones vector keeps its length, so the
    largest stretch is exactly 1.

  The check shows that the claim needs all three conditions (see "What the
  numbers show"). The bound is on $B_l$ alone,
  so it limits only the pass-through term $B_lX_l$ of (1). Each layer still adds
  its sublayer output $C_l\mathcal F_l(A_lX_l)$ on top (my reading of (1)).

  That the property "increases the numerical stability during both the forward
  pass and backpropagation" is argued in V4, not measured there.
- **"the set $\mathcal M$ is closed under multiplication, which guarantees
  stability in the scenarios of deep stacks of mHC" (§2.2).** Closure holds for
  exact members (my one line: $(B_1B_2)\mathbf 1=B_1\mathbf 1=\mathbf 1$, and in
  the same way for columns and signs). The check confirms it numerically.
  "Guarantees" is stronger than what the mHC paper measures with
  $t_{\text{max}}=20$: "In our settings, we use 20 iterations to obtain an
  approximate solution. Consequently, as shown in Fig. 7(a), the backward
  gradient gain deviates slightly from 1. In the composite case shown in
  Fig. 7(b), the deviation increases but remains bounded, reaching a maximum
  value of approximately 1.6." (mHC §5.4) In V4's notation, Fig. 7(a) shows a
  single $B_l$, Fig. 7(b) the product of the $B_l$'s across layers, and the
  "backward gradient gain" is a column sum (mHC Fig. 7 and Fig. 8 captions).
- **The loop converges.** V4 says so without a condition (quoted in the $B_l$
  row of the symbol table). The mHC paper adds the condition that the number of
  rounds goes to infinity (step 5 under "Equation (8)" in "Read from the inside
  out"). Both papers call
  $t_{\text{max}}=20$ "a practical value" (the $t_{\text{max}}$ row; mHC §4.2).
  Neither paper reports how far from doubly stochastic the 20-round matrices are
  in V4.
- **The sigmoid on $A_l$ and $C_l$ is there "to avoid the risk of signal
  cancellation" (§2.2).** This is argued. I found no experiment in either paper
  that isolates it. Neither paper explains the factor 2 in (7). One effect (my
  observation) is that $\tilde C_l=0$ gives $C_l=1$.
- **Cost:**
  - V4 says the optimizations "constrain the wall-time overhead of mHC to only
    6.7% of the overlapped 1F1B pipeline stage" (§3.4.2). This is measured
    training wall time, with fused kernels, recomputation and a modified DualPipe
    schedule. 1F1B and DualPipe are schedules for pipeline-parallel training. No
    breakdown is given.
  - HC's "minimal computational overhead" (§2.2) refers to arithmetic. The mHC
    paper's Table 2 shows that memory traffic grows roughly in proportion to
    the number of streams, $n_{\text{hc}}$ (the mHC paper's $n$).
  - Computed in the check: the residual state is 4 times larger, 28,672 values
    per token for V4-Pro instead of 7,168. One mHC step has 688,155 parameters
    for V4-Pro.
  - V4's "output dimension of only 24" (§3.3) equals $n_{\text{hc}}^2+2n_{\text{hc}}$
    for $n_{\text{hc}}=4$ (recomputed).
- **mHC helps quality.** V4 contains no ablation of mHC: a search of its TeX for
  "ablat" finds nothing. The evidence is in the mHC paper:
  - on 27B models, "a final loss reduction of 0.021 compared to the baseline"
    (mHC §5.2, Fig. 5(a)) and benchmark scores in Table 4;
  - across 3B, 9B and 27B models, compute-scaling curves (mHC §5.3, Fig. 6).

  The mHC paper gives no ± values and does not say how many runs each number
  comes from.

## Sources

- DeepSeek-AI, DeepSeek-V4, [arXiv 2606.19348v1](https://arxiv.org/abs/2606.19348v1).
  I read the TeX source and the PDF text. Quotes and numbers were checked
  against the PDF text.
- Xie et al., mHC: Manifold-Constrained Hyper-Connections,
  [arXiv 2512.24880v2](https://arxiv.org/abs/2512.24880v2). TeX source and PDF
  text.
- Official inference code linked from V4 §2.3, footnote 1:
  [`inference/` at commit a1fd202](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/a1fd202632e91bc0efe4eedb63ce554649c53997/inference),
  files `model.py`, `kernel.py`, `config.json` and `requirements.txt`. The
  repository root has a second `config.json`, which does set
  `"hc_eps": 1e-06`; every `config.json` cited above is
  `inference/config.json`.
- PyTorch 2.10 documentation, the lowest version that
  `inference/requirements.txt` allows (`torch>=2.10.0`):
  [`torch.flatten`](https://docs.pytorch.org/docs/2.10/generated/torch.flatten.html),
  [`Tensor.repeat`](https://docs.pytorch.org/docs/2.10/generated/torch.Tensor.repeat.html),
  [`torch.unsqueeze`](https://docs.pytorch.org/docs/2.10/generated/torch.unsqueeze.html),
  [`torch.sum`](https://docs.pytorch.org/docs/2.10/generated/torch.sum.html),
  [`torch.mean`](https://docs.pytorch.org/docs/2.10/generated/torch.mean.html),
  [`torch.rsqrt`](https://docs.pytorch.org/docs/2.10/generated/torch.rsqrt.html),
  [`torch.nn.functional.linear`](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.linear.html),
  [`Tensor.float`](https://docs.pytorch.org/docs/2.10/generated/torch.Tensor.float.html)
  and [Broadcasting semantics](https://docs.pytorch.org/docs/2.10/notes/broadcasting.html)
  ("Many PyTorch operations support NumPy’s broadcasting semantics").
- PyTorch source at tag v2.10.0, for the element order of `view`:
  [`torch/_refs/__init__.py`](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_refs/__init__.py)
  and [`torch/_prims/__init__.py`](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_prims/__init__.py).
- tilelang, the version that `inference/requirements.txt` pins
  (`tilelang==0.1.8`):
  [`tilelang/language/reduce_op.py` at tag v0.1.8](https://github.com/tile-ai/tilelang/blob/v0.1.8/tilelang/language/reduce_op.py),
  whose `reduce` gives an input of shape `[X, d, Y]` an output of shape
  `[X, Y]` when reducing over `dim`.
- The commit list
  (`https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Pro/commits/main`),
  used to choose the commit. The arXiv API entries for both papers, used for
  the submission dates.
- Named, not opened: Zhu et al., _Hyper-Connections_ (the original HC paper);
  Sinkhorn and Knopp (1967).
