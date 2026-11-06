# DeepSeek-V2 Multi-head Latent Attention (MLA): attention that caches one short vector per token instead of every head's keys and values

**Source:** DeepSeek-AI,
[DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434v5),
v5 (19 June 2024, the latest version), §2.1.2, equations (9)–(13), and §2.1.3,
equations (14)–(19). In v5 the query compression, equations (12)–(13), is
printed at the end of §2.1.2, and the decoupled RoPE, (14)–(19), in §2.1.3.
Other locations used: §2.1.1, equations (1)–(8) and the text after them;
Figure 2; §2.1.4 and Table 1; §3.1.2, §3.1.3, §3.1.4 and §3.2.3; Appendix B.1;
Appendix C, equations (37)–(47); Appendix D.1–D.2, Tables 8–9; the Abstract,
Figure 1(b), §1 and §5 for the 93.3% figure. DeepSeek 67B's settings come from
the cited paper [DeepSeek LLM](https://arxiv.org/abs/2401.02954v1), v1, Table 2
and §2.2. The rotation
matrix behind RoPE comes from the cited paper, Su et al.,
[RoFormer](https://arxiv.org/abs/2104.09864v5), v5, §3.2.2, equations
(14)–(16). The repository the paper links holds no model code (see "What the
code must decide").

## What it computes and what it changes

The equations describe one attention layer. For token $t$ they take that
token's input vector and the inputs of every earlier token, and return one
output vector $\mathbf u_t$ with $d=5120$ numbers. While a model generates text,
it stores, for every layer and every past token, the vectors that later tokens
need from it, so they are not recomputed; this store is the _KV cache_
(key–value cache), and the paper calls its size "a large bottleneck that limits
the maximum batch size and sequence length" (§2.1.1). Standard multi-head
attention (MHA), §2.1.1 equations (1)–(8), projects each token straight into
keys and values for all 128 heads and caches all of them. Multi-head Latent
Attention (MLA), equations (9)–(11), first squeezes the token into one short
vector, which the paper calls the "compressed latent vector" (§2.1.2), and
rebuilds every head's key and value from it. It caches only that short vector plus one small key that carries
position, (15). The rotation that encodes each token's position (RoPE, rotary
position embedding) moves out of the main keys into separate small slices,
(14)–(17), so that the rebuilding never has to happen at inference (steps 9–10
below).

## The equation

As printed in v5 (§2.1.2 and §2.1.3):

$$
\begin{aligned}
\mathbf{c}_{t}^{KV} &= W^{DKV} \mathbf{h}_{t}, &(9)\\
\mathbf{k}_{t}^{C} &= W^{UK} \mathbf{c}_{t}^{KV}, &(10)\\
\mathbf{v}_{t}^{C} &= W^{UV} \mathbf{c}_{t}^{KV}, &(11)\\
\mathbf{c}_{t}^{Q} &= W^{DQ} \mathbf{h}_{t}, &(12)\\
\mathbf{q}_{t}^{C} &= W^{UQ} \mathbf{c}_{t}^{Q}, &(13)\\
[\mathbf{q}_{t, 1}^{R};\mathbf{q}_{t, 2}^{R};...;\mathbf{q}_{t, n_{h}}^{R}] = \mathbf{q}_{t}^{R} &= \operatorname{RoPE}({W^{QR}} \mathbf{c}_{t}^{Q}), &(14)\\
\mathbf{k}_{t}^{R} &= \operatorname{RoPE}({W^{KR}} \mathbf{h}_{t}), &(15)\\
\mathbf{q}_{t, i} &= [\mathbf{q}_{t, i}^{C}; \mathbf{q}_{t, i}^{R}], &(16)\\
\mathbf{k}_{t, i} &= [\mathbf{k}_{t, i}^{C}; \mathbf{k}_{t}^{R}], &(17)\\
\mathbf{o}_{t, i} &= \sum_{j=1}^{t} \operatorname{Softmax}_j\Big(\frac{\mathbf{q}_{t, i}^T \mathbf{k}_{j, i}}{\sqrt{d_{h} + d_{h}^{R}}}\Big) \mathbf{v}_{j, i}^{C}, &(18)\\
\mathbf{u}_{t} &= W^{O} [\mathbf{o}_{t, 1};\mathbf{o}_{t, 2};...;\mathbf{o}_{t, n_{h}}], &(19)
\end{aligned}
$$

## Symbols

Sizes are DeepSeek-V2's settings from §3.1.2.

| Symbol | Shape or type | Meaning | Role | Defined at |
| ------ | ------------- | ------- | ---- | ---------- |
| $\mathbf h_t$ | vector, $d=5120$ | the input to the attention layer for token $t$ | function input: Figure 2 shows it as the output of the RMS Norm placed before "Attention" in each block | "Let $d$ be the embedding dimension, $n_h$ be the number of attention heads, $d_h$ be the dimension per head, and $\mathbf h_t\in\mathbb R^d$ be the attention input of the $t$-th token at an attention layer." (§2.1.1) |
| $t$ | integer position | the token whose output is computed (the query's position) | index | same sentence: "the $t$-th token" (§2.1.1) |
| $j$ | integer, $1\le j\le t$ | a token whose key and value token $t$ reads | index | only by the limits of the sum in (18), $\sum_{j=1}^{t}$; no sentence names it |
| $i$ | integer, $1\le i\le n_h$ | the attention head | index | "the query, key, and value of the $i$-th attention head" (§2.1.1, after (8)) |
| $d$, $n_h$, $d_h$ | integers: 5120, 128, 128 | width of $\mathbf h_t$; number of heads; size of each head's content query, key and value | fixed setting | §2.1.1 sentence above; "We set the number of Transformer layers to 60 and the hidden dimension to 5120." and "In MLA, we set the number of attention heads $n_h$ to 128 and the per-head dimension $d_h$ to 128." (§3.1.2) |
| $d_c$ | integer: 512 | length of the cached latent vector | fixed setting | "$d_c(\ll d_hn_h)$ denotes the KV compression dimension" (§2.1.2); "The KV compression dimension $d_c$ is set to 512" (§3.1.2) |
| $d_c'$ | integer: 1536 | length of the query latent vector; the prime is part of the name | fixed setting | "$d_c'(\ll d_hn_h)$ denotes the query compression dimension" (§2.1.2); "the query compression dimension $d_c'$ is set to 1536" (§3.1.2) |
| $d_h^R$ | integer: 64 | per-head length of the position-carrying query slice and of the shared position key | fixed setting | "where $d_h^R$ denotes the per-head dimension of the decoupled queries and key" (§2.1.3); "we set the per-head dimension $d_h^R$ to 64" (§3.1.2) |
| $W^{DKV}$ | $d_c\times d$ = 512 × 5120 | down-projection for keys and values | trained weight | "$W^{DKV}\in\mathbb R^{d_c\times d}$ is the down-projection matrix" (§2.1.2) |
| $W^{UK}$, $W^{UV}$ | $d_hn_h\times d_c$ = 16384 × 512 each | up-projections that rebuild keys and values from the latent | trained weight | "$W^{UK},W^{UV}\in\mathbb R^{d_hn_h\times d_c}$ are the up-projection matrices for keys and values, respectively" (§2.1.2) |
| $W^{DQ}$, $W^{UQ}$ | $d_c'\times d$ = 1536 × 5120; $d_hn_h\times d_c'$ = 16384 × 1536 | down- and up-projection for queries | trained weight | "$W^{DQ}\in\mathbb R^{d_c'\times d},W^{UQ}\in\mathbb R^{d_hn_h\times d_c'}$ are the down-projection and up-projection matrices for queries, respectively" (§2.1.2) |
| $W^{QR}$, $W^{KR}$ | $d_h^Rn_h\times d_c'$ = 8192 × 1536; $d_h^R\times d$ = 64 × 5120 | produce the position-carrying queries and key | trained weight | "$W^{QR}\in\mathbb R^{d_h^Rn_h\times d_c'}$ and $W^{KR}\in\mathbb R^{d_h^R\times d}$ are matrices to produce the decouples queries and key, respectively" (§2.1.3, typo as printed) |
| $W^O$ | $d\times d_hn_h$ = 5120 × 16384 | output projection | trained weight | "$W^O\in\mathbb R^{d\times d_hn_h}$ denotes the output projection matrix" (§2.1.1) |
| $\mathbf c_t^{KV}$ | vector, $d_c=512$ | compressed latent vector for keys and values | computed intermediate; cached at inference | "$\mathbf c_t^{KV}\in\mathbb R^{d_c}$ is the compressed latent vector for keys and values" (§2.1.2); boxed in Appendix C (41): "where the boxed vectors in blue need to be cached for generation" (after (47)) |
| $\mathbf k_t^C$, $\mathbf v_t^C$ | vectors, $d_hn_h=16384$ | keys and values of all heads, rebuilt from the latent | computed intermediate; not cached | by (10)–(11); "the naive formula needs to recover $\mathbf k_t^C$ and $\mathbf v_t^C$ from $\mathbf c_t^{KV}$ for attention" (Appendix C) |
| $\mathbf c_t^Q$ | vector, $d_c'=1536$ | compressed latent vector for queries | computed intermediate | "$\mathbf c_t^Q\in\mathbb R^{d_c'}$ is the compressed latent vector for queries" (§2.1.2) |
| $\mathbf q_t^C$ | vector, $d_hn_h=16384$ | content queries of all heads | computed intermediate | by (13) |
| $\mathbf q_{t,i}^C$, $\mathbf k_{t,i}^C$, $\mathbf v_{j,i}^C$ | vectors, $d_h=128$ | head $i$'s slice of $\mathbf q_t^C$, $\mathbf k_t^C$, $\mathbf v_j^C$ | computed intermediate | not in §2.1.2–2.1.3; Appendix C (38), (42), (45): $[\mathbf k_{t,1}^C;\mathbf k_{t,2}^C;...;\mathbf k_{t,n_h}^C]=\mathbf k_t^C$ and the same for $\mathbf q^C$, $\mathbf v^C$ |
| $\mathbf q_t^R$, $\mathbf q_{t,i}^R$ | vectors, $d_h^Rn_h=8192$ and $d_h^R=64$ | position-carrying queries, all heads and head $i$ | computed intermediate | "additional multi-head queries $\mathbf q_{t,i}^R\in\mathbb R^{d_h^R}$ and a shared key $\mathbf k_t^R\in\mathbb R^{d_h^R}$ to carry RoPE" (§2.1.3) |
| $\mathbf k_t^R$ | vector, $d_h^R=64$, no head index | one position-carrying key shared by all heads | computed intermediate; cached at inference | same sentence; "During inference, the decoupled key should also be cached." (§2.1.3) |
| $\mathbf q_{t,i}$, $\mathbf k_{t,i}$ | vectors, $d_h+d_h^R=192$ | head $i$'s full query and key | computed intermediate | by (16)–(17) |
| $\mathbf o_{t,i}$ | vector, $d_h=128$ | head $i$'s output for token $t$ | computed intermediate | by (18) |
| $\mathbf u_t$ | vector, $d=5120$ | the layer's output for token $t$ | computed intermediate (the result) | by (19); Figure 2 labels it "Output Hidden $\mathbf u_t$" |
| $\operatorname{RoPE}(\cdot)$ | vector → vector of the same length, at position $t$ | rotates the vector by angles that depend on the token's position | fixed operation | "$\operatorname{RoPE}(\cdot)$ denotes the operation that applies RoPE matrices" (§2.1.3); the matrices are in the list below |
| $[\cdot;\cdot]$ | vectors → one longer vector | concatenation; read right to left in (14), it splits a vector into $n_h$ equal slices | operation | "$[\cdot;\cdot]$ denotes the concatenation operation" (§2.1.3); "will be sliced into $n_h$ heads" (§2.1.1, before (4)) |

**Not defined in the paper:**

- **The RoPE matrices.** DeepSeek-V2 does not give the rotation angles, the
  frequency base, which coordinates are rotated together, or which position
  number (counted from 0 or 1) is used. I looked in §2.1.3, Appendix C, §3.1.2,
  §3.1.4 and Appendix B.1. The paper cites Su et al. (2024). That paper,
  RoFormer v5 §3.2.2, defines $R^d_{\Theta,m}$ in equation (15) as a block-diagonal
  matrix of 2 × 2 rotations by angles $m\theta_1,\dots,m\theta_{d/2}$, and says it
  "is the rotary matrix with pre-defined parameters
  $\Theta=\{\theta_i=10000^{-2(i-1)/d}, i\in[1,2,...,d/2]\}$". This is RoFormer's
  definition, not a statement of what DeepSeek-V2 uses.
- **How RoPE in (14) splits across heads.** (14) applies RoPE to one vector of
  $d_h^Rn_h$ numbers and slices afterwards; the paper does not give the split
  (see "RoPE across heads in (14)" under "What the code must decide").
- **The superscript $T$** in $\mathbf q_{t,i}^T$ is not defined. It can only be
  a transpose: the softmax needs one number per $j$, and $\mathbf q^T\mathbf k$
  for two column vectors of equal length is that number (their dot product). It
  is not the sequence length.
- **The subscript $j$ on $\operatorname{Softmax}_j$** is not defined. I read it
  as normalizing over $j=1,\dots,t$, the same $j$ as the sum; that is the only
  reading under which the sum is a weighted average of values.
- **The letters inside the labels** ($C$, $R$, $D$, $U$) are not spelled out.
  The paper names each object in words (table above), so nothing depends on
  them.
- **The extra RMS Norm layers and scaling factors** of §3.1.2 do not appear in
  (9)–(19). The paper says where they sit, but not the norm's weights or the
  scale values (see "What the code must decide").

## Read from the inside out

Conventions first. Bold lower-case letters are column vectors and capital $W$s
are matrices, so $W\mathbf h$ is a matrix times a vector. A matrix in
$\mathbb R^{\text{out}\times\text{in}}$ has the same shape as the `weight` of
`nn.Linear(in, out, bias=False)`, and $W\mathbf h_t$ is that layer applied to
one token. For a batch of tokens stored as rows `H` of shape `[n, d]`, it is
`H @ W.T`. ([PyTorch 2.15 `torch.nn.Linear`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Linear.html)
gives the weight "of shape (out_features, in_features)" and the computation
"$y = xA^T + b$"; with `bias=False` "the layer will not learn an additive
bias".) Superscripts such as $KV$, $DKV$, $C$, $R$ are labels, never powers;
the one exception is $T$ in (18), which can only be a transpose (list above). Subscript $t$ is the token,
$i$ the head, $j$ an earlier token. Every "=" in (9)–(19) gives the name on the
left to the value on the right. "$\in\mathbb R^{a\times b}$" in the definitions
gives a shape. $\ll$ means much smaller than: $d_c=512$ against
$d_hn_h=16384$.

1. **(9) Compress.** $W^{DKV}\mathbf h_t$: (512 × 5120) times (5120) → 512
   numbers, $\mathbf c_t^{KV}$. This, with $\mathbf k_t^R$ from step 5, is all
   the layer keeps from token $t$.
2. **(10)–(11) Rebuild keys and values.** $W^{UK}\mathbf c_t^{KV}$ and
   $W^{UV}\mathbf c_t^{KV}$: (16384 × 512) times (512) → 16384 each. Appendix C
   (42), (45) cut each into $n_h=128$ slices of $d_h=128$: one content key
   $\mathbf k_{t,i}^C$ and one value $\mathbf v_{t,i}^C$ per head. Compared with
   MHA's $\mathbf k_t=W^K\mathbf h_t$ in (2), the single matrix $W^K$ is replaced
   by the product $W^{UK}W^{DKV}$, which passes through only 512 numbers. A
   matrix product with an inner size of 512 can produce at most 512 independent
   directions (its _rank_ is at most 512); that is what "low-rank" in
   "low-rank joint compression" (§2.1.2) refers to. "Joint" because keys and
   values share one latent. (The rank statement is mine.)
3. **(12)–(13) Same pattern for queries.** $\mathbf c_t^Q=W^{DQ}\mathbf h_t$
   has 1536 numbers; $\mathbf q_t^C=W^{UQ}\mathbf c_t^Q$ has 16384, cut into
   128 head queries $\mathbf q_{t,i}^C$ of 128 by Appendix C (38). Queries are
   never cached, so this changes no cache size; the paper's reason is
   activation memory in training, the memory that holds intermediate results
   for the backward pass (see the claims section).
4. **(14) Position queries.** $W^{QR}\mathbf c_t^Q$: (8192 × 1536) times (1536)
   → 8192 numbers, from the query latent, not from $\mathbf h_t$. RoPE then
   rotates them according to position $t$. In RoFormer's definition each
   consecutive pair of numbers is rotated like a 2-D point, by an angle equal to
   the position times a fixed frequency for that pair. The result is cut into
   128 head vectors $\mathbf q_{t,i}^R$ of 64.
5. **(15) One position key.** $W^{KR}\mathbf h_t$: (64 × 5120) times (5120) →
   64 numbers, from $\mathbf h_t$ directly, not from $\mathbf c_t^{KV}$, then
   rotated for position $t$. It has no head index: all 128 heads use the same
   $\mathbf k_t^R$. It is cached already rotated.
6. **(16)–(17) Concatenate.** Per head, $\mathbf q_{t,i}=[\mathbf q_{t,i}^C;\mathbf q_{t,i}^R]$
   and $\mathbf k_{t,i}=[\mathbf k_{t,i}^C;\mathbf k_t^R]$: 128 + 64 = 192
   numbers each (`torch.cat` along the last axis,
   [PyTorch 2.15 `torch.cat`](https://docs.pytorch.org/docs/2.15/generated/torch.cat.html),
   with $\mathbf k_t^R$ repeated
   for every head).
7. **(18) Attention for one head.** Innermost: $\mathbf q_{t,i}^T\mathbf k_{j,i}$
   is the dot product of two 192-vectors, one number per earlier token $j$.
   Because both are concatenations, it equals
   $\mathbf q_{t,i}^{C\,T}\mathbf k_{j,i}^C+\mathbf q_{t,i}^{R\,T}\mathbf k_j^R$:
   a content score plus a position score (my rewrite; check 1). Each score is
   divided by $\sqrt{d_h+d_h^R}=\sqrt{192}$, the length of the vectors being
   dotted, where MHA's (7) divides by $\sqrt{d_h}$. Then
   $\operatorname{Softmax}_j$ turns the $t$ scaled scores for $j=1,\dots,t$
   into $t$ positive weights that sum to 1 (weight $j$ is
   $e^{s_j}/\sum_{j'=1}^{t}e^{s_{j'}}$). Finally the sum over $j$ adds up the
   value vectors $\mathbf v_{j,i}^C$ (128 numbers each) with those weights:
   $\mathbf o_{t,i}$ has 128 numbers. The sum stops at $j=t$, so token $t$ never
   reads later tokens; in code this upper limit is the causal mask. Queries and
   keys have 192 numbers but values 128, so the kernel must accept different
   sizes.
8. **(19) Combine heads.** Concatenate the 128 head outputs into 16384 numbers;
   $W^O$ (5120 × 16384) maps them to $\mathbf u_t$ with 5120. This line is the
   same as MHA's (8).
9. **Why the cache can stop at $\mathbf c^{KV}$ (derivation mine, following
   Appendix C).** Write $W^{UQ}_i$, $W^{UK}_i$, $W^{UV}_i$ for the 128 rows of each
   matrix that produce head $i$, and $W^O_i$ for the 128 columns of $W^O$ that
   read head $i$. Then the content score is
   $(W^{UQ}_i\mathbf c_t^Q)^T(W^{UK}_i\mathbf c_j^{KV})=\mathbf c_t^{Q\,T}\,(W^{UQ\,T}_iW^{UK}_i)\,\mathbf c_j^{KV}$,
   and head $i$'s share of $\mathbf u_t$ is
   $W^O_i\mathbf o_{t,i}=(W^O_iW^{UV}_i)\sum_j a_j\mathbf c_j^{KV}$, where $a_j$
   are the softmax weights, so the sum is a weighted average of cached latents,
   512 numbers. The bracketed matrices (1536 × 512 and
   5120 × 512 per head) do not depend on any token, so they can be formed once.
   At inference, head $i$ computes $\mathbf c_t^{Q\,T}(W^{UQ\,T}_iW^{UK}_i)$ once
   per new token $t$: 1536 numbers times a 1536 × 512 matrix give a query of
   512 numbers (applying $W^{UK}_i$ to $\mathbf q_{t,i}^C$ instead, 128 numbers
   times 128 × 512, gives the same 512). That query is dotted directly with each
   cached $\mathbf c_j^{KV}$, also 512 numbers, giving the content score for
   token $j$. Attention then runs on the cached $\mathbf c_j^{KV}$ and $\mathbf k_j^R$
   alone, and $\mathbf k^C$, $\mathbf v^C$ are never built. This regrouping of
   a product of matrices is what the paper calls absorbing: "Fortunately, due to
   the associative law of matrix multiplication, we can absorb $W^{UK}$ into
   $W^{UQ}$, and $W^{UV}$ into $W^O$." (Appendix C). §2.1.2 instead says
   "$W^{UK}$ can be absorbed into $W^Q$"; with compressed queries (12)–(13) there
   is no $W^Q$, so Appendix C's wording is the one that matches the equations.
   Check 3 confirms both forms give the same $\mathbf u_t$.
10. **Why RoPE gets its own slice.** The paper's argument (§2.1.3): "If
    we apply RoPE for the keys $\mathbf k_t^C$, $W^{UK}$ in Equation 10 will be
    coupled with a position-sensitive RoPE matrix. In this way, $W^{UK}$ cannot
    be absorbed into $W^Q$ any more during inference, since a RoPE matrix
    related to the currently generating token will lie between $W^Q$ and
    $W^{UK}$ and matrix multiplication does not obey a commutative law. As a
    result, we must recompute the keys for all the prefix tokens during
    inference, which will significantly hinder the inference efficiency." In
    symbols (mine): with rotations $R_t$ and $R_j$ on the content query and key,
    the middle matrix of step 9 becomes $W^{UQ\,T}_iR_t^TR_jW^{UK}_i$. RoFormer
    (16) gives $R_t^TR_j=R_{j-t}$, so the matrix changes with the distance
    $j-t$, and there is no single matrix to form in advance (check 4). In
    (14)–(17), all rotation sits in the 64-number slices: $\mathbf k_j^R$ is
    rotated once at position $j$ and cached, $\mathbf q_{t,i}^R$ is rotated at
    position $t$, and their dot product depends on positions through $j-t$
    only, provided each head's slice of $\mathbf q_t^R$ is rotated with the same
    angles as $\mathbf k^R$ (an open choice; see "RoPE across heads in (14)"
    under "What the code must decide"). The 128-number content slices carry no rotation, so step 9 still
    applies to them.

**Kind of statement:** a computation. Each line defines a named intermediate
from earlier ones; the chain takes $\mathbf h_1,\dots,\mathbf h_t$ and returns
$\mathbf u_t$, a vector of $d=5120$ numbers. What to check is the shape of every
intermediate: steps 1–8 give them, and check 3 asserts them.

## Where the samples come from

Nothing here is sampled or averaged over data: the equations contain no
expectation, and the softmax weights are computed, not drawn. The only
question of origin is the cache: at inference, $\mathbf c_j^{KV}$ and
$\mathbf k_j^R$ for $j<t$ were computed by the same layer when token $j$ was
processed (Appendix C). The equations detach nothing.

## Check with small numbers

Check 1 builds one head by hand with $d_h=d_h^R=2$ so that the two scaled
scores are exactly $0$ and $\ln 3$ (weights $1/4$, $3/4$); the second key gets
half its score from the content slice and half from the RoPE slice, so both
terms of (16)–(18) count. Check 2 builds RoPE from RoFormer (15). Check 3 runs
(9)–(19) with random weights, 2 heads and 4 tokens (at least two items in every
sum and softmax) and compares it with the absorbed form of step 9 that reads
only the cache. Check 4 tests step 10, check 5 the open question in (14), and
check 6 recomputes the paper's cache numbers. Check 6 also counts DeepSeek
67B's cache. That model uses GQA (grouped-query attention), which shares one
key and one value among each group of heads (each shared pair is a "KV head"
in the code); Table 1 gives its cache as $2n_gd_hl$ elements per token, where
"$n_g$ denotes the number of groups in GQA". There are no min, max, clip or
indicator branches; the one cut-off, $j\le t$, is tested by appending a token.
Code positions count from 0.

```python
"""Small-number checks for MLA, DeepSeek-V2 (arXiv 2405.04434v5), equations (9)-(19)."""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

# 1. Exact example for (16)-(18): two keys, scaled scores (0, ln 3) -> weights (1/4, 3/4).
dh, dhR = 2, 2
ln3 = np.log(3.0)
qC, qR = np.array([ln3, 0.0]), np.array([1.0, 0.0])
kC = [np.array([0.0, 1.0]), np.array([1.0, 0.0])]
kR = [np.array([0.0, 1.0]), np.array([ln3, 0.0])]
vC = [np.array([4.0, 0.0]), np.array([0.0, 4.0])]
q = np.concatenate([qC, qR])  # (16)
k = [np.concatenate([kC[j], kR[j]]) for j in range(2)]  # (17)
raw = np.array([q @ kj for kj in k])
s = raw / np.sqrt(dh + dhR)  # (18), scale sqrt(d_h + d_h^R)
w = np.exp(s) / np.exp(s).sum()
o = w[0] * vC[0] + w[1] * vC[1]
print("1. raw dot products", raw, "scaled", s, "weights", w, "o", o)
assert np.allclose(raw, [0.0, 2 * ln3]) and np.allclose(s, [0.0, ln3])
assert np.allclose(w, [0.25, 0.75]) and np.allclose(o, [1.0, 3.0])
for j in range(2):  # dot of concatenations = content part + RoPE part
    assert np.isclose(q @ k[j], qC @ kC[j] + qR @ kR[j])
w_c_only = np.exp([0.0, ln3 / 2]) / np.exp([0.0, ln3 / 2]).sum()
w_dh = np.exp(raw / np.sqrt(dh)) / np.exp(raw / np.sqrt(dh)).sum()
print("   without RoPE part:", w_c_only, " scaled by sqrt(d_h) only:", w_dh)


# 2. RoPE as RoFormer 2104.09864v5 eq. (15): adjacent pairs, theta_i = 10000^(-2(i-1)/d).
def rope(x, pos, base=10000.0):
    d = x.shape[-1]
    theta = base ** (-2 * np.arange(d // 2) / d)
    c, s_ = np.cos(pos * theta), np.sin(pos * theta)
    out = np.empty_like(x)
    out[..., 0::2] = x[..., 0::2] * c - x[..., 1::2] * s_
    out[..., 1::2] = x[..., 0::2] * s_ + x[..., 1::2] * c
    return out


def rope_matrix(d, pos):
    return np.stack([rope(e, pos) for e in np.eye(d)], axis=1)


rng = np.random.default_rng(0)
a, b = rng.normal(size=4), rng.normal(size=4)
R = rope_matrix(4, 3.0)
assert np.allclose(R[:2, :2], [[np.cos(3), -np.sin(3)], [np.sin(3), np.cos(3)]])
assert np.allclose(R.T @ R, np.eye(4))  # orthogonal
assert np.isclose(rope(a, 2) @ rope(b, 7), rope(a, 12) @ rope(b, 17))  # only j - t matters
print("2. RoPE: (R_2 a).(R_7 b) =", round(rope(a, 2) @ rope(b, 7), 6),
      " (R_12 a).(R_17 b) =", round(rope(a, 12) @ rope(b, 17), 6))

# 3. Full MLA, (9)-(19), with random weights and 4 tokens; then the absorbed form.
d, nh, dh, dhR, dc, dcq, T = 8, 2, 4, 2, 3, 5, 4
W = {name: rng.normal(size=shape) for name, shape in {
    "DKV": (dc, d), "UK": (dh * nh, dc), "UV": (dh * nh, dc), "DQ": (dcq, d),
    "UQ": (dh * nh, dcq), "QR": (dhR * nh, dcq), "KR": (dhR, d), "O": (d, dh * nh)}.items()}
H = rng.normal(size=(T, d))  # row t is h_t


def heads(x, size):  # [x_1; ...; x_nh] = x
    return x.reshape(nh, size)


def project(h, t):
    cKV = W["DKV"] @ h  # (9)
    kC, vC = heads(W["UK"] @ cKV, dh), heads(W["UV"] @ cKV, dh)  # (10), (11)
    cQ = W["DQ"] @ h  # (12)
    qC = heads(W["UQ"] @ cQ, dh)  # (13)
    qR = rope(heads(W["QR"] @ cQ, dhR), t)  # (14), RoPE per head (my choice)
    kR = rope(W["KR"] @ h, t)  # (15), one key shared by all heads
    return cKV, kC, vC, cQ, qC, qR, kR


P = [project(H[t], t) for t in range(T)]
cKV, kC, vC, cQ, qC, qR, kR = P[1]
assert cKV.shape == (dc,) and kC.shape == vC.shape == (nh, dh) and cQ.shape == (dcq,)
assert qC.shape == (nh, dh) and qR.shape == (nh, dhR) and kR.shape == (dhR,)


def mla_naive(t, P):
    o = []
    for i in range(nh):
        q_ti = np.concatenate([P[t][4][i], P[t][5][i]])  # (16)
        k_ji = [np.concatenate([P[j][1][i], P[j][6]]) for j in range(t + 1)]  # (17), j <= t
        sc = np.array([q_ti @ kk for kk in k_ji]) / np.sqrt(dh + dhR)
        a_ = np.exp(sc - sc.max()) / np.exp(sc - sc.max()).sum()  # Softmax_j
        o.append(sum(a_[j] * P[j][2][i] for j in range(t + 1)))  # (18)
    return W["O"] @ np.concatenate(o)  # (19)


def mla_absorbed(t, cache, cQ_t, qR_t):  # cache holds only (c_j^KV, k_j^R)
    u = np.zeros(d)
    for i in range(nh):
        rows = slice(i * dh, (i + 1) * dh)
        A = W["UQ"][rows].T @ W["UK"][rows]  # d'_c x d_c, W^UK absorbed on the query side
        B = W["O"][:, rows] @ W["UV"][rows]  # d x d_c, W^UV absorbed into W^O
        sc = np.array([cQ_t @ A @ c + qR_t[i] @ kr for c, kr in cache]) / np.sqrt(dh + dhR)
        a_ = np.exp(sc - sc.max()) / np.exp(sc - sc.max()).sum()
        u += B @ sum(a_[j] * cache[j][0] for j in range(len(cache)))
    return u


for t in range(T):
    cache = [(P[j][0], P[j][6]) for j in range(t + 1)]
    assert np.allclose(mla_naive(t, P), mla_absorbed(t, cache, P[t][3], P[t][5]))
u2 = mla_naive(2, P)
print("3. u_3 naive   ", u2)
print("   u_3 absorbed", mla_absorbed(2, [(P[j][0], P[j][6]) for j in range(3)], P[2][3], P[2][5]))
assert np.allclose(mla_naive(2, P[:3]), u2)  # token 4 does not change u_3: sum stops at j = t
print("   cached numbers per token per layer:", dc + dhR, "vs MHA", 2 * nh * dh)

# 4. RoPE on k^C instead: the query-key matrix depends on the offset j - t.
Wq, Wk = W["UQ"][:dh], W["UK"][:dh]
M = lambda t, j: Wq.T @ rope_matrix(dh, t).T @ rope_matrix(dh, j) @ Wk
assert np.allclose(M(3, 1), M(2, 0)) and not np.allclose(M(2, 0), M(2, 1))
print("4. |M(2,0) - M(2,1)|max =", round(np.abs(M(2, 0) - M(2, 1)).max(), 4),
      " |M(3,1) - M(2,0)|max =", round(np.abs(M(3, 1) - M(2, 0)).max(), 12))

# 5. Eq. (14) applies RoPE to a d_h^R*n_h vector; per head vs whole vector differ.
x = W["QR"] @ P[1][3]
per_head, whole = rope(heads(x, dhR), 5).ravel(), rope(x, 5)
print("5. RoPE per head", per_head, " over whole vector", whole)
assert not np.allclose(per_head, whole)

# 6. The paper's cache numbers (Section 3.1.2 settings, Table 1, Table 9).
d_h, n_h, d_c, d_hR, l = 128, 128, 512, 64, 60
mla, mha = (d_c + d_hR) * l, 2 * n_h * d_h * l
print("6. MLA", d_c + d_hR, "per layer,", mla, "per token;  MHA same heads", 2 * n_h * d_h,
      "per layer,", mha, "per token  ratio", round(mla / mha, 4), " d_c/(d_h n_h)", d_c / (d_h * n_h))
assert d_c + d_hR == 4.5 * d_h and (d_c + d_hR) / (2 * d_h) == 2.25
print("   Table 9 ratios small", round(15.6 / 110.6, 3), "large", round(34.6 / 860.2, 3))
print("   576*27 =", 576 * 27, " 2*16*128*27 =", 2 * 16 * 128 * 27, " 576*60 =", 576 * 60,
      " 860.2K/(2*128*60) =", round(860.2e3 / (2 * 128 * 60), 2))
# DeepSeek 67B, DeepSeek LLM 2401.02954v1 Table 2: 95 layers, d_model 8192, 64 heads, 8 KV heads (GQA).
gqa = 2 * 8 * (8192 // 64) * 95  # a key and a value for each of 8 KV heads of 128, every layer
b67, bv2 = gqa * 16 / 8, mla * 6 / 8  # bytes: 67B at 16 bits (assumed), V2 at 6 bits (Section 3.2.3)
print("   67B", gqa, "elements/token, V2", mla, " reduction by elements", round(1 - mla / gqa, 4))
print("   bytes/token: 67B at 16 bits", b67, " V2 at 6 bits", bv2, " reduction", round(1 - bv2 / b67, 4))
assert gqa == 194560 and round(1 - bv2 / b67, 3) == 0.933
print("all checks passed")
```

Run with `uv run --no-project --with numpy python check.py`. Output:

```text
1. raw dot products [0.     2.1972] scaled [0.     1.0986] weights [0.25 0.75] o [1. 3.]
   without RoPE part: [0.366 0.634]  scaled by sqrt(d_h) only: [0.1746 0.8254]
2. RoPE: (R_2 a).(R_7 b) = 0.852898  (R_12 a).(R_17 b) = 0.852898
3. u_3 naive    [-22.8127  12.9883 -18.8204   6.7375  -2.2063 -23.2346   1.5921   3.2213]
   u_3 absorbed [-22.8127  12.9883 -18.8204   6.7375  -2.2063 -23.2346   1.5921   3.2213]
   cached numbers per token per layer: 5 vs MHA 16
4. |M(2,0) - M(2,1)|max = 1.3293  |M(3,1) - M(2,0)|max = 0.0
5. RoPE per head [  7.7725   4.1075 -11.0313   6.5055]  over whole vector [ 7.7725  4.1075 -8.9193 -9.1901]
6. MLA 576 per layer, 34560 per token;  MHA same heads 32768 per layer, 1966080 per token  ratio 0.0176  d_c/(d_h n_h) 0.03125
   Table 9 ratios small 0.141 large 0.04
   576*27 = 15552  2*16*128*27 = 110592  576*60 = 34560  860.2K/(2*128*60) = 56.0
   67B 194560 elements/token, V2 34560  reduction by elements 0.8224
   bytes/token: 67B at 16 bits 389120.0  V2 at 6 bits 25920.0  reduction 0.9334
all checks passed
```

Check 1 gives output $(1,3)=\tfrac14(4,0)+\tfrac34(0,4)$. Dropping the RoPE slice
changes the weights to $(0.366, 0.634)$, and dividing by $\sqrt{d_h}$ instead of
$\sqrt{d_h+d_h^R}$ changes them to $(0.175, 0.825)$, so both the position score
and the scale matter. Check 2 shows the RoPE score depends only on the distance
($7-2=17-12$). Check 3 shows the cache-only form of step 9 gives the same
output as the full computation for every token, and that a later token does not
change an earlier output. Check 4 shows that with RoPE on the content keys the
query–key matrix is the same for equal distances (offset −2) but different for
different distances, which is the paper's reason for decoupling.

## What the code must decide

The paper links https://github.com/deepseek-ai/DeepSeek-V2 for checkpoints (Abstract).
On 1 November 2026 its root held a README, two licences, `figures/` and the
report PDF, with no tags and no model code. The README points to Hugging Face
model repositories; the paper does not link those, so I did not open them. Every
choice below is therefore **open**.

- **RoPE details:** frequency base, which coordinates form a pair (RoFormer
  rotates adjacent pairs), and whether positions count from 0 or 1. The start
  does not change any score, because only $j-t$ enters (check 2), as long as
  queries and cached keys use the same count.
- **RoPE across heads in (14):** the same angles for each 64-number head slice,
  or one rotation over all 8192 numbers. The results differ (check 5), and by
  my derivation only the first meets the condition in step 10.
- **Long context:** "YaRN was specifically applied to the decoupled shared key
  $\mathbf k_t^R$ as it is responsible for carrying RoPE (Su et al., 2024)."
  (§3.1.4). YaRN is the method the paper uses to extend the context window
  from 4K to 128K tokens, with "the scale $s$ to 40" and "the target maximum
  context length to 160K". DeepSeek-V2 never defines $s$ (I searched the full
  TeX); it is YaRN's setting, which
  [YaRN](https://arxiv.org/abs/2309.00071v2) v2 defines in §2.3, equation (11):
  "The ratio between the extended context length and the original context
  length … we introduce the notation $s$ defined by $s=L'/L$ … and we call $s$
  the scale factor". The paper also adjusts
  YaRN's "length scaling factor": "The factor $\sqrt t$ is computed as
  $\sqrt t=0.0707\ln s+1$" (§3.1.4). In that formula, $t$ is YaRN's factor,
  not a token position. The paper names only the key; how the queries' rotation
  and this factor enter (18) is open.
- **Extra RMS Norm and scaling factors:** §3.1.2 says "in practice, we employ
  additional RMS Norm layers after the compressed latent vectors, and multiply
  additional scaling factors at the width bottlenecks (i.e., the compressed
  latent vectors and the intermediate hidden states of routed experts) to
  ensure stable training". §2.1.2 gives the name "compressed latent vector" to
  exactly two vectors, $\mathbf c_t^{KV}$ and $\mathbf c_t^Q$ (table above), so
  each gets an RMS Norm after it and a scaling factor. Open: whether the cache
  holds $\mathbf c_t^{KV}$ before or after the norm, the norm's weights, and the
  scale values.
- **Biases:** (9)–(19) have none.
- **Absorbed or not, and when:** the paper describes absorbing for inference;
  whether training builds $\mathbf k^C$, $\mathbf v^C$ explicitly is not said.
  Also open: whether the absorbed matrices of step 9 are stored, or $W^{UK}_i$ is
  applied to the query at run time; the two orders cost different arithmetic.
- **Attention kernel and stability:** "MLA is also optimized based on an
  improved version of FlashAttention-2 (Dao, 2023)" (§3.1.3), with no details.
  The kernel must take 192-number queries and keys with 128-number values, and
  scale by $1/\sqrt{192}$. My check subtracts the maximum score before the
  exponential; the paper does not say what its kernel does.
- **Masking:** the limit $j\le t$ is the causal mask. Every row includes $j=t$,
  so no row is fully masked; the masking value, and padding when sequences of
  different lengths share a batch, are open.
- **Cache storage:** the deployed model compresses "each element in its KV cache
  into 6 bits on average" (§3.2.3); the format is not given.
- **DeepSeek-V2-Lite**, the smaller MLA model the paper also releases (§1,
  Appendix B.1): it "does not compress the queries" (Appendix B.1), so
  (12)–(14) cannot apply as printed; the paper does not print what feeds
  $W^{QR}$ there.

## What the paper claims and what it shows

- **Cache of $(d_c+d_h^R)l$ elements per token**, where "$l$ denotes the number
  of layers" (§2.1.2): follows from the definitions, since only
  $\mathbf c_t^{KV}$ and $\mathbf k_t^R$ are kept per layer (§2.1.3, Appendix C).
  Recomputed in check 6: 576 per layer, 34,560 per token for 60 layers. An MHA
  model with V2's 60 layers of 128 heads of 128 would hold 32,768 per layer and
  1,966,080 per token (1.76% ratio);
  that comparison model is my arithmetic, not a model the paper trained. Table 1
  counts "the number of elements, regardless of the storage precision".
- **"$\approx\frac92d_hl$" and "equal to GQA with only 2.25 groups"** (Table 1
  and caption): arithmetic from the stated settings, with GQA's count
  $2n_gd_hl$ as explained before the check. With $d_c=4d_h$ and $d_h^R=d_h/2$ the
  cache is exactly $\frac92d_hl$, and $576/(2\cdot128)=2.25$ (check 6).
- **Absorbing $W^{UK}$ and $W^{UV}$ removes the need to compute keys and values
  at inference:** argued from associativity (Appendix C; step 9). The identity
  holds in check 3. The paper reports no measurement of the compute saved.
- **RoPE on the content keys would force recomputing all prefix keys and
  "significantly hinder the inference efficiency":** argued (§2.1.3; step 10),
  not measured. Check 4 illustrates the argument.
- **Query compression reduces activation memory in training:** a stated
  motivation, "in order to reduce the activation memory during training"
  (§2.1.2). I found no measurement of it in the paper.
- **MLA performs better than MHA with a much smaller cache:** measured on
  mixture-of-experts (MoE) models, Table 9 (Appendix D.2). "Two small MoE models comprise about 16B total
  parameters, and we train them on 1.33T tokens. Two large MoE models comprise
  about 250B total parameters, and we train them on 420B tokens." MLA scores
  higher on 7 of 8 benchmark cells and lower on C-Eval for the small models
  (50.9 against 51.6). The table gives no ± and no number of runs, and the
  paired models differ in activated parameters (2.5B against 2.4B; 25.0B
  against 21.5B). §2.1 points to this appendix for its comparison with MHA. The
  cache columns give the paper's "14% for small MoE models and 4% for large MoE
  models": $15.6/110.6=0.141$ and $34.6/860.2=0.040$ (check 6). My inference,
  not stated: 15.6K and 110.6K match 27 layers with 16 heads of 128 (Lite's
  configuration, Appendix B.1), 34.6K matches 60 layers, and 860.2K would mean
  56 heads of 128 at 60 layers; the paper gives none of these baseline
  configurations.
- **MHA beats GQA and MQA** (MQA shares one key and value among all heads):
  measured on 7B dense models trained on 1.33T tokens, Table 8 (Appendix D.1).
  §2.1 cites it for "their performance does not match MHA".
- **"reduces the KV cache by 93.3%" compared with DeepSeek 67B** (Abstract,
  Figure 1(b), §1, §5): Figure 1(b) plots it as "KV Cache for Generation
  (KB/Token)", a size in bytes, not in elements. The paper does not say how
  this figure was measured or at what precision. My recomputation (check 6)
  takes DeepSeek 67B's settings from the paper DeepSeek-V2 cites for it,
  [DeepSeek LLM](https://arxiv.org/abs/2401.02954v1), v1: Table 2 gives 95
  layers, $d_{\mathrm{model}}=8192$, 64 heads and 8 KV heads, and §2.2 says
  "the 67B model uses Grouped-Query Attention (GQA)". With the GQA count
  $2n_gd_hl$ above, $n_g=8$, and heads $8192/64=128$ numbers wide (my
  assumption: Table 2 gives no head size), DeepSeek 67B caches
  $2\cdot8\cdot128\cdot95=194{,}560$ elements per token against V2's 34,560, an
  82.2% reduction in elements. Counting bytes, with V2's cache at the 6 bits
  per element of §3.2.3 and DeepSeek 67B's at 16 bits, the reduction is 93.3%
  (0.9334). Neither paper states the 16 bits; it is my assumption. By my
  inference, then, the 93.3% includes the cache quantization and is not MLA's
  alone.
- **5.76 times the maximum generation throughput of DeepSeek 67B** (§3.2.3):
  measured on the deployed system, which also converts weights to FP8 and
  quantizes the cache to about 6 bits per element, so the gain is not MLA's
  alone. Throughput is for generation, producing tokens one at a time (the
  decode phase), "On a single node with 8 H800 GPUs", using the prompt and
  generation lengths of the deployed DeepSeek 67B service. The paper does not
  print the inputs needed to recompute it.

## Sources

- DeepSeek-AI, DeepSeek-V2, https://arxiv.org/abs/2405.04434v5 (TeX source and
  PDF; Abstract, Figure 1(b), §1, §2.1.1–2.1.4, Table 1, Figure 2, §3.1.2–3.1.4,
  §3.2.3, §5, Appendices B.1, C, D.1, D.2).
- Su et al., RoFormer: Enhanced Transformer with Rotary Position Embedding,
  https://arxiv.org/abs/2104.09864v5 (§3.2.2, equations (14)–(16)).
- DeepSeek-AI, DeepSeek LLM: Scaling Open-Source Language Models with
  Longtermism, https://arxiv.org/abs/2401.02954v1 (the only version; Table 2,
  §2.2), cited by DeepSeek-V2 as DeepSeek-AI (2024) for DeepSeek 67B.
- Peng et al., YaRN: Efficient Context Window Extension of Large Language
  Models, https://arxiv.org/abs/2309.00071v2 (§2.3, equation (11)); v2
  (1 November 2023) is the latest version before DeepSeek-V2 v5, and v3
  (6 February 2026) came later.
- PyTorch 2.15 documentation,
  https://docs.pytorch.org/docs/2.15/generated/torch.nn.Linear.html and
  https://docs.pytorch.org/docs/2.15/generated/torch.cat.html, opened
  1 November 2026.
- https://github.com/deepseek-ai/DeepSeek-V2, root listing and tags page, opened
  1 November 2026 (no code, no tags).
- Cited by the paper, not opened: Vaswani et al. (2017), Attention Is All You
  Need; Shazeer (2019), MQA; Ainslie et al. (2023), GQA; Dao (2023),
  FlashAttention-2. Not opened: the Hugging Face model
  repositories linked from the GitHub README.
- Check script: the code block above, run with
  `uv run --no-project --with numpy`.
