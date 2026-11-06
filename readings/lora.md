# LoRA: a frozen layer plus a small trainable low-rank correction

**Source:** Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li,
Shean Wang, Lu Wang, Weizhu Chen,
[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685v2),
v2 (16 Oct 2021), §4.1, equation (3). This reading is pinned to v2; every section,
equation, table and footnote number below was read from the v2 PDF.
Other locations used: the abstract; the "Terminologies and Conventions" paragraph at the
end of §1; §2, equations (1) and (2); Figure 1; footnote 2; §4.2 and footnotes 4 and 5;
§5.1; §5.2; §7.1; Appendix D (D.1, D.2, Tables 9 to 12); Table 15. Official code: `lora/layers.py`, `lora/utils.py`, `README.md` and the setup files
listed under Sources, in [microsoft/LoRA](https://github.com/microsoft/LoRA/tree/8f4fdeb0aa05df855a9ffb359a6faa5ee48a4878)
at commit `8f4fdeb0` (16 Sep 2021; Sources says why this commit).

## What it computes and what it changes

Equation (3) is the forward pass of one dense (fully connected) layer after LoRA is
attached. The layer keeps its pre-trained weight matrix $W_0$ unchanged and adds a second
path: squeeze the input down to $r$ numbers with $A$, expand them back with $B$, and add
the result to the layer's original output. Only $A$ and $B$ are trained, so each task
needs only these small matrices, not a full copy of the model.

The earlier method is full fine-tuning, §2 equation (1): continue training _every_
parameter on the task's data. For one weight matrix, that means learning a full update
$\Delta W$ with as many numbers as $W_0$ itself ($d\cdot k$: $W_0$ has $d$ rows, one per
output, and $k$ columns, one per input). Equation (2) restates the
goal with the update produced by a smaller parameter set $\Theta$; equation (3) is the
paper's choice for producing it. $\Delta W$ is never stored or trained as its own matrix:
it is the product $BA$ of a $d\times r$ and an $r\times k$ matrix ($d\times r$ means $d$
rows and $r$ columns), and $W_0$ gets no
updates. The trained numbers per matrix drop from $d\cdot k$ to $r(d+k)$.

## The equation

$$
h = W_0 x + \Delta W x = W_0 x + BA x \qquad (3)
$$

The paragraph after (3) multiplies the second term by a constant; step 7 below shows
that form.

## Symbols

Notation used in the table:

- $\mathbb R^{d\times k}$ is a grid of real numbers with $d$ rows and $k$ columns, a
  tensor of shape `[d, k]`.
- A letter written to the right of a matrix, as in $W_0x$, is a column vector.
- The _rank_ of a matrix is the number of independent directions its outputs can point
  in. Writing a matrix as a tall, thin matrix times a short, wide one is a _low-rank
  decomposition_; step 2 below shows why such a product has a small rank.
- $\ll$ means "much smaller than".

| Symbol | Shape or type | Meaning | Role | Defined at |
| ------ | ------------- | ------- | ---- | ---------- |
| $h$ | $d$-vector, column (shape inferred, see below) | the layer's output once LoRA is added | computed intermediate | "For $h = W_0x$, our modified forward pass yields:" (§4.1, the sentence before (3)) |
| $W_0$ | $d\times k$ matrix (its layout in code: "Rows or columns" under "What the code must decide") | the pre-trained weight matrix of one dense layer | frozen model | "For a pre-trained weight matrix $W_0\in\mathbb R^{d\times k}$" and "During training, $W_0$ is frozen and does not receive gradient updates" (§4.1); "$W$ or $W_0$ refers to a pre-trained weight matrix" (§1, Terminologies and Conventions) |
| $x$ | $k$-vector, column (shape inferred) | the input to this dense layer, fed to both paths | function input: the activations that reach the layer, e.g. the input to $W_q$ or $W_v$, two of the "four weight matrices in the self-attention module" (§4.2) | "Note both $W_0$ and $\Delta W=BA$ are multiplied with the same input, and their respective output vectors are summed coordinate-wise." (§4.1). This is not the $x$ of §2, where "both $x_i$ and $y_i$ are sequences of tokens" |
| $\Delta W$ | $d\times k$ matrix | the change to $W_0$ learned for the task; in LoRA always the product $BA$ | computed intermediate | "$\Delta W$ its accumulated gradient update during adaptation" (§1, Terminologies and Conventions); "$W_0+\Delta W=W_0+BA$" (§4.1) |
| $B$ | $d\times r$ matrix | the "expand" factor: turns $r$ numbers into $d$ outputs; starts at zero under the usual initialization (step 7) | trained weight | "where $B\in\mathbb R^{d\times r}$, $A\in\mathbb R^{r\times k}$" and "while $A$ and $B$ contain trainable parameters" (§4.1); "zero for $B$" (§4.1, after (3)) |
| $A$ | $r\times k$ matrix | the "squeeze" factor: turns the $k$ inputs into $r$ numbers; starts random | trained weight | the same two sentences; "We use a random Gaussian initialization for $A$" (§4.1, after (3)) |
| $r$ | positive integer | the rank of the update: how many numbers the squeeze keeps | fixed setting (hyperparameter) | "and the rank $r\ll\min(d,k)$" (§4.1); "We use $r$ to denote the rank of a LoRA module." (§1, Terminologies and Conventions) |
| $d$, $k$ | positive integers | rows (outputs) and columns (inputs) of $W_0$ | fixed setting, given by the architecture | only through "$W_0\in\mathbb R^{d\times k}$" (§4.1). For the attention matrices: "We treat $W_q$ (or $W_k$, $W_v$) as a single matrix of dimension $d_{model}\times d_{model}$" (§4.2), so $d=k=d_{model}$; for GPT-3 175B, "the full rank (i.e., $d$) is as high as 12,288" (§1) |
| $\alpha$ | scalar | not in (3) as printed; sets the multiplier $\alpha/r$ on the LoRA path | fixed setting | "We then scale $\Delta Wx$ by $\frac{\alpha}{r}$, where $\alpha$ is a constant in $r$." (§4.1, after (3)). My reading of "a constant in $r$": $\alpha$ stays fixed when $r$ changes |

**Not defined in the paper:**

- The shapes of $x$ and $h$. The paper never writes $x\in\mathbb R^k$ or $h\in\mathbb R^d$,
  and never says whether $x$ is one token's vector or a batch of them. I inferred the
  shapes from $W_0\in\mathbb R^{d\times k}$. I looked in §1 (Terminologies and
  Conventions), §2, §4.1, §4.2 and Figure 1. Figure 1 draws a square $W\in\mathbb R^{d\times d}$.
- A bias term. Equation (3) has no bias term. §4.1 mentions biases only as optional
  extra trainable parameters ("training all biases", with footnote 2: "They represent a
  negligible number of parameters compared to weights"); §4.2 leaves "adapting the MLP
  layers, LayerNorm layers, and biases to a future work".
- $\sigma$ in Figure 1's "$A=\mathcal N(0,\sigma^2)$". It is outside (3), but it sets
  $A$'s starting values. I found no definition in §4.1, Appendix D or the TeX source,
  where `sigma` appears only in Appendix G's subspace-similarity formulas.

## Read from the inside out

1. **$Ax$.** $A$ is $r\times k$ and $x$ has $k$ entries, so the result has $r$ entries:
   $(Ax)_j=\sum_{i=1}^{k}A_{ji}x_i$ for $j=1,\dots,r$. The sum runs over the $k$ input
   coordinates. This is the squeeze.
2. **$B(Ax)$.** $B$ is $d\times r$, so the $r$ numbers become $d$ numbers:
   $(BAx)_m=\sum_{j=1}^{r}B_{mj}(Ax)_j$. The sum runs over the $r$ squeezed coordinates.
   The paper prints $BAx$ without brackets because a chain of matrix products can be
   grouped either way: $(BA)x=B(Ax)$. On its own, $BA$ is a $d\times k$ matrix, the same
   shape as $W_0$, and it is $\Delta W$. Its rank is at most $r$ for any values of $A$
   and $B$, because every output is a combination of $B$'s $r$ columns (standard linear
   algebra, checked below).
3. **$W_0x$.** $d\times k$ times $k$ gives $d$ numbers; the sum runs over the $k$ input
   coordinates. This is the unchanged pre-trained path.
4. **$+$.** Two $d$-vectors are added entry by entry, as the §4.1 quote in the $x$ row
   says.
5. **The two $=$ signs.** The second, $\Delta Wx=BAx$, substitutes the definition
   $\Delta W=BA$. For the first, the paper sets $h=W_0x$ for the layer before LoRA, then
   reuses the name $h$ for the modified layer (the $h$ row's quote). Read it like reassigning `h` in
   code: the new layer outputs $W_0x+\Delta Wx$. It does not claim that $W_0x$ equals
   $W_0x+\Delta Wx$.
6. **Marks.** The subscript $0$ in $W_0$ is a label meaning "pre-trained", not an index
   or a power. $\Delta$ is part of the name $\Delta W$ (a change); it does not multiply
   $W$. The superscripts in $\mathbb R^{d\times k}$ give shapes. Equation (3) has no
   transposes and no powers.
7. **The scale added after (3).** The scaling by $\frac{\alpha}{r}$ quoted in the
   $\alpha$ row turns the layer into $h=W_0x+\frac{\alpha}{r}BAx$. The official code
   computes this form plus the layer's bias, if it has one, with dropout on the LoRA
   path, which randomly zeroes its inputs during training, off by default
   (`lora/layers.py` lines 22 to 26, 100, 114 and 152 to 154, commit `8f4fdeb0`; details under
   "What the code must decide"). With §4.1's initialization, which Appendix D calls "the usual
   initialization", $B$ starts at zero, so "$\Delta W=BA$ is zero at the beginning of
   training" and before the first step the layer gives exactly the pre-trained output.
   Some of the paper's runs start elsewhere (see "How $A$ starts" below).

The paper writes column vectors, as in $W_0x$. With each input stored as a _row_ instead,
the same numbers come out transposed: $W_0x$ becomes $xW_0^\top$ and $BAx$ becomes
$xA^\top B^\top$, since transposing a product reverses its order. "Rows or columns" under
"What the code must decide" gives the layout the official code uses.

**Kind of statement:** a computation, the forward pass of one layer, so the check is
the shape of every intermediate (steps 1 to 4). Inside it, $\Delta W=BA$ is a definition
that also works as a constraint. The paper says "we constrain its update by representing
the latter with a low-rank decomposition". The factorization bounds the rank by
construction (step 2), so no projection or penalty is needed to enforce it.

## Where the samples come from

Equation (3) has no expectation or average. It is a fixed function of $x$, $W_0$, $A$
and $B$, and it is not the training objective. What $A$ and $B$ learn comes from the
task's objective, which LoRA does not fix: "While our proposal is agnostic to training
objective, we focus on language modeling as our motivating use case" (§2).

- **Data.** In the language-modeling setting the objective is equation (2), maximized
  over $\Theta$, here the entries of the LoRA matrices (counted under "Parameter counts
  and storage"). It sums over $\mathcal Z$, the dataset of context-target pairs. The
  RoBERTa and DeBERTa runs train on other tasks: "we evaluate on the GLUE (Wang et al.,
  2019) benchmark for RoBERTa and DeBERTa" (§5). Either way, the values of $x$ are
  activations computed from training data collected before training, not generated by
  the model.
- **Randomness.** $A$'s starting values are drawn once, at initialization, and dropout
  (step 7), when switched on, adds randomness during training. "How $A$ starts" and "Dropout on the LoRA
  path" under "What the code must decide" give the paper's and the code's choices.
- **Frozen, not detached.** A stop-gradient passes a value forward but blocks every
  gradient through it. In PyTorch it is `.detach()`, which "Returns a new Tensor,
  detached from the current graph. The result will never require gradient."
  ([PyTorch 1.9.0 `Tensor.detach`](https://github.com/pytorch/pytorch/blob/v1.9.0/torch/_tensor.py#L326-L329)).
  The paper writes no stop-gradient (I searched the v2 PDF text and the TeX for "stop",
  "detach" and "sg"). Freezing only stops $W_0$ from being updated; "Freezing $W_0$"
  below gives the code line. PyTorch's autograd notes give the `requires_grad` flag as
  the way to "freeze parts of your pretrained model during model fine-tuning", and say
  that "only leaf tensors with `requires_grad=True` will have gradients accumulated into
  their `.grad` fields", where leaf tensors include "a `nn.Module`'s parameters"
  ([PyTorch 1.9.0 autograd mechanics, "Setting `requires_grad`"](https://github.com/pytorch/pytorch/blob/v1.9.0/docs/source/notes/autograd.rst?plain=1#L56-L80)).
  The gradient still flows through $W_0x$ back to $x$, and from there to earlier layers'
  $A$ and $B$ (my derivation; check part 7 confirms that the gradient with respect to $x$
  has a part through $W_0$).
- **A consequence of the zero start (step 7)** (my derivation, checked below). For the
  training loss $L$ and its gradient $g=\partial L/\partial h$ with respect to the
  layer's output, the gradient reaching $A$ is
  $\frac{\alpha}{r}B^\top g\,x^\top$, which is exactly zero while $B=0$. The gradient
  reaching $B$ is $\frac{\alpha}{r}\,g\,(Ax)^\top$, which is not. So at the first step of
  a run that starts from $B=0$, only $B$ gets a nonzero gradient.

## Check with small numbers

The sizes are $d=3$, $k=4$, $r=2$. Every sum then has at least two terms, $d\ne k$ makes
a transposed matrix fail loudly, and $r<\min(d,k)$ makes the update genuinely low-rank.
Integer entries give exact answers. Equation (3) has no min, max, clip or indicator; the
cases that act as branches are covered instead: $B=0$ versus a trained $B$, unscaled
versus scaled by $\alpha/r$, merged weights ($W_0+\frac{\alpha}{r}BA$ stored as one
matrix) versus the two-path layer, and column versus row layout. The same script recomputes the parameter counts and sizes the paper prints.

```python
import numpy as np

# Toy sizes: d=3 outputs, k=4 inputs, rank r=2 (< min(d, k) = 3); two items in every sum.
W0 = np.array([[1, 0, 2, 0], [0, 1, 0, -1], [1, 1, 1, 1]], dtype=float)  # d x k, frozen
A = np.array([[1, 0, -1, 0], [0, 2, 0, 1]], dtype=float)                 # r x k, trained
B = np.array([[1, 0], [0, 1], [1, -1]], dtype=float)                     # d x r, trained
x = np.array([1, 2, 3, 4], dtype=float)                                   # k-vector input
d, k = W0.shape
r = A.shape[0]

# 1. Equation (3) as printed, innermost first.
Ax = A @ x                      # r-vector
BAx = B @ Ax                    # d-vector
W0x = W0 @ x                    # d-vector
h = W0x + BAx                   # coordinate-wise sum
print("Ax =", Ax, " BAx =", BAx, " W0x =", W0x, " h =", h)
assert Ax.tolist() == [-2, 8] and BAx.tolist() == [-2, 8, -10]
assert W0x.tolist() == [7, -2, 10] and h.tolist() == [5, 6, 0]
assert np.array_equal((B @ A) @ x, B @ (A @ x))  # (BA)x == B(Ax)

# 2. At initialization B = 0, so Delta W = BA = 0 and the layer is unchanged.
B_init = np.zeros((d, r))
assert np.array_equal(W0 @ x + B_init @ (A @ x), W0x)

# 3. The scaling added after (3): h = W0 x + (alpha/r) B A x.
alpha = 4.0
s = alpha / r
h_scaled = W0x + s * BAx
print("alpha/r =", s, " h_scaled =", h_scaled)
assert h_scaled.tolist() == [3, 14, -10]

# 4. Code layout: tokens are rows, as in nn.Linear (x @ W.T). Two tokens.
X = np.stack([x, np.array([0, 1, 0, -1], dtype=float)])          # [tokens, k]
H_rows = X @ W0.T + (X @ A.T @ B.T) * s                           # [tokens, d]
H_cols = np.stack([W0 @ xi + s * (B @ (A @ xi)) for xi in X])
assert np.array_equal(H_rows, H_cols)
print("H (rows = tokens) =\n", H_rows)

# 5. Merge for deployment, then unmerge to switch tasks.
W = W0 + s * (B @ A)
assert np.array_equal(X @ W.T, H_rows)
assert np.array_equal(W - s * (B @ A), W0)

# 6. Rank of the update is at most r; W0 itself is full rank here.
print("rank W0 =", np.linalg.matrix_rank(W0), " rank BA =", np.linalg.matrix_rank(B @ A),
      " rank of a rank-1 update =", np.linalg.matrix_rank(B[:, :1] @ A[:1]))
assert np.linalg.matrix_rank(B @ A) <= r and np.linalg.matrix_rank(B[:, :1] @ A[:1]) == 1

# 7. Gradients of a linear loss L = g . h (g = upstream gradient), with finite differences.
g = np.array([1, -1, 2], dtype=float)
def loss(A_, B_):
    return g @ (W0 @ x + s * (B_ @ (A_ @ x)))
def num_grad(f, M, eps=1e-3):
    G = np.zeros_like(M)
    for idx in np.ndindex(M.shape):
        P, Q = M.copy(), M.copy()
        P[idx] += eps; Q[idx] -= eps
        G[idx] = (f(P) - f(Q)) / (2 * eps)
    return G
for Bc, name in [(B, "trained B"), (B_init, "B = 0")]:
    gA = s * np.outer(Bc.T @ g, x)          # dL/dA = s * B^T g x^T
    gB = s * np.outer(g, A @ x)             # dL/dB = s * g (Ax)^T
    assert np.allclose(gA, num_grad(lambda M: loss(M, Bc), A))
    assert np.allclose(gB, num_grad(lambda M: loss(A, M), Bc))
    print(f"{name}: |dL/dA| = {np.abs(gA).sum():g}, |dL/dB| = {np.abs(gB).sum():g}")
assert np.all(s * np.outer(B_init.T @ g, x) == 0)
# Frozen W0 gets no update, but the gradient still flows through it to the input x.
gx = W0.T @ g + s * A.T @ (B.T @ g)
assert np.allclose(gx, num_grad(lambda v: g @ (W0 @ v + s * (B @ (A @ v))), x))
print("dL/dx =", gx, " of which through W0:", W0.T @ g)

# 8. Trainable numbers: full Delta W has d*k, LoRA has r*(d+k); fewer iff r < dk/(d+k).
print("toy: d*k =", d * k, " r*(d+k) =", r * (d + k), " with r=1:", d + k,
      " dk/(d+k) =", round(d * k / (d + k), 2))
assert (d * k, r * (d + k), d + k) == (12, 14, 7)  # r=2 is not smaller here, r=1 is
assert 1 < d * k / (d + k) < 2
for n, rk in [(4096, 8), (12288, 4)]:  # square d = k = n
    print(f"d=k={n}, r={rk}: d*k = {n * n:,}  r*(d+k) = {rk * 2 * n:,}  ratio = {n * n // (rk * 2 * n)}")
assert (4096 * 4096, 8 * 2 * 4096) == (16_777_216, 65_536)

dm, L = 12288, 96  # GPT-3 175B: d_model (Sec. 1) and layers (Sec. 7.1)
def lora_params(r_, n_types):          # |Theta| = 2 * L_hat * d_model * r (Sec. 5.1)
    return 2 * (n_types * L) * dm * r_
counts = {"rq=rv=1": lora_params(1, 2), "rv=2": lora_params(2, 1), "rq=rv=4": lora_params(4, 2),
          "rq=rv=8": lora_params(8, 2), "rq=rk=rv=ro=2": lora_params(2, 4),
          "rq=rk=rv=ro=4": lora_params(4, 4)}
for kname, v in counts.items():
    print(f"{kname:15s} {v:>12,d} = {v / 1e6:.2f}M")
assert counts["rq=rv=1"] == counts["rv=2"] == 4_718_592
assert counts["rq=rv=8"] == counts["rq=rk=rv=ro=4"] == 37_748_736
assert counts["rq=rv=4"] == counts["rq=rk=rv=ro=2"] == 18_874_368
assert dm * dm == 150_994_944 and 2 * dm * 4 == 98_304  # one d x d matrix vs its r=4 factors

fp16 = 2  # bytes
mb = counts["rq=rv=4"] * fp16
print(f"r=4 on Wq,Wv in FP16: {mb / 1e6:.1f} MB = {mb / 2**20:.1f} MiB")
print(f"350 GB / 35 MB = {350e9 / 35e6:,.0f};  175e9 params x 2 B = {175e9 * fp16 / 1e9:.0f} GB")
print(f"175e9 / 18,874,368 params = {175e9 / counts['rq=rv=4']:,.0f}")
print(f"footnote 4: 350 GB + 100 x 35 MB = {(350e9 + 100 * 35e6) / 1e9:.1f} GB;"
      f" 100 x 350 GB = {100 * 350e9 / 1e12:.0f} TB")

# 9. Multiply-adds for one GPT-3-sized matrix with r = 4: per token, and forming BA once per call.
rr = 4
print("per token: W0 x:", dm * dm, " B(Ax):", rr * dm + dm * rr, "  once per call, forming BA:", dm * rr * dm)

# 10. alpha/r in the paper's hyperparameter tables (Tables 9-11).
for nm, a_, r_ in [("RoBERTa base", 8, 8), ("RoBERTa large", 16, 8), ("DeBERTa XXL", 8, 8), ("GPT-2", 32, 4)]:
    print(f"{nm:14s} alpha={a_:2d} r={r_} alpha/r={a_ / r_:g}")

# 11. GPT-3 175B training figures in Sec. 4.2: footnote 5 throughputs, and the VRAM figure.
ft, lo = 32.5, 43.1  # tokens/s per V100 GPU, full fine-tuning and LoRA
gain, cut = (lo / ft - 1) * 100, (1 - ft / lo) * 100
print(f"footnote 5: throughput +{gain:.1f}%, time per token -{cut:.1f}%")
ratio, red = 1.2e12 / 350e9, (1 - 350e9 / 1.2e12) * 100
print(f"VRAM 1.2TB -> 350GB: ratio {ratio:.2f}x, reduction {red:.1f}% (2/3 = {200 / 3:.1f}%)")
assert (round(gain, 1), round(cut, 1), round(ratio, 2), round(red, 1)) == (32.6, 24.6, 3.43, 70.8)
print("all checks passed")
```

Run with `timeout 60 uv run --no-project --with "numpy>=2,<3" python -I check.py`. Output:

```text
Ax = [-2.  8.]  BAx = [ -2.   8. -10.]  W0x = [ 7. -2. 10.]  h = [5. 6. 0.]
alpha/r = 2.0  h_scaled = [  3.  14. -10.]
H (rows = tokens) =
 [[  3.  14. -10.]
 [  0.   4.  -2.]]
rank W0 = 3  rank BA = 2  rank of a rank-1 update = 1
trained B: |dL/dA| = 120, |dL/dB| = 80
B = 0: |dL/dA| = 0, |dL/dB| = 80
dL/dx = [  9. -11.  -2.  -3.]  of which through W0: [3. 1. 4. 3.]
toy: d*k = 12  r*(d+k) = 14  with r=1: 7  dk/(d+k) = 1.71
d=k=4096, r=8: d*k = 16,777,216  r*(d+k) = 65,536  ratio = 256
d=k=12288, r=4: d*k = 150,994,944  r*(d+k) = 98,304  ratio = 1536
rq=rv=1            4,718,592 = 4.72M
rv=2               4,718,592 = 4.72M
rq=rv=4           18,874,368 = 18.87M
rq=rv=8           37,748,736 = 37.75M
rq=rk=rv=ro=2     18,874,368 = 18.87M
rq=rk=rv=ro=4     37,748,736 = 37.75M
r=4 on Wq,Wv in FP16: 37.7 MB = 36.0 MiB
350 GB / 35 MB = 10,000;  175e9 params x 2 B = 350 GB
175e9 / 18,874,368 params = 9,272
footnote 4: 350 GB + 100 x 35 MB = 353.5 GB; 100 x 350 GB = 35 TB
per token: W0 x: 150994944  B(Ax): 98304   once per call, forming BA: 603979776
RoBERTa base   alpha= 8 r=8 alpha/r=1
RoBERTa large  alpha=16 r=8 alpha/r=2
DeBERTa XXL    alpha= 8 r=8 alpha/r=1
GPT-2          alpha=32 r=4 alpha/r=8
footnote 5: throughput +32.6%, time per token -24.6%
VRAM 1.2TB -> 350GB: ratio 3.43x, reduction 70.8% (2/3 = 66.7%)
all checks passed
```

By hand: $Ax=(1-3,\;4+4)=(-2,8)$, $B(Ax)=(-2,8,-2-8)=(-2,8,-10)$,
$W_0x=(1+6,\;2-4,\;1+2+3+4)=(7,-2,10)$, so $h=(5,6,0)$. The merged matrix gives the
same outputs as the two-path layer, and subtracting $\frac{\alpha}{r}BA$ recovers $W_0$
exactly. LoRA trains fewer numbers than a full update once $r(d+k)<dk$, that is
$r<dk/(d+k)$ (my derivation). At the toy size that is 1.71, so $r=2$ trains more (14
against 12) and $r=1$ already saves (7 against 12). The saving becomes large only when
$r\ll\min(d,k)$: 256 times at $d=k=4096$, $r=8$ (a size I chose), and 1,536 times for
one GPT-3 matrix with $r=4$.

## What the code must decide

The official repository is the one linked in the abstract. All lines cited below are in
[`lora/layers.py`](https://github.com/microsoft/LoRA/blob/8f4fdeb0aa05df855a9ffb359a6faa5ee48a4878/lora/layers.py),
class `Linear` unless a bullet names another class or file, and
[`lora/utils.py`](https://github.com/microsoft/LoRA/blob/8f4fdeb0aa05df855a9ffb359a6faa5ee48a4878/lora/utils.py),
both at commit `8f4fdeb0`. Everything below that involves $A$ or $B$ holds only when `r`
is set above its default of 0 (line 98, `r: int = 0`). With `r = 0` the class creates no
`lora_A` or `lora_B`, leaves the weight trainable, and returns only
`F.linear(x, T(self.weight), bias=self.bias)` (lines 111, 151 and 157). Other settings
are at their defaults unless a bullet says otherwise.

- **Rows or columns.** The code works with rows:
  `result = F.linear(x, T(self.weight), bias=self.bias)`, then
  `result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling`
  (lines 152 to 154). `F.linear` computes "$y = xA^T + b$", where the docstring's $A$ is
  the layer's weight, not LoRA's $A$, for an input of shape "$(N, *, in\_features)$" and
  a weight of shape "$(out\_features, in\_features)$"
  ([PyTorch 1.9.0 `F.linear`](https://github.com/pytorch/pytorch/blob/v1.9.0/torch/nn/functional.py#L1831-L1844)):
  the features run along the last axis, so each token is a row.
  `lora_A` is created as `new_zeros((r, in_features))` and `lora_B` as
  `new_zeros((out_features, r))` (lines 112 and 113), and `new_zeros` "Returns a Tensor
  of size `size` filled with `0`"
  ([PyTorch 1.9.0 `Tensor.new_zeros`](https://github.com/pytorch/pytorch/blob/v1.9.0/torch/_tensor_docs.py#L161-L166)).
  These shapes match $A\in\mathbb R^{r\times k}$ and $B\in\mathbb R^{d\times r}$. For
  layers that store their weight as `[in, out]`, the flag `fan_in_fan_out` (line 101:
  "Set this to True if the layer to replace stores weight like (fan_in, fan_out)")
  transposes the stored weight once (lines 118 and 119). A helper `T`,
  `return w.T if self.fan_in_fan_out else w`, transposes it back in `forward` (defined at
  lines 149 and 150, used at lines 152 and 157), and puts $BA$ into the stored layout
  when merging (lines 139, 140 and 145) and unmerging (lines 129, 130 and 135).
- **Multiplication order.** In the unmerged forward pass, Python groups
  `self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T` (line 154) from the left, so the
  product with `self.lora_A.T` is taken first, because "Operators in the same box group
  left to right"
  ([Python 3.6 reference, §6.16](https://docs.python.org/3.6/reference/expressions.html#operator-precedence);
  Sources says why 3.6). That is $B(Ax)$. The code
  forms `self.lora_B @ self.lora_A` only to merge or unmerge (lines 135 and 145). For one $12{,}288\times12{,}288$ matrix with $r=4$, $B(Ax)$ is
  98,304 multiply-adds per token, against 150,994,944 per token for $W_0x$. Forming $BA$
  would cost 603,979,776 multiply-adds once per call, shared by every token in the call,
  and then 150,994,944 per token to multiply by it (from the check).
- **Merging.** With `merge_weights: bool = True` (the default, line 102), the layer's own
  `eval()` adds `T(self.lora_B @ self.lora_A) * self.scaling` into the frozen weight if
  it is not merged yet, and its `train()` subtracts it again if it is merged (lines 128
  to 146). While merged, `forward` returns only `F.linear` with that weight (lines 151
  and 157). This is the paper's $W=W_0+BA$, including the $\alpha/r$ factor that the
  paper's formula leaves out.
- **The scale $\alpha/r$.** `self.scaling = self.lora_alpha / self.r` (line 114), with
  `lora_alpha` defaulting to 1 (line 99). The paper gives $\alpha$ in Tables 9 to 11.
  Table 12 (GPT-3) lists no $\alpha$, so its value for GPT-3 is open.
- **How $A$ starts.** The paper's start is Gaussian (the $A$ row; Figure 1's $\sigma$ is
  under "Not defined in the paper"). The code uses a uniform distribution:
  `nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))`, under the comment "initialize
  A the same way as the default for nn.Linear and B to zero" (lines 124 to 126).
  `kaiming_uniform_` gives "values sampled from $\mathcal U(-\text{bound}, \text{bound})$",
  and `nn.init.zeros_(self.lora_B)` fills $B$ "with the scalar value `0`"
  ([PyTorch 1.9.0 `nn.init`](https://github.com/pytorch/pytorch/blob/v1.9.0/torch/nn/init.py)), so paper
  and code both start $B$ at zero. The `Embedding` class in `lora/layers.py` does the
  reverse: `nn.init.zeros_(self.lora_A)` and `nn.init.normal_(self.lora_B)` (lines 59
  and 60), which draws "from the normal distribution" with mean 0 and standard deviation
  1 by default (same PyTorch file). Some experiments skip this start. Appendix D.1 (RoBERTa) and
  D.2 (DeBERTa): "we initialize the LoRA modules to our best MNLI checkpoint when
  adapting to MRPC, RTE, and STS-B, instead of the usual initialization; the pre-trained
  model stays frozen for all tasks." Only the runs marked † (RoBERTa large, §5.2 and
  D.1) start those three tasks from the pre-trained model. In the other runs on those
  tasks, $A$ and $B$ start at their MNLI-trained values, so $\Delta W\ne0$ at step 0 (my
  inference: a trained $B$ is not zero).
- **Bias.** The code passes the layer's bias to `F.linear` (`bias=self.bias`, line
  152), which adds it as the $b$ of "$y = xA^T + b$"; `nn.Linear` has one by default
  (`bias`: "Default: `True`",
  [PyTorch 1.9.0 `nn.Linear`](https://github.com/pytorch/pytorch/blob/v1.9.0/torch/nn/modules/linear.py#L36-L57)). `Linear` itself freezes only
  `self.weight` (line 116). `mark_only_lora_as_trainable` in `utils.py` (lines 13 to 30),
  with its default `bias: str = 'none'`, sets `requires_grad = False` on every parameter
  whose name lacks `lora_`, biases included (lines 14 to 16). If `bias == 'all'`, it then
  sets `requires_grad` back to `True` on every parameter whose name contains `bias` (lines 19 to 22);
  if `bias == 'lora_only'`, it does so only for the biases of `LoRALayer` modules, that
  is, of the layers replaced by the classes of `lora` (lines 23 to 28).
- **Dropout on the LoRA path.** It is not in (3), and in the code it is off by default.
  The default `lora_dropout: float = 0.` (line 100) makes it the identity: under
  "# Optional dropout", `if lora_dropout > 0.:` builds `nn.Dropout`, `else:
  self.lora_dropout = lambda x: x` (lines 22 to 26, class `LoRALayer`). Only when
  `lora_dropout` is set above 0 does the code apply `nn.Dropout`, which "During
  training, randomly zeroes some of the elements of the input tensor with probability
  `p`"
  ([PyTorch 1.9.0 `nn.Dropout`](https://github.com/pytorch/pytorch/blob/v1.9.0/torch/nn/modules/dropout.py#L24-L26)),
  to $x$ on the LoRA path only (line 154). Whether the "Dropout Prob" row of Table 11
  means this dropout is not stated, so that is open.
- **Freezing $W_0$.** `self.weight.requires_grad = False` (line 116).
- **Which matrices get LoRA.** In most experiments, $W_q$ and $W_v$ (§5.1). In the code
  the user replaces the chosen layers with the classes of `lora`, and `MergedLinear` is
  for "cases where a single `nn.Linear` represents more than one layers, such as in some
  implementations of the attention `qkv` projection" (`README.md` line 68, same commit).
- **Number format.** The paper names a format for one storage figure only: it gives the
  size of the 18M-parameter budget "if stored in FP16" (§7.1; checked under "Parameter
  counts and storage"), where FP16 is a 16-bit float of 2 bytes. Its "350GB model"
  (footnote 4) also works out to 2 bytes per parameter for 175B parameters (my arithmetic,
  check part 8), but the paper names no format for it. It does not state the training
  precision either (I searched the v2 PDF text and the TeX for "FP16", "FP32", "BF16",
  "BFloat", "precision", "half", "mixed", "float" and "16-bit"; only the §7.1 sentence
  matches), so that is open. A later paper, QLoRA (Dettmers et al.,
  arXiv 2305.14314v1), stores the frozen $W_0$ in 4 bits: it "backpropagates gradients
  through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters
  (LoRA)" (abstract), and dequantizes to BFloat16, another 16-bit float format, for the
  forward and backward pass, computing weight gradients
  only for the LoRA parameters (§3).

## What the paper claims and what it shows

- **Weight updates have a low "intrinsic rank".** A hypothesis: "Inspired by this, we
  hypothesize the updates to the weights also have a low “intrinsic rank” during
  adaptation" (§4.1). It is inspired by Aghajanyan et al. (2020), which this reading did
  not open. The paper studies it with experiments in §7, "Understanding the Low-Rank
  Updates", which this reading did not check.
- **$\Delta W$ has rank at most $r$ and, under the usual initialization, is zero at the
  start.** Both follow from the definitions (steps 2 and 7); the check confirms them.
- **No extra inference latency.** A derivation: $W_0$ and $BA$ are both
  $d\times k$, so the merged layer is an ordinary layer, "by construction". The check
  confirms that merged and unmerged outputs match. §4.2 states the limit: "it is not
  straightforward to batch inputs to different tasks with different $A$ and $B$ in a
  single forward pass, if one chooses to absorb $A$ and $B$ into $W$ to eliminate
  additional inference latency". The latency that LoRA avoids is that of adapter layers,
  extra layers that a competing method inserts into the model (§5.1) and that "have to be
  processed sequentially" (§3). Table 1 measures it (GPT-2 medium, "averaged over 100
  trials"); the caption does not say what its $\pm$ values are.
- **Tuning $\alpha$ is like tuning the learning rate.** An argument, with no derivation
  or experiment in §4.1: "When optimizing with Adam, tuning $\alpha$ is roughly the same
  as tuning the learning rate if we scale the initialization appropriately." It is
  hedged ("roughly") and conditional. That the scaling "helps to reduce the need to
  retune hyperparameters when we vary $r$" is attributed to Yang & Hu (2021), not
  opened here.
- **"We simply set $\alpha$ to the first $r$ we try and do not tune it."** A statement
  of practice. The final configurations have $\alpha/r$ of 1, 2, 1 and 8 (RoBERTa base,
  RoBERTa large, DeBERTa XXL, GPT-2; Tables 9 to 11, computed in the check). RoBERTa
  base and DeBERTa XXL have $\alpha=r=8$, which fits the statement if 8 was the first $r$
  tried. RoBERTa large
  ($\alpha=16$, $r=8$) fits it only if the first $r$ tried was 16, and GPT-2 ($\alpha=32$,
  $r=4$) only if it was 32. The paper does not record the first $r$.
- **LoRA roughly recovers full fine-tuning.** An argument, hedged: applying LoRA to all
  weight matrices, training all biases and setting $r$ to the rank of the pre-trained
  matrices, "we roughly recover the expressiveness of full fine-tuning" (§4.1). No
  experiment is attached to it there, and most experiments adapt only $W_q$ and $W_v$
  ("Which matrices get LoRA"), so they do not meet these conditions.
- **Parameter counts and storage.** The count
  "$|\Theta| = 2\times\hat L_{LoRA}\times d_{model}\times r$, where $\hat L_{LoRA}$ is the
  number of weight matrices we apply LoRA to" (§5.1) follows from the shapes in (3) with
  $d=k=d_{model}$. My recomputation for GPT-3 175B (96 layers, §7.1;
  $d_{model}=12{,}288$, §1):
    - The 4.7M budget ($r_q=r_v=1$ or $r_v=2$) and the 37.7M budget ($r_q=r_v=8$) match.
    - $r=4$ on $W_q$ and $W_v$ gives 18,874,368 parameters. The paper prints "18M" in §7.1
      and 18.8M in Table 15.
    - In FP16 that is 37.7 MB (36.0 MiB), against "roughly 35MB" (§7.1).
    - "Reduced by roughly 10,000× (from 350GB to 35MB)" (§4.2): $350/0.035=10{,}000$
      exactly, while counting parameters gives $175\text{B}/18.87\text{M}\approx9{,}272$.
    - Footnote 4's "350GB + 35MB * 100 ≈ 354GB" (my total: 353.5 GB) and
      "100 * 350GB ≈ 35TB" match.
    - One mismatch. Appendix D.4 lists "37.7M ($r_q=r_v=8$ or $r_q=r_k=r_v=r_o=2$)".
      The §5.1 formula gives 18.87M for $r_q=r_k=r_v=r_o=2$, and Table 15 lists that
      configuration at 18.8M and $r_q=r_k=r_v=r_o=4$ at 37.7M. So the "$=2$" in D.4
      disagrees with the paper's own formula and table.
- **Memory and speed on GPT-3 175B.** Figures reported in §4.2 (recomputed in check
  part 11).
    - Memory on the GPU (VRAM): "we reduce the VRAM consumption during training from 1.2TB to 350GB", with
      no table or setup; I found none elsewhere (searched "VRAM", "1.2TB", "350GB"). §4.2
      credits optimizer state, the extra numbers the optimizer stores for each trained
      parameter: "we reduce that VRAM usage by up to 2/3 if r ≪ dmodel as we do not need
      to store the optimizer states for the frozen parameters", so the optimizer keeps
      that state only for $A$ and $B$ (§1: "we do not need to calculate the
      gradients or maintain the optimizer states for most parameters"). The figures give
      a 3.43× ratio, in line with the abstract's "GPU memory requirement by 3 times",
      and a 70.8% cut, which is larger than the "up to 2/3" (66.7%) that the same
      paragraph gives. The 350 GB equals the size of the model's weights alone ("Number
      format"); the paper does not break the figure down.
    - Speed: "a 25% speedup during training on GPT-3 175B compared to full fine-tuning"
      carries footnote 5: "the training throughput for full fine-tuning is 32.5 tokens/s
      per V100 GPU; with the same number of weight shards for model parallelism, the
      throughput is 43.1 tokens/s per V100 GPU for LoRA." That is 32.6% more throughput
      and 24.6% less time per token, so "25%" matches the cut in time per token (my
      reading).

## Sources

- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685v2),
  arXiv 2106.09685v2: the PDF, its `pdftotext` text, and the TeX source
  (`iclr2022_conference.tex`, `expt.tex`) from the v2 e-print.
- microsoft/LoRA at commit `8f4fdeb0aa05df855a9ffb359a6faa5ee48a4878` (16 Sep 2021):
  [`lora/layers.py`](https://github.com/microsoft/LoRA/blob/8f4fdeb0aa05df855a9ffb359a6faa5ee48a4878/lora/layers.py),
  [`lora/utils.py`](https://github.com/microsoft/LoRA/blob/8f4fdeb0aa05df855a9ffb359a6faa5ee48a4878/lora/utils.py),
  [`README.md`](https://github.com/microsoft/LoRA/blob/8f4fdeb0aa05df855a9ffb359a6faa5ee48a4878/README.md),
  [`setup.py`](https://github.com/microsoft/LoRA/blob/8f4fdeb0aa05df855a9ffb359a6faa5ee48a4878/setup.py),
  [`NLG/requirement.txt`](https://github.com/microsoft/LoRA/blob/8f4fdeb0aa05df855a9ffb359a6faa5ee48a4878/NLG/requirement.txt)
  and [`NLU/environment.yml`](https://github.com/microsoft/LoRA/blob/8f4fdeb0aa05df855a9ffb359a6faa5ee48a4878/NLU/environment.yml).
  The repository's only tags, `RoBERTa-base`, `RoBERTa-large`, `GPT-2` and `DeBERTa`,
  all point to this commit, and their releases were published on 16 Sep 2021, before v2
  (16 Oct 2021). Its package is named `lora` (`setup.py` line 7, same commit).
- PyTorch 1.9.0. The `lora` package pins no PyTorch version (the `setuptools.setup` call
  in `setup.py`, lines 6 to 22 at the same commit, lists no dependencies), and the two
  example setups behind the results used here pin different ones: the RoBERTa and
  DeBERTa setup pins 1.9.0 (`NLU/environment.yml` line 51:
  `pytorch=1.9.0=py3.7_cuda11.1_cudnn8.0.5_0`), the GPT-2 setup 1.7.1
  (`NLG/requirement.txt` line 2: `torch==1.7.1+cu101`). I chose 1.9.0 because more of
  the experiments this reading uses are on RoBERTa and DeBERTa (Tables 9 and 10,
  Appendices D.1 and D.2) than on GPT-2 (Tables 1 and 11). The quotes are from the
  docstrings at tag `v1.9.0` of pytorch/pytorch:
  [`torch/nn/modules/linear.py`](https://github.com/pytorch/pytorch/blob/v1.9.0/torch/nn/modules/linear.py#L36-L57),
  [`torch/nn/functional.py`](https://github.com/pytorch/pytorch/blob/v1.9.0/torch/nn/functional.py#L1831-L1844),
  [`torch/_tensor_docs.py`](https://github.com/pytorch/pytorch/blob/v1.9.0/torch/_tensor_docs.py#L161-L166),
  [`torch/_tensor.py`](https://github.com/pytorch/pytorch/blob/v1.9.0/torch/_tensor.py#L326-L329),
  [`torch/nn/init.py`](https://github.com/pytorch/pytorch/blob/v1.9.0/torch/nn/init.py),
  [`torch/nn/modules/dropout.py`](https://github.com/pytorch/pytorch/blob/v1.9.0/torch/nn/modules/dropout.py#L24-L26)
  and [`docs/source/notes/autograd.rst`](https://github.com/pytorch/pytorch/blob/v1.9.0/docs/source/notes/autograd.rst?plain=1#L56-L80).
  At tag `v1.7.1` every quoted passage reads the same, except the autograd notes: there
  the flag "is especially useful when you want to freeze part of your model", and both
  leaf-tensor passages are absent.
- Python 3.6 language reference,
  [§6.16 Operator precedence](https://docs.python.org/3.6/reference/expressions.html#operator-precedence).
  3.6 is the minimum that `setup.py` allows (line 21 at commit `8f4fdeb0`:
  `python_requires='>=3.6'`), so I chose it.
- Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314v1),
  arXiv 2305.14314v1: the PDF and its `pdftotext` text (abstract and §3 only).
- Cited by the paper and not opened: Aghajanyan et al. (2020), Yang & Hu (2021),
  Houlsby et al. (2019).
