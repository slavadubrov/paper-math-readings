# RoFormer: turning queries and keys by their position so attention sees only the distance between tokens

**Source:** Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng
Liu, [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864v5),
v5 (8 Nov 2023), §3.2, equations (12)–(16). Other locations used: §1 (why
position is needed); §2.1,
equations (1)–(2) (what $\boldsymbol x$, $\boldsymbol q$, $\boldsymbol k$, $m$,
$n$ are); §2.2, equation (3) (the method RoPE changes); §3.1, equation (11)
(the requirement); §3.3 (a second formula for $\theta_i$); §3.4.1, equations
(20)–(33) (the 2D derivation); §3.4.2, equation (34) (the element-wise form);
§3.4.3, equations (35)–(37) and Figure 2; §4.1–§4.2, §4.5.5; Figure 1; the README of the linked
repository [ZhuiyiTechnology/roformer](https://github.com/ZhuiyiTechnology/roformer/tree/dfc678ad506fc527ba17ead8db23cbe4d947a9b4)
at commit `dfc678ad50` (21 Mar 2022, the latest commit, so the one closest to
v5).

## What it computes and what it changes

Attention scores a query against a key with a dot product,
$\boldsymbol q_m^\top\boldsymbol k_n$ (equation (2)), where $m$ and $n$ are the
positions of the two tokens. RoPE _rotates_ each projected query and key, that
is, turns it around the origin without changing its length, by an angle
proportional to its own position. §3.1 states the goal as equation (11): the
score should depend on the two positions only through their difference.
Equations (12)–(13) do this for vectors of 2 numbers, (14)–(15) for any even
size $d$ by turning $d/2$ separate pairs of coordinates, and (16) writes out the
resulting score.

Position has to enter somewhere because attention alone ignores order. The
output of (2) for token $m$, $\boldsymbol o_m$, is a weighted sum over all $n$
of the values $\boldsymbol v_n$, with weights from the scores. If $f_q$, $f_k$,
$f_v$, the functions in equation (1) that turn a token's embedding and its
position into its query, key and value, ignore position, reordering the keys
and values together only reorders
the terms of that sum, so every $\boldsymbol o_m$ is unchanged (my derivation).
§1: "the self-attention architecture of the current PLMs has shown to be
position-agnostic Yun et al. [2020]". That is why (1) gives each $f$ a position
argument.

The method it changes is equation (3) in §2.2,
$f_{t}(\boldsymbol x_i,i):=\boldsymbol W_{t}(\boldsymbol x_i+\boldsymbol p_i)$:
a position vector $\boldsymbol p_i$ is _added_ to the token's embedding before
the projection, so the score expands into four terms that mix content and
position (equation (6)). RoPE adds nothing. It multiplies the already projected
query and key by a matrix that depends on the position, and §3 gives the value
vectors no position at all. In the paper's words: "In contrast to the additive
nature of position embedding method adopted in the previous works, i.e.,
Equations (3) to (10), our approach is multiplicative." (§3.2.2).

## The equation

As printed in §3.2.1 (page 4):

$$
\begin{aligned}
f_q(\boldsymbol x_m, m) &= (\boldsymbol W_q\boldsymbol x_m)e^{im\theta}\\
f_k(\boldsymbol x_n, n) &= (\boldsymbol W_k\boldsymbol x_n)e^{in\theta}\\
g(\boldsymbol x_m, \boldsymbol x_n, m - n) &= \operatorname{Re}\big[(\boldsymbol W_q\boldsymbol x_m)(\boldsymbol W_k\boldsymbol x_n)^{*}e^{i(m - n)\theta}\big]
\end{aligned}
\tag{12}
$$

$$
f_{\{q,k\}}(\boldsymbol x_m, m) =
\begin{pmatrix}\cos m\theta & -\sin m\theta\\ \sin m\theta & \cos m\theta\end{pmatrix}
\begin{pmatrix}W^{(11)}_{\{q,k\}} & W^{(12)}_{\{q,k\}}\\ W^{(21)}_{\{q,k\}} & W^{(22)}_{\{q,k\}}\end{pmatrix}
\begin{pmatrix}x^{(1)}_m\\ x^{(2)}_m\end{pmatrix}
\tag{13}
$$

As printed in §3.2.2 (page 5):

$$
f_{\{q,k\}}(\boldsymbol x_m, m) = \boldsymbol R^d_{\Theta,m}\boldsymbol W_{\{q,k\}}\boldsymbol x_m
\tag{14}
$$

$$
\boldsymbol R^d_{\Theta,m} =
\begin{pmatrix}
\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots & 0 & 0\\
\sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots & 0 & 0\\
0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots & 0 & 0\\
0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots & 0 & 0\\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots\\
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2} & -\sin m\theta_{d/2}\\
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2} & \cos m\theta_{d/2}
\end{pmatrix}
\tag{15}
$$

$$
\boldsymbol q_m^\top\boldsymbol k_n
= (\boldsymbol R^d_{\Theta,m}\boldsymbol W_q\boldsymbol x_m)^\top(\boldsymbol R^d_{\Theta,n}\boldsymbol W_k\boldsymbol x_n)
= \boldsymbol x^\top\boldsymbol W_q R^d_{\Theta,n-m}\boldsymbol W_k\boldsymbol x_n
\tag{16}
$$

The right-hand side of (16) is printed with no subscript on $\boldsymbol x$ and
no transpose on $\boldsymbol W_q$; step 11 below gives the corrected form.

## Symbols

| Symbol | Shape or type | Meaning | Role | Defined at |
| ------ | ------------- | ------- | ---- | ---------- |
| $\boldsymbol x_m$, $\boldsymbol x_n$ | column vector of $d$ numbers ($d=2$ in (12)–(13)) | embedding of the token at position $m$ (or $n$), with no position information in it | input data | "$\boldsymbol x_i\in\mathbb R^d$ is the d-dimensional word embedding vector of token $w_i$ without position information" (§2.1) |
| $m$, $n$ | integer | position of the query's token and of the key's token; tokens are numbered from 1 to $N$ | index | "where $\boldsymbol q_m$, $\boldsymbol k_n$ and $\boldsymbol v_n$ incorporate the $m$th and $n$th positions through $f_q$, $f_k$ and $f_v$, respectively" (§2.1, after (1)); "Let $\mathbb S_N=\{w_i\}_{i=1}^N$ be a sequence of $N$ input tokens" (§2.1) |
| $d$ | even integer | number of entries in $\boldsymbol x$; also the superscript on $\boldsymbol R$, where it is a size label, not a power | fixed setting | "In order to generalize our results in 2D to any $\boldsymbol x_i\in\mathbb R^d$ where $d$ is even" (§3.2.2); "We begin with a simple case with a dimension $d = 2$." (§3.2.1) |
| $\boldsymbol W_q$, $\boldsymbol W_k$; $\boldsymbol W_{\{q,k\}}$ | matrix; $d\times d$ by the shape check in step 8 | the matrices that turn an embedding into a query and into a key | not stated in the paper | "simply rotate the affine-transformed word embedding vector by amount of angle multiples of its position index" (§3.2.1, after (13), page 4), which names $\boldsymbol W\boldsymbol x$; shape and training: see the list below |
| $W^{(11)}_{\{q,k\}},\dots,W^{(22)}_{\{q,k\}}$ | numbers | the four entries of the $2\times2$ $\boldsymbol W_q$ or $\boldsymbol W_k$: $(rc)$ is row $r$, column $c$, a label, not a power | as for $\boldsymbol W_q$, $\boldsymbol W_k$ | no defining sentence (see the list below) |
| $x^{(1)}_m$, $x^{(2)}_m$ | numbers | the two coordinates of $\boldsymbol x_m$; the superscripts are labels | input data | "where $(x^{(1)}_m, x^{(2)}_m)$ is $\boldsymbol x_m$ expressed in the 2D coordinates" (§3.2.1, after (13)) |
| $\theta$ (in (12)–(13)) | real number, not 0 | angle, in radians, that the vector turns for each step in position, in the 2D case | fixed setting | "$\theta\in\mathbb R$ is a preset non-zero constant." (§3.2.1) |
| $\Theta$, $\theta_i$ | set of $d/2$ real numbers | one angle per coordinate pair: pair $i$ turns by $\theta_i$ for each step in position | fixed setting (not trained) | "is the rotary matrix with pre-defined parameters $\Theta=\{\theta_i=10000^{-2(i-1)/d}, i\in[1, 2, ..., d/2]\}$" (§3.2.2, after (15)); §3.3 gives another formula, see "What the code must decide" |
| $\boldsymbol R^d_{\Theta,m}$ | $d\times d$ matrix | the rotation for position $m$: $d/2$ blocks of $2\times2$ on the diagonal, zeros elsewhere | computed intermediate, from $m$ and $\Theta$ only; nothing in it is trained | "is the rotary matrix with pre-defined parameters $\Theta$" (§3.2.2); "Note that $\boldsymbol R^d_\Theta$ is an orthogonal matrix" (§3.2.2) |
| $\boldsymbol R^d_{\Theta,n-m}$ (non-bold in (16)) | $d\times d$ matrix | the transposed rotation for $m$ times the rotation for $n$ | computed intermediate | "where $\boldsymbol R^d_{\Theta,n-m}=(\boldsymbol R^d_{\Theta,m})^\intercal\boldsymbol R^d_{\Theta,n}$" (§3.2.2, after (16)) |
| $f_q$, $f_k$; $f_{\{q,k\}}$ | function: (embedding, position) → vector the size of $\boldsymbol W\boldsymbol x$ | the rule that makes the query (or key) of a token from its embedding and its position; (12) and (14) are RoPE's choice of rule | computed intermediate (its value is $\boldsymbol q_m$ or $\boldsymbol k_n$) | "where $\boldsymbol q_m$, $\boldsymbol k_n$ and $\boldsymbol v_n$ incorporate the $m$th and $n$th positions through $f_q$, $f_k$ and $f_v$, respectively" (§2.1); the $\{q,k\}$ subscript: see the list below |
| $\boldsymbol q_m$, $\boldsymbol k_n$ | column vector of $d$ numbers | the query at position $m$ and the key at position $n$, position already applied | computed intermediate | "$\boldsymbol q_m=f_q(\boldsymbol x_m,m)$", "$\boldsymbol k_n=f_k(\boldsymbol x_n,n)$" (equation (1), §2.1); "The query and key values are then used to compute the attention weights" (§2.1) |
| $g$ | function → one real number | the query–key score, written so that it receives the positions only as $m-n$ | computed intermediate | "we require the inner product of query $\boldsymbol q_m$ and key $\boldsymbol k_n$ to be formulated by a function $g$, which takes only the word embeddings $\boldsymbol x_m$, $\boldsymbol x_n$, and their relative position $m-n$ as input variables" (§3.1) |
| $(\cdot)^*$ | complex number → complex number | complex conjugate: flips the sign of the imaginary part. Not the _optimum_ reading of a star | computed intermediate: $(\boldsymbol W_k\boldsymbol x_n)^*$ | "$(\boldsymbol W_k\boldsymbol x_n)^*$ represents the conjugate complex number of $(\boldsymbol W_k\boldsymbol x_n)$" (§3.2.1) |

**Not defined in the paper:**

- **$i$ in $e^{im\theta}$.** Never defined. I looked in §2, §3.2.1, §3.4.1
  (equations (23)–(25)) and §3.4.3. The surrounding text fixes it as the
  imaginary unit, the number with $i^2=-1$: §3.2.1 speaks of the 2D vector's
  "complex form" and defines $\operatorname{Re}[\cdot]$ as "the real part of a
  complex number", and (25) writes $\boldsymbol q=\|\boldsymbol q\|e^{i\theta_q}$
  where "$\|\boldsymbol q\|$, $\|\boldsymbol k\|$ and $\theta_q$, $\theta_k$ are
  the radial and angular part of $\boldsymbol q$ and $\boldsymbol k$ on the 2D
  plane". The same letter is the index in $\Theta$ (§3.2.2), and §3.4.3,
  equation (35), uses both meanings in one exponent, $e^{i(m-n)\theta_i}$.
- **Which coordinate is the real part.** §3.2.1 says only "its complex form".
  The matrix in (13) is multiplication by $e^{im\theta}$ only if coordinate 1 is
  the real part and coordinate 2 the imaginary part (my derivation; Part 1 of
  the check).
- **$\boldsymbol W_q$, $\boldsymbol W_k$.** No sentence gives their shape or
  says that they are trained. §3.2.1 calls $\boldsymbol W\boldsymbol x$ "the
  affine-transformed word embedding vector" and says no more. I looked in §2.1,
  §2.2 (first use, equation (3)), §2.3, §3 and the implementation details of
  §4. Their shape
  follows from (14)–(15) (step 8). §4.2.2 says RoFormer replaces "the sinusoidal
  position encoding in the self-attention block of the baseline model", BERT,
  which suggests they are that model's query and key matrices, but the paper
  does not say so. The README pseudo-code starts after the projection, from
  `qw`, "query hiddens".
- **$W^{(rc)}_{\{q,k\}}$.** No sentence. I read them as matrix entries because
  (13) puts them in the place that $\boldsymbol W_{\{q,k\}}$ holds in (12).
- **The subscript $\{q,k\}$ on $f$ and $\boldsymbol W$.** Not defined. I read it
  as one formula used twice, once for $q$ and once for $k$. (13) and (14) write $\boldsymbol x_m$
  and $m$ for both; for the key, read $\boldsymbol x_n$ and $n$.
- **$\boldsymbol x$ without a subscript in (16)**, the non-bold
  $R^d_{\Theta,n-m}$ there, and $\boldsymbol R^d_\Theta$ with no position in the
  sentence after (16). Not defined. From the middle of (16) the first must be
  $\boldsymbol x_m$ (step 11); I read the other two as the same matrices as
  $\boldsymbol R^d_{\Theta,n-m}$ and $\boldsymbol R^d_{\Theta,m}$ for any $m$.

## Read from the inside out

Equation (12), $d=2$:

1. $\boldsymbol W_q\boldsymbol x_m$: a $2\times2$ matrix times a column of 2
   numbers gives a column of 2 numbers, the query before any position is
   applied. $\boldsymbol W_k\boldsymbol x_n$ gives the key the same way. With
   each token stored as a row instead, the form Part 3 of the check uses, this
   is `x @ Wq.T`.
2. Each column of 2 numbers $(a,b)$ is treated as one complex number $a+ib$. A
   complex number is a point in the plane: real part across, imaginary part up
   (see the list above for which coordinate is which).
3. $e^{im\theta}$: here the superscript is a power, and $m$ times $\theta$ is an
   angle. $e^{i\alpha}=\cos\alpha+i\sin\alpha$ (Euler's formula), a complex
   number of length 1 at angle $\alpha$. Multiplying a point by it turns the
   point by $\alpha$ radians counterclockwise and keeps its distance from the
   origin.
4. $f_q(\boldsymbol x_m,m)=(\boldsymbol W_q\boldsymbol x_m)e^{im\theta}$: the
   query turned by $m\theta$; $f_k$ is the key turned by $n\theta$. Each result
   is 2 numbers. The $=$ defines $f_q$ and $f_k$: this is the solution the paper
   picks.
5. $g$: $(\boldsymbol W_k\boldsymbol x_n)^*$ mirrors the key across the
   horizontal axis. For two points $u=u_1+iu_2$ and $z=z_1+iz_2$ in the plane,
   $\operatorname{Re}[uz^*]=u_1z_1+u_2z_2$, their dot product (my derivation;
   Part 1 of the check). Conjugating the turned key flips its angle:
   $(\boldsymbol W_k\boldsymbol x_ne^{in\theta})^*=(\boldsymbol W_k\boldsymbol x_n)^*e^{-in\theta}$.
   So the turned query times the conjugated turned key is
   $(\boldsymbol W_q\boldsymbol x_m)(\boldsymbol W_k\boldsymbol x_n)^*e^{i(m-n)\theta}$,
   whose real part is $g$ (my derivation). By the rule above, $g$ is the dot
   product of the turned query and the turned key, one real number, and the
   positions enter only as $m-n$.
   $\operatorname{Re}[\cdot]$ is "the real part of a complex number" (§3.2.1).

Equation (13), the same thing in real numbers:

6. Each matrix multiplies from the left: $\boldsymbol W$ ($2\times2$) times
   the column $(x^{(1)}_m, x^{(2)}_m)$ ($2\times1$) gives a $2\times1$ column,
   and the turning matrix ($2\times2$) times that column gives the $2\times1$
   result. Project first, rotate second. The left matrix is step 3
   written for coordinates: it sends $(a,b)$ to
   $(a\cos m\theta-b\sin m\theta,\ a\sin m\theta+b\cos m\theta)$, the real and
   imaginary parts of $(a+ib)(\cos m\theta+i\sin m\theta)$.

Equations (14)–(15), any even $d$:

7. $\Theta$: the angles $\theta_i=10000^{-2(i-1)/d}$ for
   $i=1,\dots,d/2$. Here $[1, 2, ..., d/2]$ is the list of whole numbers 1 to
   $d/2$, not an interval, and $-2(i-1)/d$ is a real power of 10000. Pair 1
   turns 1 radian per position, and each later pair turns more slowly. For
   $d=4$, $\Theta=(1, 0.01)$. For $d=128$, $\theta_1=1$ and
   $\theta_{64}=10000^{-126/128}\approx1.15\times10^{-4}$, so pair 1 makes a
   full turn every $2\pi\approx6.3$ positions and pair 64 needs about 54,400
   (check output). A fast pair returns near its starting angle within a few
   positions, so it tells neighbours apart but repeats; a slow pair barely
   moves between neighbours and does not wrap around for thousands of
   positions, so distant positions stay apart (my reading of these numbers; the
   paper does not discuss it).
8. $\boldsymbol R^d_{\Theta,m}$ ($d\times d$): block $i$, rows and columns
   $2i-1$ and $2i$ counted from 1, is the turning matrix of (13) with angle
   $m\theta_i$. All other entries are 0. Applied to a vector, it turns
   coordinates (1, 2) by $m\theta_1$, coordinates (3, 4) by $m\theta_2$, and so
   on: $d/2$ independent copies of the 2D case. For the product
   $\boldsymbol R^d_{\Theta,m}\boldsymbol W_{\{q,k\}}\boldsymbol x_m$ to exist,
   $\boldsymbol W\boldsymbol x_m$ must have $d$ entries, so $\boldsymbol W$ is
   $d\times d$ when $\boldsymbol x_m$ has $d$ entries (my shape check; the
   paper states no shape).
9. (14) gives $\boldsymbol q_m=\boldsymbol R^d_{\Theta,m}\boldsymbol W_q\boldsymbol x_m$
   and $\boldsymbol k_n=\boldsymbol R^d_{\Theta,n}\boldsymbol W_k\boldsymbol x_n$,
   each $d$ numbers.

Equation (16):

10. $(\cdot)^\top$ is the transpose, which turns a column into a row. A row of
    $d$ times a column of $d$ is one number, the sum over all $d$ coordinates of
    the products: the dot product. Because $\boldsymbol R$ is block diagonal,
    that sum splits into $d/2$ pair sums, each the 2D score $g$ of step 5 with
    its own $\theta_i$; the paper's phrase is "combine them in the merit of the
    linearity of the inner product" (§3.2.2). Part 2 of the check confirms it.
11. Using
    $(\boldsymbol R\boldsymbol W\boldsymbol x)^\top=\boldsymbol x^\top\boldsymbol W^\top\boldsymbol R^\top$
    (the transpose of a product; my step), the score is
    $\boldsymbol x_m^\top\boldsymbol W_q^\top(\boldsymbol R^d_{\Theta,m})^\top\boldsymbol R^d_{\Theta,n}\boldsymbol W_k\boldsymbol x_n$.
    The paper names the middle product $\boldsymbol R^d_{\Theta,n-m}$.
    Transposing a turning matrix turns back by the same angle, and two turns
    add their angles, so the product is the matrix (15) with $n-m$ in place of
    $m$ (my derivation; checked for four $(m,n)$ pairs). The score therefore
    depends on $m$ and $n$ only through $n-m$, which is (11) holding. The
    printed right-hand side, $\boldsymbol x^\top\boldsymbol W_qR^d_{\Theta,n-m}\boldsymbol W_k\boldsymbol x_n$,
    drops the subscript $m$ and the transpose on $\boldsymbol W_q$. With a
    non-symmetric $\boldsymbol W_q$ it gives a different number (check:
    −5.1920 instead of 20.1742). The middle expression is what code computes;
    the corrected right-hand side is a statement about it.
12. (12) writes the angle as $m-n$ and (16) writes $n-m$. Both describe the same
    number: turning the query by $m\theta$ and the key by $n\theta$ changes
    their dot product exactly as turning the key alone by $(n-m)\theta$ does
    (Part 1 checks the $m-n$ form of (12) and Part 2 the $n-m$ form of (16),
    each with both signs). The word orthogonal in "Note that
    $\boldsymbol R^d_\Theta$ is an orthogonal matrix" means
    $\boldsymbol R^\top\boldsymbol R=I$: it keeps lengths and the angles between
    vectors.

**Kind of statement:** (11) is a constraint, a property the score must have.
(12) and (14)–(15) are definitions that produce queries and keys with that
property; substituted back, they say what the code computes,
$\boldsymbol q_m=\boldsymbol R^d_{\Theta,m}\boldsymbol W_q\boldsymbol x_m$.
(13) restates (12) with real matrices. (16) is a computation, an identity, whose
result shows that the constraint holds with
$g=\boldsymbol x_m^\top\boldsymbol W_q^\top\boldsymbol R^d_{\Theta,n-m}\boldsymbol W_k\boldsymbol x_n$.
For each pair $(m,n)$ it returns one real number, the score that equation (2)
then divides by $\sqrt d$ and passes through a softmax over $n$.

## Where the samples come from

(12)–(16) contain no expectation, average or sampling: $m$ and $n$ are the
positions of tokens in the input sequence, $\Theta$ is fixed, and nothing is
detached.

## Check with small numbers

Part 1 uses $d=2$ and $\theta=\pi/2$, a quarter turn per position (the paper
allows any non-zero $\theta$ in 2D). Part 1 is chosen for exact answers, not
for coverage: $\boldsymbol W\boldsymbol x$ has a zero coordinate, and each
angle has a zero sine or cosine. Its position pairs make $m-n$ positive,
negative and zero, and $(7,6)$ repeats the difference of $(1,0)$. Part 1 also
takes $\boldsymbol W_q\boldsymbol x=\boldsymbol W_k\boldsymbol x=(1,0)$ with
$\theta=\pi/4$, where the score is $\cos((m-n)\pi/4)$.
Part 2 uses $d=4$, so the sum over pairs has two terms, with the paper's
$\Theta$, and prints the angles for $d=128$. Part 3 stores tokens as rows, with
the features on the last axis as in the README's `qw.shape`, and rotates them
with my NumPy translation of the README's element-wise code. The README calls
that code "pseudo code for instruction" and does not say which library
`repeat_elements`, `stack`, `reshape` and `shape` come from, so the translation
assumes what they do. For a row $(1,2,3,4)$: `[..., ::2]` gives $(1,3)$ and
`[..., 1::2]` gives $(2,4)$; `stack([-qw[..., 1::2], qw[..., ::2]], 4)`, reshaped
to the shape of `qw`, gives $(-2,1,-4,3)$; `repeat_elements` with
`rep=2, axis=-1` turns $(1,2)$ into $(1,1,2,2)$; `None` inserts an axis of size
1; and `*` multiplies entry by entry, stretching a size-1 axis to match the
other side. Part 3 checks this translation against (15), not the README's own
functions. Part 3's projected rows and all its sines and cosines are non-zero, so every product
in the $2\times2$ blocks of (15) contributes. Part 1 makes all four entries of
$\boldsymbol W$ non-zero. (12)–(16) have no min,
max, clip or indicator, so there are no branches to cover.

```python
import numpy as np

# ---- Part 1: d = 2, equations (12) and (13), with exact answers ----
# In 2D the paper only asks that theta be "a preset non-zero constant";
# pi/2 (a quarter turn per position) makes every answer exact.
theta = np.pi / 2


def rot2(alpha):  # the 2x2 matrix on the left of (13)
    return np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])


def as_complex(v):  # coordinate 1 is the real part, as (13) requires
    return complex(v[0], v[1])


Wq = np.array([[2.0, -1.0], [1.0, -1.0]])  # all four entries W^(11)..W^(22) non-zero
Wk = np.array([[1.0, -1.0], [2.0, -1.0]])
x = np.array([1.0, 1.0])  # the same token at both positions, both coordinates non-zero
q, k = Wq @ x, Wk @ x  # (1, 0) and (0, 1)
assert np.allclose(q, [1, 0]) and np.allclose(k, [0, 1])

print("d=2, theta=pi/2:  m  n  m-n  dot(f_q, f_k)  g from (12)")
for m, n in [(1, 0), (0, 1), (5, 5), (7, 6)]:
    fq = rot2(m * theta) @ Wq @ x  # (13)
    fk = rot2(n * theta) @ Wk @ x
    fq_c = as_complex(Wq @ x) * np.exp(1j * m * theta)  # (12), complex form
    assert np.allclose([fq_c.real, fq_c.imag], fq)  # (12) and (13) agree
    g = (as_complex(q) * np.conj(as_complex(k)) * np.exp(1j * (m - n) * theta)).real
    dot = fq @ fk
    assert np.isclose(dot, g)
    print(f"{'':18}{m:2d} {n:2d} {m - n:4d} {dot:14.6f} {g:12.6f}")

# theta = pi/4 (an eighth of a turn), q = k = (1, 0): the score is cos((m - n) pi/4)
theta8 = np.pi / 4
e1 = np.array([1.0, 0.0])
print("d=2, theta=pi/4, q=k=(1,0):  m  n  score")
for m, n in [(2, 3), (10, 11), (2, 4)]:
    score = (rot2(m * theta8) @ e1) @ (rot2(n * theta8) @ e1)
    assert np.isclose(score, np.cos((m - n) * theta8))
    print(f"{'':28}{m:2d} {n:2d} {score:6.4f}")

# ---- Part 2: general d, equations (14)-(16), with the paper's Theta ----
d = 4
i = np.arange(1, d // 2 + 1)
theta_322 = 10000.0 ** (-2 * (i - 1) / d)  # section 3.2.2: theta_i = 10000^(-2(i-1)/d)
theta_33 = 10000.0 ** (-2 * i / d)  # section 3.3: theta_i = 10000^(-2i/d)
print("Theta, sec 3.2.2:", theta_322, "  sec 3.3:", theta_33)
th = theta_322
th128 = 10000.0 ** (-2 * (np.arange(1, 65) - 1) / 128)  # d = 128, i = 1..64
print("d=128: theta_1 =", th128[0], " theta_64 =", th128[-1])
print("       positions per full turn (2 pi / theta): pair 1", 2 * np.pi / th128[0], " pair 64", 2 * np.pi / th128[-1])


def R(m):  # equation (15): d/2 rotation blocks on the diagonal
    M = np.zeros((d, d))
    for j, th_j in enumerate(th):
        c, s = np.cos(m * th_j), np.sin(m * th_j)
        M[2 * j : 2 * j + 2, 2 * j : 2 * j + 2] = [[c, -s], [s, c]]
    return M


Wq = np.array([[1, 2, 0, -1], [0, 1, 3, 1], [2, 0, 1, 1], [1, -1, 0, 2]], float)
Wk = np.array([[2, 0, 1, 0], [1, 1, 0, -2], [0, 3, 1, 1], [-1, 0, 2, 1]], float)
xm = np.array([1.0, -1.0, 2.0, 0.5])
xn = np.array([0.5, 2.0, -1.0, 1.0])

for m in [0, 1, 7]:
    assert np.allclose(R(m).T @ R(m), np.eye(d))  # R is orthogonal
for m, n in [(3, 1), (1, 3), (4, 4), (10, 2)]:
    assert np.allclose(R(m).T @ R(n), R(n - m))  # the identity under (16)

pairs = [(3, 1), (1, 3), (4, 4), (10, 2)]
print("d=4:  m  n  q_m.k_n      x_m'Wq'R(n-m)Wk x_n  printed x'Wq R(n-m)Wk x_n  sum of 2D g's")
for m, n in pairs:
    lhs = (R(m) @ Wq @ xm) @ (R(n) @ Wk @ xn)  # (14) then the dot product in (16)
    rhs = xm @ Wq.T @ R(n - m) @ Wk @ xn  # (16) with x_m and W_q transposed
    printed = xm @ Wq @ R(n - m) @ Wk @ xn  # (16) as printed (no transpose on W_q)
    qc = (Wq @ xm)[0::2] + 1j * (Wq @ xm)[1::2]  # pair j as one complex number
    kc = (Wk @ xn)[0::2] + 1j * (Wk @ xn)[1::2]
    by_pairs = np.sum(qc * np.conj(kc) * np.exp(1j * (m - n) * th)).real  # (12) per pair
    assert np.isclose(lhs, rhs) and np.isclose(lhs, by_pairs)
    assert not np.isclose(lhs, printed)
    print(f"{'':5}{m:2d} {n:2d} {lhs:9.4f} {rhs:21.4f} {printed:26.4f} {by_pairs:14.4f}")

# Same offset added to both positions: the score does not change.
for off in [1, 100]:
    s0 = (R(3) @ Wq @ xm) @ (R(1) @ Wk @ xn)
    s_off = (R(3 + off) @ Wq @ xm) @ (R(1 + off) @ Wk @ xn)
    assert np.isclose(s0, s_off)

# ---- Part 3: the same thing with tokens stored as rows ----
N = 3
pos = np.arange(1, N + 1)  # the paper counts tokens from 1
X = np.tile(xm, (N, 1))  # the same token at every position
Q, K = X @ Wq.T, X @ Wk.T  # row r of Q is Wq @ X[r], written as a row

# Dense: rotate row m by R(m)
Q_dense = np.stack([R(p) @ Q[r] for r, p in enumerate(pos)])
K_dense = np.stack([R(p) @ K[r] for r, p in enumerate(pos)])

# Element-wise: my NumPy translation of the README pseudo-code (section 3.4.2, eq. (34)):
# sinusoidal_pos[..., 2t] = sin, [..., 2t+1] = cos (the layout of eq. (4)).
ang = pos[:, None] * th[None, :]  # [N, d/2]
sinusoidal_pos = np.stack([np.sin(ang), np.cos(ang)], -1).reshape(N, d)
cos_pos = np.repeat(sinusoidal_pos[..., 1::2], 2, axis=-1)
sin_pos = np.repeat(sinusoidal_pos[..., ::2], 2, axis=-1)


def rope(rows):
    rows2 = np.stack([-rows[..., 1::2], rows[..., ::2]], -1).reshape(rows.shape)
    return rows * cos_pos + rows2 * sin_pos


assert np.allclose(rope(Q), Q_dense) and np.allclose(rope(K), K_dense)

S = rope(Q) @ rope(K).T  # S[m, n] = q_m . k_n, before the 1/sqrt(d) and softmax of (2)
for row in range(N):
    for col in range(N):
        assert np.isclose(S[row, col], X[row] @ Wq.T @ R(pos[col] - pos[row]) @ Wk @ X[col])
# Identical tokens: each diagonal of S is constant, because only n - m matters.
assert np.allclose(np.diag(S, 1), np.diag(S, 1)[0]) and np.allclose(np.diag(S, -1), np.diag(S, -1)[0])
print("scores S[m, n] for the same token at positions 1..3:\n", S.round(4))
print("multiplications per vector: dense R @ v =", d * d, " element-wise (34) =", 2 * d)
print("all checks passed")
```

Run with `uv run --no-project --with numpy python check.py`. Output:

```text
d=2, theta=pi/2:  m  n  m-n  dot(f_q, f_k)  g from (12)
                   1  0    1       1.000000     1.000000
                   0  1   -1      -1.000000    -1.000000
                   5  5    0       0.000000     0.000000
                   7  6    1       1.000000     1.000000
d=2, theta=pi/4, q=k=(1,0):  m  n  score
                             2  3 0.7071
                            10 11 0.7071
                             2  4 0.0000
Theta, sec 3.2.2: [1.   0.01]   sec 3.3: [0.01   0.0001]
d=128: theta_1 = 1.0  theta_64 = 0.00011547819846894582
       positions per full turn (2 pi / theta): pair 1 6.283185307179586  pair 64 54410.14313077675
d=4:  m  n  q_m.k_n      x_m'Wq'R(n-m)Wk x_n  printed x'Wq R(n-m)Wk x_n  sum of 2D g's
      3  1   20.1742               20.1742                    -5.1920        20.1742
      1  3   22.5280               22.5280                   -10.0131        22.5280
      4  4   25.2500               25.2500                    -7.2500        25.2500
     10  2   19.3080               19.3080                    -5.1513        19.3080
scores S[m, n] for the same token at positions 1..3:
 [[ -3.25    19.1138  31.3297]
 [-15.0415  -3.25    19.1138]
 [ -5.2616 -15.0415  -3.25  ]]
multiplications per vector: dense R @ v = 16  element-wise (34) = 8
all checks passed
```

In 2D the projected query $(1,0)$ and key $(0,1)$ score 1 when the query is one quarter
turn ahead ($m-n=1$), −1 when it is one behind, and 0 at equal positions;
$(7,6)$ scores the same as $(1,0)$. With $\theta=\pi/4$, $(2,3)$ and $(10,11)$
both score $\cos(\pi/4)\approx0.7071$ and $(2,4)$ scores 0. With $d=4$ the score equals the corrected
(16) and the sum of the two 2D pair scores, the printed right-hand side does
not, and the score matrix for one repeated token is constant along each
diagonal: it sees only $n-m$.

## What the code must decide

- **Which $\theta_i$.** §3.2.2 gives
  "$\theta_i=10000^{-2(i-1)/d}, i\in[1, 2, ..., d/2]$"; §3.3 says "Following
  Vaswani et al. [2017], we set $\theta_i=10000^{-2i/d}$" with no range for
  $i$. With $i$ from 1 these are different sets: $(1, 0.01)$ against
  $(0.01, 0.0001)$ for $d=4$ (check output). With $i$ counted from 0, as §3.4.3
  does in its sum from $i=0$ to $d/2-1$, the §3.3 formula gives the §3.2.2 set.
  Official code: open. The README pseudo-code takes `sinusoidal_pos` as given
  ("Sinusoidal position embeddings") and points to the source of bert4keras,
  which is not the repository the paper links, so I did not open it.
- **Which coordinates form a pair.** (15) pairs neighbours: (1, 2), (3, 4), …
  README at `dfc678ad50`: `qw[..., ::2]` and `qw[..., 1::2]`, also neighbours
  if its functions do what Part 3 of the check assumes; under those assumptions
  Part 3 matches (15).
- **Dense matrix or element-wise.** README at `dfc678ad50`:
  `qw = qw * cos_pos + qw2 * sin_pos`, read as Part 3 assumes, is the
  element-wise form of §3.4.2,
  equation (34), where the paper's $\otimes$, which it does not define, is an
  element-wise product. My count: $d^2$ multiplications per vector for the
  dense matrix, $2d$ element-wise (16 against 8 for $d=4$).
- **Where in the layer.** RoPE sits in the self-attention layer: "We carry out
  some modifications on self-attention layer of the baseline model" (§4.1.2);
  "we replace the sinusoidal position encoding in the self-attention block of
  the baseline model with our proposed RoPE" (§4.2.2); "You can implement the
  RoPE with a few lines of changes in the self-attention layer." (README at
  `dfc678ad50`). So, in my reading, it rotates the queries and keys that every
  attention layer computes. The additive $\boldsymbol p_i$ of (3) is instead
  added to the word embedding $\boldsymbol x_i$; in the original Transformer
  that happens once, "to the input embeddings at the bottoms of the encoder and
  decoder stacks" (Vaswani et al., 1706.03762v7, §3.5). Within the
  layer, (14) rotates after the projection $\boldsymbol W$. README at
  `dfc678ad50` rotates `qw` and `kw`, "query hiddens" and "key hiddens", then
  computes `tf.einsum('bjhd,bkhd->bhjk', qw, kw)`.
- **Values.** (14) defines only $f_q$ and $f_k$, and §3 gives no $f_v$. README
  at `dfc678ad50` rotates only `qw` and `kw`.
- **Heads.** §3 has one head of size $d$. README at `dfc678ad50`:
  `qw.shape = [batch_size, seq_len, num_heads, hidden_size]` and
  `sinusoidal_pos.shape = [1, seq_len, hidden_size]`, indexed as
  `sinusoidal_pos[..., None, 1::2]`. With `None` and `*` read as Part 3
  assumes, the inserted axis lines up with the heads axis: $d$ is the size of
  one head, and all heads use the same angles.
- **Where positions start.** §2.1 numbers tokens from 1 and Figure 1 labels
  them 1 to 6; README: open. Adding the same offset to both positions leaves the
  score unchanged (Part 2, offsets 1 and 100), but it changes the stored rotated
  vectors themselves.
- **Precision of $m\theta_i$, $\cos$ and $\sin$ for large positions.** Open in
  the paper and the README.

## What the paper claims and what it shows

- "a solution to our formulation Equation (11)" (§3.2.1): **derived**, for
  $d=2$ only, in §3.4.1, equations (20)–(33). The derivation makes choices: it
  takes "a straightforward solution" for the lengths (27), defines
  $f(\boldsymbol x,0)=\boldsymbol W\boldsymbol x$ "To make our results
  comparable to Equation (3)" (32), and sets $\gamma=0$ (33). The paper claims
  a solution, not the only one. §3.4.1 writes the position argument of $g$ as
  $n-m$ (21), where §3.1 writes $m-n$.
- The step from 2D to any even $d$: **argued** in one sentence, "we divide the
  d-dimension space into $d/2$ sub-spaces and combine them in the merit of the
  linearity of the inner product" (§3.2.2), with no separate proof. Part 2 of
  my check agrees for one example.
- (16) and $\boldsymbol R^d_{\Theta,n-m}=(\boldsymbol R^d_{\Theta,m})^\intercal\boldsymbol R^d_{\Theta,n}$:
  **stated** without proof, with the misprint of step 11. My derivation and
  check support the corrected form.
- "$\boldsymbol R^d_\Theta$ is an orthogonal matrix, which ensures stability
  during the process of encoding position information" (§3.2.2): orthogonality
  holds (my check, $d=4$, three positions). The "ensures stability" part is
  **asserted**, with no derivation or experiment.
- "applying matrix multiplication directly as in Equation (16) is not
  computationally efficient" (§3.2.2): **asserted**; the paper gives the
  element-wise form (34) and no timing. My count of multiplications is above.
- Long-term decay (§3.3, §3.4.3, Figure 2) uses the §3.3 formula for
  $\theta_i$ and lies outside (12)–(16). (35) writes the score of (16) as the
  real part of a sum over pairs,
  $\operatorname{Re}\big[\sum_{i=0}^{d/2-1}\boldsymbol q_{[2i:2i+1]}\boldsymbol k^*_{[2i:2i+1]}e^{i(m-n)\theta_i}\big]$.
  Here $\boldsymbol q$ and $\boldsymbol k$ are the projections before rotation,
  not $\boldsymbol q_m$ and $\boldsymbol k_n$: "We can group entries of vectors
  $\boldsymbol q=\boldsymbol W_q\boldsymbol x_m$ and
  $\boldsymbol k=\boldsymbol W_k\boldsymbol x_n$ in pairs", "where
  $\boldsymbol q_{[2i:2i+1]}$ represents the $2i^{th}$ to $(2i+1)^{th}$ entries
  of $\boldsymbol q$" (§3.4.3), each pair read as one complex number as in step
  2, with pairs and entries counted from 0. §3.4.3 **derives** an upper bound,
  (37), on the absolute value of that sum. The real part of a complex number is
  never larger in size than the number itself, so the bound also limits the
  absolute value of the score (my step). The bound's
  distance-dependent part is built from
  $S_j=\sum_{i=0}^{j-1}e^{i(m-n)\theta_i}$, which depends only on $m-n$, and
  Figure 2 **plots** "the value of $\frac{1}{d/2}\sum_{i=1}^{d/2}|S_i|$", with
  the y-axis labelled "relative upper bound", against relative distance. The
  curve falls steeply at first, then rises and falls many times as it trends
  down (figure image). The paper reads this downward trend as weaker dependence
  between distant tokens: "This property coincides with the intuition that a
  pair of tokens with a long relative distance should have less connection"
  (§3.3). Figure 2 shows the bound trending down, not falling at every step;
  it does not plot the score itself.
- Results of the whole model: **measured**, as single numbers with no $\pm$ and
  no count of runs. Table 1: BLEU 27.5 for RoFormer against 27.3 for
  Transformer-base on WMT 2014 English–German, both trained by the authors
  "under the same settings" (§4.1.3). §4.2.3: "Compare to the vanilla BERT,
  RoFormer experiences faster convergence", read from loss curves (Figure 3,
  left). These test RoPE inside a full model, not (12)–(16) on their own. The
  authors list as a limitation (§4.5.5) that "there lacks of thorough
  explanations on why it converges faster than baseline models that
  incorporates other position encoding strategies."

## Sources

- Su, Lu, Pan, Murtadha, Wen, Liu, RoFormer: Enhanced Transformer with Rotary
  Position Embedding, <https://arxiv.org/abs/2104.09864v5>: TeX source, PDF and
  PDF text of v5; version date from the arXiv abstract page.
- ZhuiyiTechnology/roformer at commit
  `dfc678ad506fc527ba17ead8db23cbe4d947a9b4` (21 Mar 2022), the repository the
  paper links in §1: `README.md` (implementation pseudo-code) and `train.py`
  (builds the model with bert4keras, `model='roformer'`).
- Vaswani et al., Attention Is All You Need,
  <https://arxiv.org/abs/1706.03762v7>: §3.5, where the positional encodings
  are added.
- Named, not opened: bert4keras; Yun et al. 2020; the EleutherAI and kexue.fm
  blog posts linked from the README.
