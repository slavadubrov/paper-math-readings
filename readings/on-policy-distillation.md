# MiniLLM and DeepSeek-V4: train a student on text it writes itself, scored by a frozen teacher

**Source:** Yuxian Gu, Li Dong, Furu Wei, Minlie Huang,
[MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543v4),
v4, §2.1, equation (1), and §2.2, equation (7). Other locations used: §2 and
its footnote 3; equations (2)–(6) in §2.2; §2.3 and Algorithm 1; §3.1; §3.4
(Table 4, Figure 8); Appendices A.1, A.2, A.3, B.1 and C.4 (Figure 16); the
official code in
`microsoft/LMOps`, directory `minillm/`, at commit
[`4c59a4e`](https://github.com/microsoft/LMOps/tree/4c59a4e1bcb239d2eeb823d015c7aa00222e0976/minillm)
(committed 2024-04-10, the day v4 was posted).

DeepSeek-AI,
[DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348v1),
v1, §5.1.2, equation (29). Other locations used: §1 (the post-training
paragraph), §2.1, §5.1, §5.2.1–§5.2.3. The paper links model checkpoints and
inference code, but no training code for distillation.

## What it computes and what it changes

A small model (the _student_, $q_\theta$ with weights $\theta$) writes answers
$\boldsymbol y$ to prompts $\boldsymbol x$ drawn from a distribution
$p_{\boldsymbol x}$. A large model that is not being trained (the _teacher_,
$p$) then says, for every token the student wrote,
how likely it would have been to write that token itself. In the paper's
intuitive reading (§2.2), the student is trained to write text the teacher
finds likely and to "stay diverse", not putting all its probability on the
text it writes. Because the text being scored is produced by the model
being trained, this is called _on-policy_ distillation; in reinforcement
learning (RL) a model that picks actions (here, tokens) is called a _policy_.

MiniLLM equation (1) states that goal as a loss, the _reverse_ KL divergence
(defined in step 4 of the first reading below). Equation (7) is the gradient
the method actually follows: the exact gradient of (1), changed in three ways
"To further stabilize and accelerate training" (§1). The method it changes is
standard knowledge distillation (KD), which, as §2 puts it, "approximately"
minimizes a _forward_ KL,
$\mathrm{KL}[p\Vert q_\theta]=\mathbb E_{x\sim p_x,\,y\sim p'}\log\frac{p(y|x)}{q_\theta(y|x)}$,
"where $p'$ can be real data distribution (word-level KD) or teacher
distribution $p$ (sequence-level KD)". The expectation is that KL when
$p'=p$ (my note). Footnote 3 explains the "approximately": "for word-level KD,
y is sampled from the real distribution, not the teacher distribution. For a
strong enough teacher model, we can consider the two distributions
approximately the same." The difference is who writes the text
being scored: there, the data or the teacher ($y\sim p'$); in MiniLLM, the
student ($y\sim q_\theta$), and the two distributions swap places inside the
KL.

DeepSeek-V4 equation (29) applies the same reverse KL to more than ten
teachers "covering various domains", with a weight for each. What it changes is
the way "prior works usually" compute this loss (§5.1.2): they estimate the KL at each
position from the one token the student sampled, the log-ratio of teacher
$\pi_{E_i}$ to student $\pi_\theta$ for that token,
$\mathtt{sg}\big[\log\frac{\pi_{E_i}(y_t|x,y_{<t})}{\pi_\theta(y_t|x,y_{<t})}\big]$,
and feed it to an RL loss as a per-token advantage (a score for that token).
$\mathtt{sg}$ "represents the stop gradient operation": the value is used,
but no gradient flows through it, as with PyTorch's `.detach()`, so the score
is a constant when the loss is differentiated.
DeepSeek-V4 computes the exact KL over the whole vocabulary at each position
instead. The reading of (29) below relates both to MiniLLM's terms.

## The equation

MiniLLM, §2.1:

$$
\theta = \arg\min_\theta \mathcal{L}(\theta) = \arg\min_\theta \operatorname{KL}[q_\theta||p]
= \arg\min_\theta \left[-\mathop{\mathbb{E}}_{\boldsymbol x \sim p_{\boldsymbol x},\boldsymbol y \sim q_\theta} \log \frac{p (\boldsymbol y|\boldsymbol x)}{q_\theta(\boldsymbol y|\boldsymbol x)}\right]. \tag{1}
$$

MiniLLM, §2.2, "In Summary":

$$
\nabla \mathcal{L}(\theta) = -\mathop{\mathbb{E}}_{\substack{\boldsymbol x \sim p_{\boldsymbol x} \\ \boldsymbol y \sim \widetilde{p}(\cdot | \boldsymbol x)}} \Big[ \sum_{t=1}^T w_t\Big[ \underbrace{\nabla \sum_{y' \in V} q_\theta (y'|\boldsymbol y_{<t}, \boldsymbol x) \log \frac{p (y'|\boldsymbol y_{<t}, \boldsymbol x)}{q_\theta (y'|\boldsymbol y_{<t}, \boldsymbol x)}}_{(\nabla \mathcal{L})_{\text{Single}} \text{ part}} + \underbrace{ R_{t+1}^{\text{Norm}} \frac{\nabla q_\theta(y_t | \boldsymbol y_{<t}, \boldsymbol x)}{q_\theta(y_t | \boldsymbol y_{<t}, \boldsymbol x)}}_{ (\nabla \mathcal{L})_{\text{Long}}^{\text{Norm}} \text{ part}}\Big]\Big]. \tag{7}
$$

Equation (7) uses two equations printed just before it in §2.2:

$$
\widetilde p(y_t | \boldsymbol y_{<t}, \boldsymbol x) = \alpha \cdot p(y_t | \boldsymbol y_{<t}, \boldsymbol x) + (1-\alpha) \cdot q_{\theta}(y_t | \boldsymbol y_{<t}, \boldsymbol x), \tag{4}
$$

$$
R^{\text{Norm}}_{t+1} = \frac{1}{T-t-1}\sum_{t'=t+1}^T \log \frac{p(y_{t'} | \boldsymbol y_{<t'}, \boldsymbol x)}{q_{\theta}(y_{t'} | \boldsymbol y_{<t'}, \boldsymbol x)}. \tag{6}
$$

DeepSeek-V4, §5.1.2:

$$
\mathcal{L}_{\text{OPD}}(\theta) = \sum_{i=1}^{N} w_i \cdot \text{D}_{\text{KL}} \left( \pi_{\theta} \parallel \pi_{E_i} \right). \tag{29}
$$

## Symbols

MiniLLM, equations (1) and (7). Shapes are for one prompt and one response; in
code each gains a batch axis.

| Symbol | Shape or type | Meaning | Role | Defined at |
| ------ | ------------- | ------- | ---- | ---------- |
| $\boldsymbol x$ | token ids of the prompt | the prompt | random sample from the prompt set | "conditioning on a prompt $\boldsymbol x$ sampled from the distribution $p_{\boldsymbol x}$" (§2) |
| $p_{\boldsymbol x}$ | a distribution over prompts | where prompts come from; in training, the prompts of dataset $\mathcal D$ | input data | same quote (§2); "Sample a mini-batch of prompts from $\mathcal D$" (Algorithm 1) |
| $\boldsymbol y$, $y_t$, $T$ | ids, shape `[T]`; one id; an integer that differs per response | the response, its token at position $t$, its length | random sample (generated during training) | "the model produces a response $\boldsymbol y=\{y_t\}_{t=1}^T$" (§2); "where $T=\vert\boldsymbol y\vert$" (§2.2, after (2)) |
| $\boldsymbol y_{<t}$ | ids, shape `[t-1]` | the response tokens before position $t$ | computed intermediate (a slice of the sample) | "the state (prefix) $(\boldsymbol y_{<t},\boldsymbol x)$" (Appendix A.1) |
| $t$, $t'$ | integers $1..T$ | positions in the response | index | introduced with $\boldsymbol y=\{y_t\}_{t=1}^T$ (§2) |
| $p$ | for each prompt and prefix, a probability vector of length $\vert V\vert$ | the teacher | frozen model | "a fixed teacher model distribution $p(\boldsymbol y\vert\boldsymbol x)$" (§2) |
| $q_\theta$ | same as $p$ | the student | its weights $\theta$ are trained | "a student model distribution $q_\theta(\boldsymbol y\vert\boldsymbol x)$ parameterized by $\theta$" (§2) |
| $\theta$ | all student weights | under $\arg\min$: the variable searched over; on the left of (1): the weights the search returns | trained weight | same quote (§2) |
| $\mathcal L(\theta)$ | scalar | the training loss | computed intermediate | named by equation (1) itself (§2.1) |
| $\operatorname{KL}[q_\theta\vert\vert p]$ | scalar | reverse KL divergence, student first | computed intermediate | "We consider minimizing the _reverse_ KLD between the student and teacher model distributions as the learning objective" (§2.1); the forward one is written "$\operatorname{KL}[p\vert\vert q_\theta]$" (§2) |
| subscript of $\mathbb E$, with $\sim$ | | which distributions the prompts and responses are drawn from; $\boldsymbol x\sim p_{\boldsymbol x}$ reads "$\boldsymbol x$ is sampled from $p_{\boldsymbol x}$" | random sample (see "Where the samples come from") | "The expectation in Eq. 2 is computed by Monte-Carlo sampling." (§2.2); for $\sim$, "sampled from the distribution $p_{\boldsymbol x}$" (§2, the $\boldsymbol x$ row's quote) |
| $\mid$ | notation, no value | "conditioned on": $q_\theta(y_t\vert\boldsymbol y_{<t},\boldsymbol x)$ is the student's probability of token $y_t$ given the prefix and the prompt; with $\cdot$ left of the bar, the whole next-token distribution | operator the paper names | "conditioning on the state (prefix) $(\boldsymbol y_{<t},\boldsymbol x)$ based on the policy (generation probability) $q_\theta(y_t\vert\boldsymbol y_{<t},\boldsymbol x)$" (Appendix A.1); "conditioning on a prompt $\boldsymbol x$" (§2) |
| $\nabla$ | same shape as $\theta$ | gradient with respect to the student weights | computed intermediate | "We compute the gradient of $\mathcal L(\theta)=\operatorname{KL}[q_\theta\vert\vert p]$ with respect to $\theta$" (Appendix A.2) |
| $r_t$, $R_t$ | scalars per position | $r_t$: the log-ratio teacher over student for the sampled token $y_t$; $R_t$: the sum of $r_{t'}$ from $t$ to $T$ | computed intermediate | "where $T=\vert\boldsymbol y\vert$ and $R_t=\sum_{t'=t}^T\log\frac{p(y_{t'}\vert\boldsymbol y_{<t'},\boldsymbol x)}{q_\theta(y_{t'}\vert\boldsymbol y_{<t'},\boldsymbol x)}$ is the accumulation of $r_{t'}=\log\frac{p(y_{t'}\vert\boldsymbol y_{<t'},\boldsymbol x)}{q_\theta(y_{t'}\vert\boldsymbol y_{<t'},\boldsymbol x)}$ that measures the quality of each step's generation" (§2.2, after (2)) |
| $R^{\text{Norm}}_{t+1}$ | scalar per position | $R_{t+1}$ divided by a length | computed intermediate, outside $\nabla$ | equation (6), introduced by "we add length normalization to $R_{t+1}$ in Eq. 3" (§2.2) |
| $q_\theta(t)$ | probability vector, length $\vert V\vert$ | shorthand for the student's next-token distribution at $t$ | computed intermediate | "where $q_\theta(t)=q_\theta(\cdot\vert\boldsymbol y_{<t},\boldsymbol x)$" (§2.2, after (3)) |
| $\widetilde p$ | probability vector per position | the sampler: teacher and student mixed at each position | computed from a frozen and a trained model | equation (4) |
| $\alpha$ | scalar; 0.2 | how much teacher is mixed in | fixed setting | "where $\alpha$ controls the strength of the teacher mix-in" (§2.2, after (4)); "We set the teacher-mix-in strength $\alpha=0.2$ throughout the experiments in Eq. 4." (§3.1) |
| $w_t$ | scalar per position | importance weight: corrects for sampling from $\widetilde p$ instead of $q_\theta$ | computed intermediate, outside $\nabla$ | "where $w_t=\prod_{t'=1}^{t}\frac{q_\theta(y_{t'}\vert\boldsymbol y_{<t'},\boldsymbol x)}{\widetilde p(y_{t'}\vert\boldsymbol y_{<t'},\boldsymbol x)}$ is the importance weight", then "we approximately set $w_t\approx\frac{q_\theta(y_t\vert\boldsymbol y_{<t},\boldsymbol x)}{\widetilde p(y_t\vert\boldsymbol y_{<t},\boldsymbol x)}$ to reduce the variance of the estimator in Eq. 5" (§2.2, after (5)) |
| $V$, $y'$ | a set of token ids; one id | the vocabulary; the loop variable of the sum over it | index | "where $V$ is the vocabulary size of the language model" (§2.2, after (7)); but $y'\in V$ uses $V$ as a set, as in "the action space (vocabulary) $V$" (Appendix A.1) |
| $(\nabla\mathcal L)_{\text{Single}}$, $(\nabla\mathcal L)^{\text{Norm}}_{\text{Long}}$ | same shape as $\theta$ | names of the two parts of the gradient | computed intermediate | split in equation (3); "$(\nabla\mathcal L)^{\text{Norm}}_{\text{Long}}$ is $(\nabla\mathcal L)_{\text{Long}}$ with $R^{\text{Norm}}_{t+1}$" (§2.2, after (7)) |

DeepSeek-V4, equation (29):

| Symbol | Shape or type | Meaning | Role | Defined at |
| ------ | ------------- | ------- | ---- | ---------- |
| $\pi_\theta$, $\theta$ | a next-token distribution per position, over a vocabulary of $\vert V\vert>100\text{k}$ tokens; $\theta$ the weights | the student, the single "unified" model | trained weight | "Computing the reverse KL loss $\text{D}_{\text{KL}}(\pi_\theta\parallel\pi_{E_i})$ requires sampling training trajectories from the student $\pi_\theta$ to maintain on-policy learning." (§5.1.2); "a vocabulary size $\vert V\vert>100\text{k}$" (§5.2.2) |
| $N$, $i$ | integer, more than 10; index $1..N$ | number of experts; which expert | fixed setting; index | "given a set of $N$ expert models $\{\pi_{E_1},\pi_{E_2},\dots,\pi_{E_N}\}$" and "more than ten teacher models covering various domains are employed to distill a single student model" (§5.1.2) |
| $\pi_{E_i}$ | same as $\pi_\theta$ | expert $i$, used as a teacher: a whole model trained separately for one domain, not one of the "routed experts and shared experts" inside DeepSeek-V4's own feed-forward layers (§2.1), whose weights §5.2.1 calls "MoE expert weights" | frozen model: §5.2.2 describes only a "teacher forward pass" | same quote (§5.1.2); "for each target domain — such as mathematics, coding, agent, and instruction following — a separate expert model is trained independently" (§1); "all teacher weights are offloaded to a centralized distributed storage and are loaded on demand during the teacher forward pass" (§5.2.2) |
| $w_i$ | scalar | weight of expert $i$ | fixed setting | "In this formulation, $w_i$ represents the assigned weight for each expert, typically determined by the relative importance of the expert." (§5.1.2) |
| $\text{D}_{\text{KL}}(\cdot\parallel\cdot)$, $\parallel$ | scalar | the reverse KL: student before $\parallel$, teacher after | computed intermediate | "the reverse KL loss $\text{D}_{\text{KL}}(\pi_\theta\parallel\pi_{E_i})$" (§5.1.2); "the exact KL divergences between teacher and student logits are computed using a specialized TileLang kernel" (§5.2.2) |
| $\mathcal L_{\text{OPD}}(\theta)$ | scalar | the distillation loss; OPD is "On-Policy Distillation" | computed intermediate | "the OPD objective function is defined as:" (§5.1.2) |

**Not defined in the paper:**

- MiniLLM, the base of $\log$: not stated in §2, Algorithm 1 or Appendix A.
  For the official code's choice, see "What the code must decide".
- MiniLLM, $R^{\text{Norm}}_{t+1}$ at the last two positions: equation (6)
  divides by $T-t-1$, which is $0$ at $t=T-1$ and $-1$ at $t=T$. §2.2,
  Algorithm 1 and Appendix A say nothing about these positions.
- MiniLLM, which $w_t$ equation (7) means: §2.2 defines $w_t$ as a product
  and then sets it approximately to one ratio (both quoted in the table), and
  (7) does not say which. Algorithm 1's $\rho_t$, "where
  $\rho_t(\theta)=\frac{q_\theta(y_t|\boldsymbol y_{<t},\boldsymbol x)}{\widetilde p(y_t|\boldsymbol y_{<t},\boldsymbol x)}$"
  (Algorithm 1), and the code ("$w_t$" under "What the code must decide") use
  the single ratio.
- DeepSeek-V4, what $\text{D}_{\text{KL}}(\pi_\theta\parallel\pi_{E_i})$ is
  taken over: equation (29) prints no prompt, response, position or average.
  §5.1.2 and §5.2.2 say it is computed on student-sampled trajectories
  (responses the student wrote), over
  the full vocabulary, from logits, but not how the per-position values are
  combined (summed or averaged, per sequence or per token; this matters with
  very long responses), or which prompts are used.
- DeepSeek-V4, the values of $w_i$, whether they sum to 1, and how they
  combine with the statement that "the unified policy $\pi_\theta$ selectively
  learns from the specialized expert relevant to the current task context"
  (§5.1.2). As printed, (29) puts every expert into the loss for every sample.
  §5.2.2 says "we order training samples by teacher index during data
  dispatching". My reading is that each sample is paired with one teacher, but
  the paper does not say so or say how $w_i$ enters then.
- DeepSeek-V4, $x$, $y_t$, $y_{<t}$ and the bar $\mid$ in the prior-works expression
  $\mathtt{sg}[\cdot]$ (§5.1.2): not defined in §5. They do not appear in
  (29) itself.

Where I looked in DeepSeek-V4: §1, §5.1, §5.1.1, §5.1.2, §5.2 and §5.2.1–§5.2.4,
and every occurrence of $\pi$, "KL", "OPD" and "trajectories" in the TeX
source, and of "condition" in the PDF text.

## Read from the inside out

### MiniLLM equation (1)

1. **Probability of one response.** For a prompt and a response, the student
   gives a next-token distribution at every position: row $t$ of the softmax
   of its logits over the vocabulary axis, shape `[T, |V|]`. The logits are
   the raw scores the model outputs, one per vocabulary entry; softmax
   exponentiates them and divides by their sum. The probability of the whole
   response is the product of the chosen entries,
   $q_\theta(\boldsymbol y|\boldsymbol x)=\prod_{t=1}^T q_\theta(y_t|\boldsymbol y_{<t},\boldsymbol x)$;
   Appendix A.2 uses this factorization when it turns $\nabla\log q_\theta(\boldsymbol y|\boldsymbol x)$
   into a sum over $t$. Its log is a scalar, the sum of the gathered
   log-probabilities. The teacher's $p(\boldsymbol y|\boldsymbol x)$ is built
   the same way from the teacher's logits on the same tokens.
2. **Log-ratio.** $\log\frac{p(\boldsymbol y|\boldsymbol x)}{q_\theta(\boldsymbol y|\boldsymbol x)}=\log p-\log q_\theta$:
   one scalar per (prompt, response). It is positive when the teacher finds
   the response more likely than the student does.
3. **Average.** $\mathbb E_{\boldsymbol x\sim p_{\boldsymbol x},\boldsymbol y\sim q_\theta}$
   averages over prompts, and over responses that the student itself
   generates for each prompt ($\boldsymbol y\sim q_\theta$ is short for
   $\boldsymbol y\sim q_\theta(\cdot|\boldsymbol x)$, as (2) writes it). Result:
   a scalar.
4. **Minus sign, and the name.** $-\mathbb E\log\frac{p}{q_\theta}=\mathbb E_{\boldsymbol y\sim q_\theta}\log\frac{q_\theta}{p}$.
   An average, under one distribution $a$, of $\log\frac{a}{b}$ is the KL
   divergence $\operatorname{KL}[a||b]$. It is never negative and is 0 only
   when the two distributions agree. The order matters: the average runs under
   the first argument. With the student first, a response the student writes
   but the teacher almost never would costs a lot ($\log\frac{q}{p}$ is large);
   a response the teacher would write but the student never does costs
   nothing, because the student never samples it. The paper calls what
   minimizing this KL does "mode-seeking": "$q_\theta$ assigns high probabilities
   to $p$'s large modes and ignore the small ones" (§2.1), a _mode_ being a
   peak of a distribution. The forward KL, which standard KD
   approximately minimizes (see "What it computes and what it changes"),
   averages under the teacher and penalizes the opposite case, so a student
   too small to copy the teacher tends to spread its probability over all of the teacher's
   options and the gaps between them: "Minimizing forward KLD causes $q_\theta$
   to place large probability masses on the zero-probability regions of $p$,
   corresponding to the generation of low-quality text in practice" (§2.1).
   That capacity limit is the paper's reason for the reverse KL: "the
   low-capacity student models cannot perfectly imitate the complex text
   generation distribution of the teacher models or humans" (§2), so MiniLLM
   "encourages the student to generate samples preferred by the teacher within
   its own capacities" (§2.1).
5. **$\arg\min_\theta$.** It returns the weights that make $\mathcal L$
   smallest, not the smallest value, and says nothing about how to find them;
   §2.2 supplies the gradient. The first $=$ assigns that result to $\theta$;
   the second and third $=$ restate $\mathcal L$ by definition.

Read token by token (my derivation, confirmed in the check). By step 1's
factorization,
$\log\frac{q_\theta(\boldsymbol y|\boldsymbol x)}{p(\boldsymbol y|\boldsymbol x)}=\sum_t\log\frac{q_\theta(y_t|\boldsymbol y_{<t},\boldsymbol x)}{p(y_t|\boldsymbol y_{<t},\boldsymbol x)}$.
Averaging term $t$ over $y_t\sim q_\theta(\cdot|\boldsymbol y_{<t},\boldsymbol x)$
with the prefix held fixed gives the per-token KL (later tokens do not enter
term $t$), so
$\operatorname{KL}[q_\theta||p]=\mathbb E_{\boldsymbol y\sim q_\theta}\sum_{t=1}^T\operatorname{KL}\big[q_\theta(\cdot|\boldsymbol y_{<t},\boldsymbol x)\,||\,p(\cdot|\boldsymbol y_{<t},\boldsymbol x)\big]$.
The sequence-level loss is the expected sum, over positions of student-written
text, of per-token KLs, each a sum over the vocabulary. DeepSeek-V4's
"full-vocabulary" KL is this per-token quantity.

**Kind of statement:** an objective, minimized. It averages over prompts from
$p_{\boldsymbol x}$ and over the student's own samples. Because the sampling
distribution depends on $\theta$, the gradient is not the gradient of a batch
mean on fixed data. If you sample responses, take the batch mean of
$\log q_\theta(\boldsymbol y|\boldsymbol x)-\log p(\boldsymbol y|\boldsymbol x)$
and differentiate it with the samples held fixed, you get only the "$-1$" term of
equation (2) below, whose average is zero (my derivation; checked under "What the numbers show").

### MiniLLM equation (7), built up from (2)–(6)

The paper reaches (7) in four steps (§2.2). Each changes one thing.

- **(2), the exact gradient**, derived with the policy gradient theorem
  (Appendix A.2):
  $\nabla\mathcal L(\theta)=-\mathbb E_{\boldsymbol x\sim p_{\boldsymbol x},\boldsymbol y\sim q_\theta(\cdot|\boldsymbol x)}\sum_{t=1}^T(R_t-1)\nabla\log q_\theta(y_t|\boldsymbol y_{<t},\boldsymbol x)$.
  The idea, in plain words: sampling a token cannot be differentiated, but the
  gradient of an average over samples equals the average of "score of the
  sample" times "gradient of the log-probability of having sampled it". Here
  the score of token $t$ is $R_t$, the teacher-over-student log-ratio summed
  from $t$ to the end, so a token gets credit for the tokens that follow it.
  As a loss to differentiate, this is
  $-\sum_tR_t\log q_\theta(y_t|\boldsymbol y_{<t},\boldsymbol x)$ with $R_t$
  held constant (a _REINFORCE_ loss), up to the $-1$, which averages to zero.
- **(3), single-step decomposition.** $R_t=r_t+R_{t+1}$. The $r_t$ piece
  is replaced by its exact average over the vocabulary,
  $\mathbb E_{y_t\sim q_\theta(t)}[r_t]=\sum_{y'}q_\theta(y')\log\frac{p(y')}{q_\theta(y')}$,
  differentiated directly (the "Single" part); $R_{t+1}$ keeps the sampled
  form (the "Long" part). The $-1$ disappears in Appendix A.3's product-rule
  step, because
  $\mathbb E_{y_t\sim q_\theta(t)}[\nabla\log q_\theta(y_t|\cdot)]=\sum_{y'}\nabla q_\theta(y'|\cdot)=\nabla1=0$;
  so $\nabla\mathbb E_{y_t\sim q_\theta(t)}[r_t]=\mathbb E_{y_t\sim q_\theta(t)}[(r_t-1)\nabla\log q_\theta(y_t|\cdot)]$
  (my one-line version). Value: equal to (2).
- **(4)–(5), teacher-mixed sampling.** The paper observes that, trained with
  (2), $q_\theta$ "sometimes produces degenerated sentences $\boldsymbol y$
  that receive high scores from the teacher (e.g., repeated phrases) during
  sampling, especially for small student models" (§2.2). So responses are
  sampled from $\widetilde p$, not from the student, and each term is
  multiplied by $w_t$.
  An importance weight is the ratio "probability under the distribution you
  want to average over" divided by "probability under the one you sampled
  from"; multiplying by it turns an average over $\widetilde p$ into one over
  $q_\theta$. It is a product over positions because a response's probability
  is a product over positions.
- **(6), length normalization.** $R_{t+1}$ is divided by $T-t-1$. The sum in
  (6) has $T-t$ terms, so this is not their mean (for $T=4$, $t=1$ it divides
  three terms by 2).

Now (7) from the inside out, for one response:

1. $q_\theta(y'|\boldsymbol y_{<t},\boldsymbol x)$ and $p(y'|\boldsymbol y_{<t},\boldsymbol x)$
   for all $y'\in V$: two probability vectors of shape `[|V|]` at position
   $t$ (rows of student and teacher softmax; `[B, T, |V|]` for a batch).
   $y'$ runs over the vocabulary; it is not the sampled token $y_t$.
2. $\sum_{y'\in V}q_\theta\log\frac{p}{q_\theta}$: sum over the vocabulary
   axis, one scalar per position. It equals
   $-\operatorname{KL}[q_\theta(t)||p(t)]$, the per-token reverse KL with a
   minus sign (my rewriting).
3. $\nabla$ of that scalar: shape of $\theta$. The gradient flows through
   $q_\theta$ in both places; $p$ is fixed.
4. $\frac{\nabla q_\theta(y_t|\cdot)}{q_\theta(y_t|\cdot)}=\nabla\log q_\theta(y_t|\cdot)$:
   shape of $\theta$, the gradient of the log-probability of the token that
   was actually sampled.
5. $R^{\text{Norm}}_{t+1}$: a scalar per position, from the sampled tokens
   after $t$ (equation (6)). It stands outside $\nabla$, so it is a constant
   for differentiation.
6. Add the two parts; multiply by $w_t$, a scalar per position, also outside
   $\nabla$.
7. $\sum_{t=1}^T$: over the positions of this response. $T$ differs between
   responses, so a long response contributes more terms.
8. $\mathbb E_{\boldsymbol x\sim p_{\boldsymbol x},\boldsymbol y\sim\widetilde p(\cdot|\boldsymbol x)}$:
   the average over prompts and over responses sampled from the mix.
   Algorithm 1 estimates it with a mini-batch of $M$ responses and a factor
   $\frac1M$.
9. The leading minus: the update in Algorithm 1 is
   $\theta\leftarrow\theta-\eta[\ldots]$, with "Learning rate $\eta$" among
   Algorithm 1's inputs, so it lowers each per-token KL and raises the log-probability of tokens followed by a high $R^{\text{Norm}}$.

Superscripts and subscripts: "Norm", "Single" and "Long" are labels;
$\nabla$ without a subscript is with respect to $\theta$. The $=$ in (7)
names the gradient the method uses. After the per-token $w_t$ and the length
normalization it is not the exact gradient of (1); the paper says "we
approximately set", and the check below measures how far off it is in a toy.

**Kind of statement:** an estimator, of a gradient. It is the direction used in
the update rule of Algorithm 1. What is counted: the per-position terms of $M$
sampled responses, each with an exact sum over the vocabulary in the Single
part and one sampled token in the Long part.

### DeepSeek-V4 equation (29)

1. $\text{D}_{\text{KL}}(\pi_\theta\parallel\pi_{E_i})$: a scalar. The student
   comes first, so this is the reverse KL of MiniLLM (1) with $q_\theta=\pi_\theta$
   and $p=\pi_{E_i}$; the paper calls it "the reverse KL loss". Equation (29)
   prints no prompts, positions or vocabulary. From §5.1.2 ("full-vocabulary
   logit distillation") and §5.2.2 ("the exact KL divergences between teacher
   and student logits"), the value is computed at each position of a
   student-sampled trajectory over the whole vocabulary, which is the
   per-token KL of step 2 above (my identification). The prior works'
   advantage $\mathtt{sg}[\cdot]$ is instead the sampled $r_t$ of MiniLLM (2), with
   $p=\pi_{E_i}$ and $q_\theta=\pi_\theta$ (my observation, not a statement of
   either paper). How positions are combined: see "Not defined in the paper".
2. $w_i\cdot$: scales expert $i$'s KL by a fixed number.
3. $\sum_{i=1}^N$: over experts. Result: one scalar,
   $\mathcal L_{\text{OPD}}(\theta)$. Whether each sample really uses every
   expert: see "Not defined in the paper".
4. The $=$ is a definition ("the OPD objective function is defined as").

The paper prints no gradient. If an implementation differentiates only the
per-position KLs, it gets MiniLLM's Single part and drops the Long part, the
effect of a token on which later positions are visited. How far the Single
part alone is from the exact gradient of the sequence-level reverse KL in a
toy: see "What the numbers show" (MiniLLM). The paper does not say whether a
Long-type term is added.

**Kind of statement:** an objective, minimized: a weighted sum of $N$ reverse
KLs. The average over student-sampled trajectories is stated in the text, not
printed.

## Where the samples come from

- **MiniLLM (1).** $\boldsymbol x\sim p_{\boldsymbol x}$: in training, prompts
  from the fixed dataset $\mathcal D$ (Algorithm 1), built from
  databricks-dolly-15K with "about 12.5K examples for training" (§3.1).
  $\boldsymbol y\sim q_\theta$: generated during training by the student
  whose weights $\theta$ are being optimized.
- **MiniLLM (7).** $\boldsymbol x$ as above. $\boldsymbol y\sim\widetilde p$:
  generated during training, each token drawn from
  $0.2\,p+0.8\,q_\theta$. The student in the mix is the copy that existed when
  the batch was collected: "we collect 256 sentences at once and adopt 4 inner
  epochs when doing the policy optimization" (Appendix B.1); an inner epoch
  is one pass of updates over those 256 sentences. How soon the
  weights move on depends on the size of each update. Read with B.1's "a
  mini-batch size 64 in all cases", the 256 sentences give four updates per
  inner epoch, so the weights already move within the first one (my reading).
  The official script takes one update per inner epoch: 256 sampled
  responses (_rollouts_), 16 on each of 16 GPUs, a batch size of 16,
  gradient accumulation 1 (each update uses the gradients of one batch, not
  several) and 4 inner epochs (`train_base_xl.sh` lines 7, 27, 28, 70 and 71;
  `arguments.py` line 288; `trainer.py` lines 246–247 and 390–391). Each
  batch ends with one call to `step` on the DeepSpeed engine (`trainer.py`
  lines 138 and 295). In DeepSpeed 0.10.0, the version `install.sh`
  installs, that call takes an optimizer step only at a "boundary of
  gradient accumulation", which "will trigger gradient reductions and an
  optimizer step" (`deepspeed/runtime/engine.py` lines 1935–1949, 1985–1994
  and 2070–2087). With the setting 1, which `train_minillm.py` line 71
  writes into the DeepSpeed config and DeepSpeed keeps because that config
  gives no total batch size (`deepspeed/runtime/config.py` lines 773–776 and
  920–923; `deepspeed/runtime/constants.py` line 18), every call is a
  boundary (`engine.py` lines 234 and 1945–1947). The engine counts
  a step as skipped when the optimizer reports an overflow, which the code's
  comment ties to the "DS fp16 optimizer" (`engine.py` lines 2025–2032); the
  DeepSpeed config the script passes enables fp16, 16-bit floating point
  (`train_base_xl.sh` line 87; `configs/deepspeed/ds_config.json` lines 8–9). In the script, then, the one update of
  inner epoch 1 uses the collecting copy, and inner epochs 2–4 use weights
  that have moved on. Algorithm 1 puts the current student in the numerator
  of $\rho_t$ and clips $\rho_t$ (what that clip bounds: "Clipping" under
  "What the code must decide"). In the code, $w$ is computed with the
  collecting copy, written $q_{\text{old}}$, under `torch.no_grad()`
  (`sampler.py` lines 108–115), and `_pg_loss` multiplies it by
  $\exp(\log q_\theta-\log q_{\text{old}})$, which gives $\rho_t$
  (`losses.py` lines 72–74). The inner $\mathbb E_{y_t\sim q_\theta(t)}$ is
  not sampled: it is the sum over the vocabulary. Held fixed: $w_t$ and
  $R^{\text{Norm}}_{t+1}$, which sit outside $\nabla$ (code: computed under
  `torch.no_grad()`, the Long-part multiplier returned with `.detach()`,
  `losses.py` line 56), and the teacher (logits under `torch.no_grad()`,
  `losses.py` lines 99–100, `reward.py` line 41). PyTorch 2.0 documents that
  under `torch.no_grad()` "the result of every computation will have
  `requires_grad=False`", and that the result of `Tensor.detach` never
  requires gradient. The language-modelling term of
  Algorithm 1 uses $\boldsymbol d\sim\mathcal D_{\text{PT}}$, a fixed corpus.
- **DeepSeek-V4 (29).** No expectation is printed. The trajectories are
  generated during training by the student (§5.1.2, quoted under $\pi_\theta$
  in the table), produced by a separate
  "generation service for RL/OPD rollout" (§5.2.3). The paper does not say
  whether that service uses the current weights or a lagging copy, nor which
  prompts are used. Teacher outputs come from a forward pass whose last-layer
  hidden states (one vector per token) are cached and passed through the
  teacher's prediction head, the module that turns them into logits, "to
  reconstruct the full logits on the fly" (§5.2.2).

## Check with small numbers

For MiniLLM: two prompts, a two-token vocabulary and 3-token responses with
no end token (my simplification), with random logit tables for student and
teacher, so every expectation is an exact sum over all 16 (prompt, response)
pairs and the true gradient comes from finite differences (moving each logit
up and down by $10^{-6}$ and measuring the change in the loss); this exercises two
items in every average and sum, all positions, $\alpha=0.2$ and $\alpha=0$, the
three readings of $w_t$, and (6) at every position including the division by
zero. For the clipping of Algorithm 1 as the code applies it (described under
"What the code must decide"): one token whose starting ratio falls below,
one inside and one above $[0.8,1.2]$, each with a multiplier $A=+1$ and
$A=-1$, so each argument of the $\max$ wins at least once. For DeepSeek-V4: one position, three tokens, two experts with unequal
weights and KLs of exactly $\frac14\log2$ and $\log1.25$, both KL orders, a
zero KL and an infinite one.

```python
"""Small-number checks for MiniLLM (2306.08543v4) eqs (1)-(7) and DeepSeek-V4 (2606.19348v1) eq (29).

Toy world: 2 prompts, vocabulary {0, 1}, every response exactly T = 3 tokens.
Every model is a table of logits, one row per (prompt, prefix), so all
expectations can be computed exactly by listing all 2 * 8 responses.
"""

import itertools

import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)

V, T, X = 2, 3, 2
p_x = np.array([0.5, 0.5])  # prompt distribution
prefixes = [()] + [(a,) for a in range(V)] + [(a, b) for a in range(V) for b in range(V)]
S = {pre: i for i, pre in enumerate(prefixes)}  # 7 prefix states per prompt
seqs = list(itertools.product(range(V), repeat=T))  # 8 responses per prompt

teacher = rng.normal(0, 1.5, (X, len(S), V))  # fixed logits of p
theta = rng.normal(0, 1.0, (X, len(S), V))  # student logits = the parameters theta


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def dist(logits, x, y, t):  # next-token distribution at 0-based position t, given y[:t]
    return softmax(logits[x, S[tuple(y[:t])]])


def mix(th, x, y, t, alpha):  # eq (4): p~ = alpha * p + (1 - alpha) * q
    return alpha * dist(teacher, x, y, t) + (1 - alpha) * dist(th, x, y, t)


def seq_p(probs_fn, x, y):
    return np.prod([probs_fn(x, y, t)[y[t]] for t in range(T)])


def q_of(th):
    return lambda x, y, t: dist(th, x, y, t)


p_fn = lambda x, y, t: dist(teacher, x, y, t)


def L(th):  # eq (1): E_{x ~ p_x, y ~ q} log q(y|x) / p(y|x) = KL[q || p]
    return sum(
        p_x[x] * seq_p(q_of(th), x, y) * np.log(seq_p(q_of(th), x, y) / seq_p(p_fn, x, y))
        for x in range(X)
        for y in seqs
    )


def grad_fd(f, th, h=1e-6):
    g = np.zeros_like(th)
    for i in np.ndindex(th.shape):
        a, b = th.copy(), th.copy()
        a[i] += h
        b[i] -= h
        g[i] = (f(a) - f(b)) / (2 * h)
    return g


def dlogq(th, x, y, t):  # gradient of log q(y_t | y_<t, x) w.r.t. the logit table
    g = np.zeros_like(th)
    q = dist(th, x, y, t)
    g[x, S[tuple(y[:t])]] = np.eye(V)[y[t]] - q
    return g


def r(th, x, y, t):  # r_t = log p(y_t|.) / q(y_t|.)
    return np.log(dist(teacher, x, y, t)[y[t]] / dist(th, x, y, t)[y[t]])


def R(th, x, y, t):  # R_t = sum_{t' >= t} r_t' ; R_{T+1} = 0
    return sum(r(th, x, y, s) for s in range(t, T))


def kl_step(th, x, y, t):  # KL[q(.|y_<t,x) || p(.|y_<t,x)], summed over the vocabulary
    q, p = dist(th, x, y, t), dist(teacher, x, y, t)
    return float(np.sum(q * np.log(q / p)))


def dEr(th, x, y, t):  # gradient of E_{y_t ~ q(t)}[r_t] = -KL_step, w.r.t. the logit table
    q, p = dist(th, x, y, t), dist(teacher, x, y, t)
    g = np.zeros_like(th)
    g[x, S[tuple(y[:t])]] = -(q * (np.log(q / p) - np.sum(q * np.log(q / p))))
    return g


def expect(probs_fn, term):  # sum_x p_x sum_y P(y|x) term(x, y)
    return sum(p_x[x] * seq_p(probs_fn, x, y) * term(x, y) for x in range(X) for y in seqs)


cos = lambda a, b: float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
q = q_of(theta)

# ---- eq (1): the loss, exactly and by Monte Carlo ----
L0 = L(theta)
chain = expect(q, lambda x, y: sum(kl_step(theta, x, y, t) for t in range(T)))
assert np.isclose(L0, chain)  # sequence KL = expected sum of per-token KLs (chain rule)
N = 20000
xs = rng.choice(X, size=N, p=p_x)
samples = []
for x in xs:
    y = []
    for t in range(T):
        y.append(rng.choice(V, p=dist(theta, x, y, t)))
    samples.append(np.log(seq_p(q, x, y) / seq_p(p_fn, x, y)))
mc, se = np.mean(samples), np.std(samples) / np.sqrt(N)
assert abs(mc - L0) < 4 * se
print(f"eq (1) exact L = {L0:.4f}; chain-rule sum of per-token KLs = {chain:.4f}")
print(f"eq (1) Monte Carlo, {N} student samples: {mc:.4f} +- {se:.4f} (standard error)")

# ---- eq (2) and eq (3): exact gradient identities ----
G = grad_fd(L, theta)
G2 = -expect(q, lambda x, y: sum((R(theta, x, y, t) - 1) * dlogq(theta, x, y, t) for t in range(T)))
minus_one = expect(q, lambda x, y: sum(dlogq(theta, x, y, t) for t in range(T)))
single = -expect(q, lambda x, y: sum(dEr(theta, x, y, t) for t in range(T)))
long_ = -expect(q, lambda x, y: sum(R(theta, x, y, t + 1) * dlogq(theta, x, y, t) for t in range(T)))
assert np.allclose(G2, G, atol=1e-6)
assert np.allclose(minus_one, 0, atol=1e-12)  # the "-1" in (R_t - 1) averages to zero
assert np.allclose(single + long_, G, atol=1e-6)
print(f"|grad L| (finite differences) = {np.linalg.norm(G):.4f}")
print(f"eq (2) matches: {np.allclose(G2, G, atol=1e-6)}; size of the '-1' term: {np.linalg.norm(minus_one):.1e}")
print(f"eq (3) Single + Long matches: {np.allclose(single + long_, G, atol=1e-6)}")
print(f"Single part alone: cos with grad L = {cos(single, G):.4f}, |Single| = {np.linalg.norm(single):.4f}")

# ---- eq (4)-(5): teacher-mixed sampling with importance weights ----
alpha = 0.2
pt = lambda x, y, t: mix(theta, x, y, t, alpha)
ratio = lambda x, y, t: dist(theta, x, y, t)[y[t]] / pt(x, y, t)[y[t]]
w_upto_t = lambda x, y, t: np.prod([ratio(x, y, s) for s in range(t + 1)])  # as printed in eq (5)
w_whole = lambda x, y, t: np.prod([ratio(x, y, s) for s in range(T)])  # whole response
w_token = lambda x, y, t: ratio(x, y, t)  # the approximation w_t ~ q / p~


def eq5(w):
    s = -expect(pt, lambda x, y: sum(w(x, y, t) * dEr(theta, x, y, t) for t in range(T)))
    l = -expect(pt, lambda x, y: sum(w(x, y, t) * R(theta, x, y, t + 1) * dlogq(theta, x, y, t) for t in range(T)))
    return s, l


for name, w in [("product up to t", w_upto_t), ("product over whole response", w_whole), ("per-token q/p~", w_token)]:
    s, l = eq5(w)
    print(
        f"eq (5), alpha={alpha}, w_t = {name:28s}: Single exact {np.allclose(s, single, atol=1e-9)!s:5s} "
        f"Long exact {np.allclose(l, long_, atol=1e-9)!s:5s} cos(sum, grad L) = {cos(s + l, G):.4f}"
    )
s_t, l_t = eq5(w_upto_t)
s_w, l_w = eq5(w_whole)
assert np.allclose(s_t, single, atol=1e-9) and not np.allclose(l_t, long_, atol=1e-6)
assert np.allclose(s_w + l_w, G, atol=1e-6)

# ---- eq (6): the printed length-normalization factor ----
with np.errstate(divide="ignore"):
    for TT in (3, 4):
        t1 = np.arange(1, TT + 1)  # 1-based positions
        printed = 1.0 / (TT - t1 - 1).astype(float)
        print(f"T={TT}: 1/(T-t-1) for t=1..T = {printed}; terms in R_(t+1) = {TT - t1}")
        assert np.isinf(printed[TT - 2]) and printed[TT - 1] == -1  # t = T-1 divides by 0
# LMOps losses.py (commit 4c59a4e) divisor, rebuilt in NumPy: tokens left from t to the end, inclusive
mask = np.array([1.0, 1, 1, 0])
lens = mask - np.cumsum(mask) + np.cumsum(mask)[-1]
lens[lens == 0] = 1
print(f"LMOps divisor for mask {mask.astype(int)}: {lens}")
assert lens.tolist() == [3, 2, 1, 1]

# ---- eq (7): everything combined ----
def eq7(alpha_):
    ptil = lambda x, y, t: mix(theta, x, y, t, alpha_)
    wt = lambda x, y, t: dist(theta, x, y, t)[y[t]] / ptil(x, y, t)[y[t]]
    rnorm = lambda x, y, t: R(theta, x, y, t + 1) / (T - 1 - t) if T - 1 - t > 0 else 0.0  # (6) modified at every t: divide by the T - t terms (1-based t), not T - t - 1
    return -expect(ptil, lambda x, y: sum(wt(x, y, t) * (dEr(theta, x, y, t) + rnorm(x, y, t) * dlogq(theta, x, y, t)) for t in range(T)))


G7 = eq7(alpha)
G7_no_mix = eq7(0.0)
print(f"eq (7), alpha=0.2: cos with grad L = {cos(G7, G):.4f}; alpha=0 (only length norm.): {cos(G7_no_mix, G):.4f}")
assert not np.allclose(G7, G, atol=1e-3)
eta = 1e-2
print(f"loss after one step of size {eta}: start {L0:.6f}, along -grad L {L(theta - eta * G):.6f}, along -eq7 {L(theta - eta * G7):.6f}")
assert L(theta - eta * G7) < L0

# ---- Algorithm 1 clipping as the LMOps code forms it: rho_t = q_theta / p~ starts at w_t, not at 1 ----
eps = 0.2


def clip_loss(q, q_old, p, A):  # max(-A rho, -A clip(rho, 1-eps, 1+eps)); rho = q / p~, p~ fixed when collected
    rho = q / (alpha * p + (1 - alpha) * q_old)
    return max(-A * rho, -A * min(max(rho, 1 - eps), 1 + eps))


def slope(q, q_old, p, A, h=1e-9):  # d loss / d q_theta at q_theta = q
    return (clip_loss(q + h, q_old, p, A) - clip_loss(q - h, q_old, p, A)) / (2 * h)


zero = {(0.05, 0.5): -1.0, (0.9, 0.05): 1.0}  # starting rho below / above [0.8, 1.2]: the sign of A that gets no gradient
for q_old, p_ in [(0.05, 0.5), (0.3, 0.3), (0.9, 0.05)]:  # starting rho below, inside and above [0.8, 1.2]
    w0 = q_old / (alpha * p_ + (1 - alpha) * q_old)
    s_pos, s_neg = slope(q_old, q_old, p_, 1.0), slope(q_old, q_old, p_, -1.0)
    print(f"q_old={q_old}, p={p_}: rho before any update = w_t = {w0:.4f}; slope with A=+1 {s_pos:.4f}, with A=-1 {s_neg:.4f}")
    for A in (1.0, -1.0):
        assert (slope(q_old, q_old, p_, A) == 0) == (zero.get((q_old, p_)) == A)
ptil0 = alpha * 0.5 + (1 - alpha) * 0.05
q_cap = (1 + eps) * ptil0
assert slope(q_cap - 1e-4, 0.05, 0.5, 1.0) < 0 and slope(q_cap + 1e-4, 0.05, 0.5, 1.0) == 0
print(f"q_old=0.05, p=0.5, A=+1: the gradient stops at q_theta = 1.2 * p~ = {q_cap:.4f} = {q_cap / 0.05:.2f} x q_old")
grid = np.linspace(1e-3, 1, 400)
w_max = max(qo / (alpha * pp + (1 - alpha) * qo) for qo in grid for pp in grid)
w_p0 = 0.3 / (alpha * 0.0 + (1 - alpha) * 0.3)  # a teacher probability of 0
assert w_max < w_p0 and np.isclose(w_p0, 1 / (1 - alpha))
print(f"w_t on a grid of q_old, p in (0, 1]: max {w_max:.4f}; with p = 0: {w_p0:.4f} = 1/(1-alpha)")

# ---- DeepSeek-V4 eq (29) at one token position, vocabulary of 3 ----
qs = np.array([0.5, 0.25, 0.25])  # student pi_theta(. | x, y_<t)
E = np.array([[0.25, 0.5, 0.25], [0.8, 0.1, 0.1]])  # experts pi_E1, pi_E2
w = np.array([0.75, 0.25])
rev = np.array([np.sum(qs * np.log(qs / e)) for e in E])  # D_KL(pi_theta || pi_Ei)
fwd = np.array([np.sum(e * np.log(e / qs)) for e in E])  # D_KL(pi_Ei || pi_theta), for contrast
L_opd = float(w @ rev)
assert np.isclose(rev[0], 0.25 * np.log(2)) and np.isclose(rev[1], np.log(1.25))
assert np.isclose(L_opd, 0.1875 * np.log(2) + 0.25 * np.log(1.25))
print(f"eq (29): reverse KLs {rev}, forward KLs {fwd}, L_OPD = {L_opd:.4f}")
assert np.isclose(np.sum(qs * np.log(qs / qs)), 0.0)
with np.errstate(divide="ignore"):
    E3 = np.array([2 / 3, 1 / 3, 0.0])
    rev3 = np.sum(qs * np.log(qs / E3))
    fwd3 = np.sum(np.where(E3 > 0, E3 * np.log(np.where(E3 > 0, E3, 1) / qs), 0.0))
    assert np.isinf(rev3) and np.isclose(fwd3, np.log(4 / 3))
    print(f"expert with a zero where the student is 0.25: reverse KL {rev3}, forward KL {fwd3:.4f} = log(4/3)")

# full-vocabulary gradient vs the one-sampled-token "sg advantage" estimate (prior works)
z = np.log(qs)  # student logits at this position
kl_z = lambda zz: float(np.sum(softmax(zz) * np.log(softmax(zz) / E[0])))
g_full = qs * (np.log(qs / E[0]) - rev[0])  # gradient of the exact per-token KL
assert np.allclose(g_full, grad_fd(kl_z, z), atol=1e-8)
g_samp = [-np.log(E[0][k] / qs[k]) * (np.eye(3)[k] - qs) for k in range(3)]  # grad of -sg[log pE/q] * log q(k)
mean_samp = sum(qs[k] * g_samp[k] for k in range(3))
var_samp = sum(qs[k] * np.sum((g_samp[k] - g_full) ** 2) for k in range(3))
val_samp = np.log(qs / E[0])
assert np.allclose(mean_samp, g_full)
print(f"per-token gradient: full vocabulary {g_full}; mean of 1-sample estimate {mean_samp}")
print(f"1-sample estimate: gradient variance (trace) {var_samp:.4f}; value log(q/pE) per token {val_samp}, "
      f"mean {qs @ val_samp:.4f}, variance {qs @ (val_samp - rev[0]) ** 2:.4f}")
# MiniLLM Table 4 (validation R-L, DollyEval R-L): gap of the full method over "w/o Single-Step"
full, no_single = np.array([27.4, 24.6]), np.array([27.0, 23.7])
print(f"Table 4 gap, full minus w/o Single-Step: {np.round(full - no_single, 1)}")
print("all asserts passed")
```

Output:

```text
eq (1) exact L = 0.6619; chain-rule sum of per-token KLs = 0.6619
eq (1) Monte Carlo, 20000 student samples: 0.6636 +- 0.0086 (standard error)
|grad L| (finite differences) = 0.4174
eq (2) matches: True; size of the '-1' term: 6.7e-17
eq (3) Single + Long matches: True
Single part alone: cos with grad L = 0.9598, |Single| = 0.3781
eq (5), alpha=0.2, w_t = product up to t             : Single exact True  Long exact False cos(sum, grad L) = 0.9947
eq (5), alpha=0.2, w_t = product over whole response : Single exact True  Long exact True  cos(sum, grad L) = 1.0000
eq (5), alpha=0.2, w_t = per-token q/p~              : Single exact False Long exact False cos(sum, grad L) = 0.9924
T=3: 1/(T-t-1) for t=1..T = [ 1. inf -1.]; terms in R_(t+1) = [2 1 0]
T=4: 1/(T-t-1) for t=1..T = [ 0.5  1.   inf -1. ]; terms in R_(t+1) = [3 2 1 0]
LMOps divisor for mask [1 1 1 0]: [3. 2. 1. 1.]
eq (7), alpha=0.2: cos with grad L = 0.9806; alpha=0 (only length norm.): 0.9913
loss after one step of size 0.01: start 0.661932, along -grad L 0.660191, along -eq7 0.660382
q_old=0.05, p=0.5: rho before any update = w_t = 0.3571; slope with A=+1 -7.1429, with A=-1 0.0000
q_old=0.3, p=0.3: rho before any update = w_t = 1.0000; slope with A=+1 -3.3333, with A=-1 3.3333
q_old=0.9, p=0.05: rho before any update = w_t = 1.2329; slope with A=+1 0.0000, with A=-1 1.3699
q_old=0.05, p=0.5, A=+1: the gradient stops at q_theta = 1.2 * p~ = 0.1680 = 3.36 x q_old
w_t on a grid of q_old, p in (0, 1]: max 1.2497; with p = 0: 1.2500 = 1/(1-alpha)
eq (29): reverse KLs [0.1733 0.2231], forward KLs [0.1733 0.1927], L_OPD = 0.1858
expert with a zero where the student is 0.25: reverse KL inf, forward KL 0.2877 = log(4/3)
per-token gradient: full vocabulary [ 0.2599 -0.2166 -0.0433]; mean of 1-sample estimate [ 0.2599 -0.2166 -0.0433]
1-sample estimate: gradient variance (trace) 0.0788; value log(q/pE) per token [ 0.6931 -0.6931  0.    ], mean 0.1733, variance 0.3303
Table 4 gap, full minus w/o Single-Step: [0.4 0.9]
all asserts passed
```

What the numbers show, MiniLLM:

- Equation (1) is 0.6619; a batch mean over 20000 student samples gives
  0.6636 ± 0.0086, and the sum of per-token KLs gives 0.6619 again.
- Equations (2) and (3) match the finite-difference gradient exactly; the
  "$-1$" term averages to zero ($6.7\times10^{-17}$); the Single part alone
  has cosine similarity 0.96 with it (the cosine of the angle between the
  two gradients as vectors; 1 means the same direction).
- Equation (5) with $w_t$ as printed (product up to $t$) matches the Single
  part but not the Long part when $\alpha>0$; a product over the whole
  response matches both. The Long term at $t$ contains $R_{t+1}$, which uses
  the later tokens $y_{t+1},\dots,y_T$, also drawn from $\widetilde p$, so its
  weight must run over them too (my explanation). So the paper's "unbiased
  estimator" (one whose average equals the exact gradient) holds here for the Long part only with the whole-response
  product (my finding). The per-token approximation the paper adopts matches
  neither (cosine 0.9924): it corrects only position $t$, and both parts also
  depend on the earlier tokens.
- Equation (6)'s factor is infinite at $t=T-1$ and multiplies an empty sum at
  $t=T$; the official code divides by the tokens left, counting $t$
  (`[3, 2, 1, 1]`).
- Equation (7), with (6) changed at every position to divide by the $T-t$
  future terms, so that $R^{\text{Norm}}_{t+1}$ is their mean (my change; at
  $t=1$ with $T=3$ the printed (6) divides by 1, the check by 2), has cosine
  0.98 with the true gradient, and one step of 0.01 along it still lowers the
  loss (0.661932 to 0.660382, against 0.660191 along the true gradient).
  These numbers are for the changed (6), not the printed one, which cannot be
  evaluated at $t=T-1$.
- The clip, before any update: a token with $q_{\text{old}}=0.05$ and $p=0.5$
  starts at $\rho_t=w_t=0.3571$ and gets no gradient with $A=-1$; with $A=+1$
  it keeps a gradient until $q_\theta$ reaches $1.2\,\widetilde p=0.168$, 3.36
  times $q_{\text{old}}$. One with $q_{\text{old}}=0.9$ and $p=0.05$ starts
  at 1.2329 and gets none with $A=+1$; one starting at 1 gets a gradient with either sign. $w_t$ stays
  below $1/(1-\alpha)=1.25$ (grid maximum 1.2497) and reaches it where the
  teacher probability is 0.

What the numbers show, DeepSeek-V4:

- $\mathcal L_{\text{OPD}}=0.75\cdot0.1733+0.25\cdot0.2231=0.1858$; with the
  KL order swapped the second term is 0.1927, and an expert with probability
  0 where the student has 0.25 makes the reverse KL infinite (forward:
  $\log\frac43=0.2877$).
- At one position, the prior works' one-sampled-token estimate has the same
  average gradient as the full-vocabulary KL but a variance of 0.0788, against
  none. This illustrates the direction of the paper's statement in a toy; it
  does not measure DeepSeek-V4, and the prefixes are still sampled in both
  methods.

## What the code must decide

MiniLLM. The official code is at commit `4c59a4e`. `losses.py`, `sampler.py`,
`reward.py`, `utils.py` and `trainer.py` are in `minillm/minillm/`;
`arguments.py`, `install.sh`, `train_minillm.py`, a second `utils.py`
(written `minillm/utils.py` below) and the run script
`scripts/gpt2/minillm/train_base_xl.sh`, from which the run settings below
come, are in `minillm/`. What a PyTorch or DeepSpeed function named below does
is taken from that library's documentation or source at the version `install.sh`
installs (`torch==2.0.1`, `deepspeed==0.10.0`); the pages are listed under
Sources.

- **Discount.** The sum behind $R$ multiplies each later term by
  $\gamma=0.95$ per step (`losses.py` line 45; `--gamma` defaults to 0.95 in
  `arguments.py` line 158, not overridden by the script).
  Equations (2) and (6) have no discount.
- **What the Long part's multiplier sums, and the divisor.** The code's
  counterpart of $R^{\text{Norm}}_{t+1}$ is called the advantage
  (`advantages`, `losses.py` line 51); $A$ below. The code sums from $t$
  itself, not $t+1$, and divides by the number of response tokens from $t$ to
  the end (`losses.py` lines 39–49, `torch.cumsum` and `torch.masked_fill`).
  Equation (6) sums from $t+1$ and divides by $T-t-1$. Because the code also
  adds the single-step KL loss (`losses.py` lines 165–169), the current
  token's $r_t$ is in both the advantage and the exact KL term; equation (3)
  moves it from one to the other.
- **Whitening.** The advantages are shifted to mean 0 and scaled by
  $1/\sqrt{\text{var}+10^{-8}}$ (`torch.rsqrt`) over the batch before use
  (`losses.py` lines 53–54, `utils.py` lines 99–109). The variance divides by
  $n$, the number of entries summed over all processes (`get_global_statistics`,
  `utils.py` lines 84–96, `dist.all_reduce` with `dist.ReduceOp.SUM`).
  `whiten` takes that branch whenever `torch.distributed` is initialized
  (`utils.py` line 101), and the training entry point always initializes it:
  `main` calls `initialize` (`train_minillm.py` line 58), which, through
  `init_distributed_ds` or `init_distributed`, calls either
  `deepspeed.init_distributed` or `dist.init_process_group`
  (`minillm/utils.py` lines 100, 117 and 120–125). The other branch,
  `torch.var_mean` (`utils.py` line 104), does not run in training. Equation
  (7) has no whitening.
- **Scale of $r_t$.** The code's per-token `rewards` are
  $r_t=\log p-\log q_{\text{old}}$ (`sampler.py` lines 77–78 and 122–124;
  `reward.py` lines 57–69), divided by 0.5 and clipped to $\pm100$
  (`sampler.py` lines 126–131, `torch.clip`; the script's
  `--reward-scaling 0.5` and `--cliprange-reward 100`).
- **$w_t$.** One ratio per token, $q_{\text{old}}/\widetilde p$
  (`sampler.py` lines 108–115; held fixed: "Where the samples come from"), as
  in the paper's approximation.
- **$w_t$ on the Single part.** The code does not apply it: `_reg_loss` is a
  plain per-token average of $\operatorname{KL}[q_\theta||p]$, computed as
  the cross-entropy $-\sum_{y'}q_\theta\log p$ minus the entropy
  $-\sum_{y'}q_\theta\log q_\theta$ (`losses.py` lines 98–108). Equation (7) and
  Algorithm 1 multiply this part by $w_t$.
- **Clipping.** The per-token loss is
  $\max\big(-A\rho_t,\,-A\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\big)$,
  elementwise (`losses.py` lines 88–94, `torch.max` and `torch.clamp`), with
  $\epsilon=0.2$, the `--cliprange` default (`arguments.py` line 156, not
  overridden by the script). §2.3
  adds clipping "to further improve stability". As in Algorithm 1, the
  clipped quantity is $\rho_t$ (how the code forms it: "Where the samples
  come from"), so before any update it equals $w_t$, not 1. The clip
  therefore bounds $q_\theta$ relative to $\widetilde p$, not relative to the
  copy that collected the batch. A token whose $w_t$ is already outside
  $[0.8,1.2]$ on the side its advantage pushes toward (above 1.2 with $A>0$,
  below 0.8 with $A<0$) gives no gradient in an update made before the
  weights move: in the script, the one update of inner epoch 1 (see "Where
  the samples come from"). Worked numbers and the range of $w_t$: "What the
  numbers show". Algorithm 1 prints $R^{\text{Norm}}_{t+1}$ outside the
  $\min$, where the code multiplies before comparing. Equation (7) has no
  clipping.
- **Averaging.** Both losses are averaged over all response tokens in each
  process's batch (`losses.py` lines 70, 94 and 105; the token count is not
  summed across processes), so every token in that batch has equal weight.
  Equation (7) sums over $t$ inside each response, and Algorithm 1 then
  divides by $M$.
- **Log, softmax axis, stability.** Natural logarithms: `torch.log` is
  documented as "the natural logarithm", `F.log_softmax` as "mathematically
  equivalent to log(softmax(x))", and `sampler.py` line 112 turns
  log-probabilities back into probabilities with `torch.exp`, documented as
  $e^{x_i}$ (`F.log_softmax` at `utils.py` lines 32, 50 and 71; `torch.log` at
  `sampler.py` line 113). `F.log_softmax` runs over the vocabulary axis
  (`dim=-1`). The entropy and cross-entropy of the Single part are computed
  in float32 (`utils.py` lines 31–32, 70–71). The sampled-token
  log-probabilities used in $\rho_t$ (`get_log_probs`, line 50, called at
  `losses.py` line 150) are computed without `dtype=torch.float32`; only
  their difference is cast to float32 before `exp` (`losses.py` line 73).
  Vocabulary entries set to
  $-\infty$ during sampling are zeroed in the $q\log$ products (`utils.py`
  lines 23–75, `torch.isinf` and `masked_fill`).
- **Teacher mixing.** It is passed as `mix_in_model` and `mix_in_alpha` to a
  `generate` function in the repository's own copy of `transformers`
  (`trainer.py` lines 593–624). I did not open that file, so how a token is
  drawn from the mix is open.
- **Language-modelling term.** The script passes `--kd-ratio 0.5`, so
  `pt_loss` is half next-token cross-entropy and half word-level distillation
  toward the teacher on the $\mathcal D_{\text{PT}}$ batch (`losses.py`
  lines 202–222, `nn.CrossEntropyLoss`), added with weight `lm_coef`, default
  1 (`trainer.py` line 283, `arguments.py` line 169).
  Algorithm 1 has only $-\log q_\theta(\boldsymbol d)$.
- **Sampling.** Temperature 1 (script line 84), the value Appendix B.1
  states. The script also passes `--top-k 0` and `--top-p 1.0` (lines
  82–83), which go with the temperature to the same `generate` (`trainer.py`
  lines 89–93 and 615–625), so whether they leave the vocabulary untruncated
  is open, as for the teacher mixing. Inner epochs, rollouts and update size:
  "Where the samples come from".

DeepSeek-V4 links no training code, so every choice is open:

- how per-position KLs are combined, and the values and use of $w_i$: see
  "Not defined in the paper"; whether a Long-type term is added: see the
  reading of equation (29);
- which student copy generates the trajectories, how stale it may be, and
  whether any importance weight or clipping corrects for that;
- the precision and the log-sum-exp handling (subtracting the largest logit
  before exponentiating, so nothing overflows) inside the "specialized
  TileLang kernel" (§5.2.2);
- the sampling temperature, and whether the teacher logits use the same one.

## What the paper claims and what it shows

MiniLLM:

- Reverse KL makes the student focus on the teacher's "major modes" and give
  low probability to its "void regions": supported by cited work ("has been
  shown to cause the mode-seeking behavior"), illustrated by a toy fit of one
  Gaussian (bell curve) to a mixture of Gaussians (Figure 2); for language models the paper studies it
  experimentally ("In this work, we first study this property", §2.1).
- Equations (2) and (3): derived (Appendices A.2 and A.3), under the stated
  fact that "$\log q_\theta(y_t|\boldsymbol y_{<t},\boldsymbol x)$ can only
  affect tokens at $\ge t$ positions in $\boldsymbol y$". The check
  reproduces both.
- Equation (5) is "an unbiased estimator of the gradient": stated, with a
  citation (Precup, Sutton and Singh, 2000). Whether it holds in my check
  depends on which $w_t$ is used (see "What the numbers show"). The paper
  itself then replaces $w_t$ by an approximation "to reduce the variance".
- $R_t$ "favors short sentences, which causes the student model to output
  empty responses", and "long sequences tend to have small $R_{t+1}$":
  observations ("we notice", "We found"); the ablation in the next item is the
  only related measurement.
- The three strategies help: measured. They are ablated (removed one at a
  time) in a single experiment,
  GPT-2-1.5B teacher and GPT-2-125M student (§3.4, Table 4). Table 4 gives
  validation Rouge-L, a score the paper uses "to measure the precision of
  the model generation" (§3.1), of 27.4 for the full method against 17.4 without length
  normalization, 22.3 without teacher-mixed sampling and 27.0 without
  single-step decomposition; on DollyEval, 24.6, 14.7, 20.4 and 23.7.
  Teacher-mixed sampling is also swept over $\alpha$, where "$\alpha=0.0$
  means we only sample from the student distribution", for three student and
  teacher pairs: GPT-2-125M, OPT-1.3B and LLaMA-7B with GPT-2-1.5B, OPT-13B
  and LLaMA-13B teachers (Appendix C.4, Figure 16); the paper finds "$\alpha =
  0.2$ is generally suitable across different model families and sizes".
  Neither Table 4 nor Figure 16 gives a ± or a number of runs, so the 0.4 and
  0.9 gaps for single-step decomposition come with no interval. "Single-Step Decomposition effectively reduces the variance of the
  training process" is read off training curves "smoothed by 32 steps"
  (Figure 8).
- Main results: Table 1 reports averages "across 5 random seeds", with no ±.

DeepSeek-V4:

- The prior per-token estimate "leads to high variance in gradient estimation
  and often causes training instability", and full-vocabulary distillation
  "yields more stable gradient estimates": stated, with no experiment. I found
  no ablation of this choice in the PDF text (searched for "variance",
  "instability", "ablat" and "full-vocab"). The check above shows the
  per-position effect in a toy only.
- OPD is "practically circumventing the performance degradation often
  encountered in traditional weight-merging or mixed RL techniques": stated;
  §5 shows no comparison.
- Caching teacher hidden states "incurs negligible recomputation overhead":
  stated; §5.2.2 gives no measurement.
- "more than ten teacher models" and $|V|>100\text{k}$: stated facts of the
  setup.

## Sources

- Gu, Dong, Wei, Huang, MiniLLM, [arXiv 2306.08543v4](https://arxiv.org/abs/2306.08543v4)
  (v4 posted 2024-04-10; v5 and v6 exist, posted 2025-11-21 and 2026-01-31, not
  used): TeX source and PDF.
- DeepSeek-AI, DeepSeek-V4, [arXiv 2606.19348v1](https://arxiv.org/abs/2606.19348v1)
  (posted 2026-04-26, the only version): TeX source and PDF.
- `microsoft/LMOps` at commit `4c59a4e1bcb239d2eeb823d015c7aa00222e0976`:
  [`minillm/minillm/losses.py`](https://github.com/microsoft/LMOps/blob/4c59a4e1bcb239d2eeb823d015c7aa00222e0976/minillm/minillm/losses.py),
  [`sampler.py`](https://github.com/microsoft/LMOps/blob/4c59a4e1bcb239d2eeb823d015c7aa00222e0976/minillm/minillm/sampler.py),
  [`reward.py`](https://github.com/microsoft/LMOps/blob/4c59a4e1bcb239d2eeb823d015c7aa00222e0976/minillm/minillm/reward.py),
  [`utils.py`](https://github.com/microsoft/LMOps/blob/4c59a4e1bcb239d2eeb823d015c7aa00222e0976/minillm/minillm/utils.py),
  [`trainer.py`](https://github.com/microsoft/LMOps/blob/4c59a4e1bcb239d2eeb823d015c7aa00222e0976/minillm/minillm/trainer.py),
  [`minillm/arguments.py`](https://github.com/microsoft/LMOps/blob/4c59a4e1bcb239d2eeb823d015c7aa00222e0976/minillm/arguments.py),
  [`minillm/scripts/gpt2/minillm/train_base_xl.sh`](https://github.com/microsoft/LMOps/blob/4c59a4e1bcb239d2eeb823d015c7aa00222e0976/minillm/scripts/gpt2/minillm/train_base_xl.sh),
  [`minillm/train_minillm.py`](https://github.com/microsoft/LMOps/blob/4c59a4e1bcb239d2eeb823d015c7aa00222e0976/minillm/train_minillm.py),
  [`minillm/utils.py`](https://github.com/microsoft/LMOps/blob/4c59a4e1bcb239d2eeb823d015c7aa00222e0976/minillm/utils.py),
  [`minillm/install.sh`](https://github.com/microsoft/LMOps/blob/4c59a4e1bcb239d2eeb823d015c7aa00222e0976/minillm/install.sh),
  [`minillm/configs/deepspeed/ds_config.json`](https://github.com/microsoft/LMOps/blob/4c59a4e1bcb239d2eeb823d015c7aa00222e0976/minillm/configs/deepspeed/ds_config.json).
- PyTorch 2.0.1 documentation, read as the docstrings it is built from, at
  tag `v2.0.1`:
  [`torch/_torch_docs.py`](https://github.com/pytorch/pytorch/blob/v2.0.1/torch/_torch_docs.py)
  (`torch.log`, `torch.exp`, `torch.clamp`, `torch.clip`, `torch.max`,
  `torch.maximum`, `torch.cumsum`, `torch.rsqrt`, `torch.isinf`,
  `torch.numel`),
  [`torch/_tensor_docs.py`](https://github.com/pytorch/pytorch/blob/v2.0.1/torch/_tensor_docs.py)
  (`masked_fill`, `masked_fill_`, `float`),
  [`torch/nn/functional.py`](https://github.com/pytorch/pytorch/blob/v2.0.1/torch/nn/functional.py)
  (`softmax`, `log_softmax`),
  [`torch/nn/modules/loss.py`](https://github.com/pytorch/pytorch/blob/v2.0.1/torch/nn/modules/loss.py)
  (`CrossEntropyLoss`),
  [`torch/autograd/grad_mode.py`](https://github.com/pytorch/pytorch/blob/v2.0.1/torch/autograd/grad_mode.py)
  (`no_grad`),
  [`torch/distributed/distributed_c10d.py`](https://github.com/pytorch/pytorch/blob/v2.0.1/torch/distributed/distributed_c10d.py)
  (`is_initialized`, `init_process_group`, `all_reduce`); and the rendered
  2.0 page for
  [`Tensor.detach`](https://docs.pytorch.org/docs/2.0/generated/torch.Tensor.detach.html).
- DeepSpeed 0.10.0 documentation for `deepspeed.init_distributed`, at tag
  `v0.10.0`:
  [`docs/code-docs/source/initialize.rst`](https://github.com/microsoft/DeepSpeed/blob/v0.10.0/docs/code-docs/source/initialize.rst)
  ("Useful in scenarios where the user wants to use torch distributed calls
  before calling `deepspeed.initialize()`") and its docstring in
  [`deepspeed/comm/comm.py`](https://github.com/microsoft/DeepSpeed/blob/v0.10.0/deepspeed/comm/comm.py);
  and the DeepSpeed source at the same tag for the engine's `step`,
  `_take_model_step` and `is_gradient_accumulation_boundary`,
  [`deepspeed/runtime/engine.py`](https://github.com/microsoft/DeepSpeed/blob/v0.10.0/deepspeed/runtime/engine.py),
  and for the batch settings,
  [`deepspeed/runtime/config.py`](https://github.com/microsoft/DeepSpeed/blob/v0.10.0/deepspeed/runtime/config.py)
  (`_initialize_params`, `_set_batch_related_parameters`) and
  [`deepspeed/runtime/constants.py`](https://github.com/microsoft/DeepSpeed/blob/v0.10.0/deepspeed/runtime/constants.py).
- Named, not opened: K. Lu and Thinking Machines Lab, "On-policy distillation"
  (2025), cited by DeepSeek-V4 for OPD; Precup, Sutton and Singh, "Eligibility
  traces for off-policy policy evaluation" (ICML 2000), cited by MiniLLM for
  equation (5); Sutton et al. (1999) and Williams (1992), cited by MiniLLM for
  the policy gradient.
- No documentation is cited for `torch.gather`, `torch.logsumexp`, or
  `torch.sum` and `Tensor.sum` (a sum along `dim` when one is given, over
  every element otherwise): the code uses each only in its basic meaning.
