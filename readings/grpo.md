# GRPO: score each sampled answer against the other answers to the same question, then make the better answers more likely

**Source:** DeepSeek-AI,
[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948v1),
v1, §2.2.1, equations (1)–(3); also §2.2.2 (rewards). Z. Shao et al.,
[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300v3),
v3, §4.1.1, equations (1)–(4), with Algorithm 1; also Figure 4, §4.1.2
(advantage), §4.1.4, §4.2 (settings), §5.2.1 (iterative RL), Table 10,
Appendix A.1.5, and Appendix A.1.6 with equations (19)–(21). Code: DeepSeekMath links
`github.com/deepseek-ai/DeepSeek-Math`. At commit `b8b0f8c` (15 Apr 2024, the
last one before v3) it holds only evaluation and inference code
(`evaluation/`, `replicate/predict.py`) and no RL training code. DeepSeek-R1
v1 links no code.

## What it computes and what it changes

For one question, the model writes $G$ answers, and each answer gets a score
called its _reward_. In this one-reward-per-answer form, which DeepSeek-R1 v1
uses and DeepSeekMath calls _outcome supervision_ (§4.1.2), GRPO turns each
reward into a measure of how much better
the answer did than the other answers to the same question: the reward minus the group's average,
divided by the group's standard deviation. Then it changes the weights so that
the tokens of above-average answers become more likely and the tokens of
below-average answers less likely. Two brakes limit each step. A _clip_ stops
a token from adding more to the number that training maximizes once its
probability has moved by more than a fraction $\epsilon$ since the answers were
sampled, in the direction its answer favours: up for an above-average answer,
down for a below-average one. In R1 v1 the clip acts on the probability of
the whole answer instead (end of the next paragraph). A penalty, weighted by
$\beta$, grows as the model moves away from a fixed _reference model_. The equation
defines the single number that training maximizes. DeepSeekMath also gives a
form that scores each reasoning step rather than the whole answer, _process
supervision_ (§4.1.3); step 3 of the inside-out reading gives its rule.

DeepSeekMath prints the method GRPO changes, PPO, as equation (1), with PPO's
KL penalty in equation (2). The main difference is where the _advantage_ comes
from. The advantage is the number that says how much better a token's choice
was than expected. PPO computes it with a second network, a value model
$V_\psi$, which must be trained alongside. GRPO computes
it from the rewards of the $G$ answers to the same question, so it needs no
value model. A second difference is the penalty for moving away from the
reference model. PPO subtracts it from the reward at every token (equation
(2)), and DeepSeekMath calls it the "KL penalty" (KL is explained in step 5 of
the inside-out reading). GRPO subtracts $\beta$ times the estimate (4) directly
from the objective (3). DeepSeek-R1 v1 gives the penalty no name in words:
"KL" occurs only inside its equations (1) and (2). Finally, DeepSeek-R1 v1 writes GRPO per answer: one
probability ratio and one KL term for the whole answer $o_i$. DeepSeekMath
writes it per token and averages over each answer's tokens. The check below
shows that the two forms give different numbers.

## The equation

DeepSeek-R1 v1, §2.2.1:

$$
\begin{aligned}
\mathcal{J}_{GRPO}(\theta) &= \mathbb{E}{[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)]} \\
& \frac{1}{G}\sum_{i=1}^G \left( \min \left( \frac{\pi_\theta(o_i |q)}{\pi_{\theta_{old}}(o_i |q)} A_i, \text{clip} \left( \frac{\pi_\theta(o_i |q)}{\pi_{\theta_{old}}(o_i |q)}, 1 - \epsilon, 1 + \epsilon \right) A_i \right) - \beta \mathbb{D}_{KL}\left(\pi_{\theta} || \pi_{ref}\right)\right) ,
\end{aligned}
\tag{1}
$$

$$
\mathbb{D}_{KL}\left(\pi_{\theta} || \pi_{ref}\right) = \frac{\pi_{ref}(o_i|q)}{\pi_{\theta}(o_i|q)}- \log\frac{\pi_{ref}(o_i|q)}{\pi_{\theta}(o_i|q)} - 1,
\tag{2}
$$

$$
A_i = \frac{r_i - {\mathrm{mean}(\{r_1, r_2, \cdots, r_G\})}}{{\mathrm{std}(\{r_1, r_2, \cdots, r_G\})}}.
\tag{3}
$$

DeepSeekMath v3, §4.1.1 (PPO, then GRPO):

$$
\mathcal{J}_{PPO}(\theta) = \mathbb{E}{[q \sim P(Q), o \sim \pi_{\theta_{old}}(O|q)]} \frac{1}{|o|} \sum_{t=1}^{|o|} \min \left[ \frac{\pi_\theta(o_{t} | q, o_{<t})}{\pi_{\theta_{old}}(o_{t} | q, o_{<t})} A_{t}, \text{clip} \left( \frac{\pi_\theta(o_{t} | q, o_{<t})}{\pi_{\theta_{old}}(o_{t} | q, o_{<t})}, 1 - \epsilon, 1 + \epsilon \right) A_{t} \right] ,
\tag{1}
$$

$$
r_{t} = r_\phi(q, o_{\le t}) - \beta \log\frac{\pi_{\theta}(o_{t}|q, o_{<t})}{\pi_{ref}(o_{t}|q, o_{<t})},
\tag{2}
$$

$$
\begin{aligned}
\mathcal{J}_{GRPO}(\theta) &= \mathbb{E}{[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)]} \\
& \frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min \left[ \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})} \hat{A}_{i,t}, \text{clip} \left( \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})}, 1 - \epsilon, 1 + \epsilon \right) \hat{A}_{i,t} \right] - \beta \mathbb{D}_{KL}\left[\pi_{\theta} || \pi_{ref}\right]\right\} ,
\end{aligned}
\tag{3}
$$

$$
\mathbb{D}_{KL}\left[\pi_{\theta} || \pi_{ref}\right] = \frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}- \log\frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})} - 1,
\tag{4}
$$

DeepSeekMath v3, Algorithm 1 (Iterative Group Relative Policy Optimization):

```text
Input initial policy model π_θinit; reward models r_φ; task prompts D; hyperparameters ε, β, μ
 1: policy model π_θ ← π_θinit
 2: for iteration = 1, ..., I do
 3:     reference model π_ref ← π_θ
 4:     for step = 1, ..., M do
 5:         Sample a batch D_b from D
 6:         Update the old policy model π_θold ← π_θ
 7:         Sample G outputs {o_i}_{i=1}^G ~ π_θold(· | q) for each question q ∈ D_b
 8:         Compute rewards {r_i}_{i=1}^G for each sampled output o_i by running r_φ
 9:         Compute Â_{i,t} for the t-th token of o_i through group relative advantage estimation.
10:         for GRPO iteration = 1, ..., μ do
11:             Update the policy model π_θ by maximizing the GRPO objective (Equation 21)
12:     Update r_φ through continuous training using a replay mechanism.
Output π_θ
```

Line 11 cites equation (21). In the printed paper, (21) is not the objective.
It is the _gradient coefficient_ of Appendix A.1.6,
$GC_{GRPO}=\hat A_{i,t}+\beta\left(\frac{\pi_{ref}(o_{i,t}|o_{i,<t})}{\pi_\theta(o_{i,t}|o_{i,<t})}-1\right)$:
the factor that multiplies $\nabla_\theta\log\pi_\theta(o_{i,t}|q,o_{i,<t})$,
the gradient of that token's log-probability with respect to the weights, in
the gradient (20). The TeX source shows that line 11 points to the label
`eq:GC-GRPO`. The objective that is maximized is (3): §4.1.2 says it
"optimizes the policy by maximizing the objective defined in equation (3)".

A.1.6 obtains (20) and (21) under one assumption, stated with its objective
(19): "(assume 𝜋𝜃𝑜𝑙𝑑 = 𝜋𝜃 for simplified analysis)". Why (21) follows
(my derivation): take one token's term of (3), the expression in its braces,
as a function of that token's log-probability
$\ell=\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})$. The ratio is
$e^{\ell}/\pi_{\theta_{old}}(o_{i,t}\mid q,o_{i,<t})$, so its derivative in
$\ell$ is the ratio itself. At $\pi_\theta=\pi_{\theta_{old}}$ the ratio is
1, strictly inside $[1-\epsilon,1+\epsilon]$, so near that point the min
equals the unclipped term, whose derivative is $1\cdot\hat A_{i,t}$. With
$x=\pi_{ref}/\pi_\theta=\pi_{ref}\,e^{-\ell}$, $dx/d\ell=-x$, so the
derivative of $-\beta(x-\log x-1)$ is $-\beta(-x+1)=\beta(x-1)$. The sum is
(21). The chain rule multiplies it by $\nabla_\theta\ell$, which gives that
token's term of (20). Check 6 below evaluates both sides at two points.
So on the first of the $\mu$ updates, when the two copies are equal,
maximizing (3) and the update rule that line 11 cites move the weights the
same way.

## Symbols

Throughout, "DSM" means DeepSeekMath v3 and "R1" means DeepSeek-R1 v1. Shapes
are given for one question. $T$ is the longest answer in the group; shorter
answers are padded (padding is the code's choice, see below).

| Symbol | Shape or type | Meaning | Role | Defined at |
| ------ | ------------- | ------- | ---- | ---------- |
| $\mathcal J_{GRPO}(\theta)$ | scalar | the number that training makes larger | computed intermediate (the result) | "optimizes the policy model 𝜋𝜃 by maximizing the following objective" (R1 §2.2.1); DSM §4.1.1 has the same sentence without "𝜋𝜃" |
| $\theta$, $\pi_\theta$ | $\theta$: all weights of the language model; $\pi_\theta$ of one token or one answer: a probability in $[0,1]$ | the model being trained. "Policy" means the model viewed as a rule for choosing the next token | trained weight | "𝜋𝜃 and 𝜋𝜃𝑜𝑙𝑑 are the current and old policy models" (DSM §4.1.1, after eq. (1)); Figure 4 draws the GRPO "Policy Model" under "Trained Models" |
| $\pi_{\theta_{old}}$ | same as $\pi_\theta$ | a copy of $\pi_\theta$ taken before sampling; it writes the answers. $\pi_{\theta_{old}}(\cdot\mid q)$ (Algorithm 1, line 7) is the whole distribution over answers to $q$; the dot is a placeholder for the answer | frozen copy | same quote; "Update the old policy model 𝜋𝜃𝑜𝑙𝑑 ← 𝜋𝜃" (Algorithm 1, line 6) |
| $\pi_{ref}$ | same as $\pi_\theta$ | a fixed model that the penalty keeps $\pi_\theta$ close to; when it is replaced: "Where the samples come from" | frozen model | "𝜋𝑟𝑒 𝑓 is the reference model, which is usually the initial SFT model" (DSM §4.1.1, after eq. (2); SFT is supervised fine-tuning); "reference model 𝜋𝑟𝑒 𝑓 ← 𝜋𝜃" (Algorithm 1, line 3); Figure 4 draws "Reference Model" under "Frozen Models" |
| $q$ | one question: a token sequence | the prompt | random sample | "𝑞, 𝑜 are questions and outputs sampled from the question dataset and the old policy 𝜋𝜃𝑜𝑙𝑑 , respectively" (DSM §4.1.1, after eq. (1)) |
| $P(Q)$ | a distribution over questions | the question dataset | input data | same quote; DSM §4.2 gives the set: "around 144K questions" |
| $G$ | integer | number of answers sampled per question | fixed setting | "GRPO samples a group of outputs { 𝑜1 , 𝑜2 , · · · , 𝑜𝐺 } from the old policy 𝜋𝜃𝑜𝑙𝑑" (R1 §2.2.1); "For each question, we sample 64 outputs." (DSM §4.2) |
| $o_i$; $\{o_i\}_{i=1}^G$ | one token sequence of length $\vert o_i\vert$; $G$ of them | answer $i$; the group of answers | random sample | same R1 quote |
| $i$ | integer, $1\ldots G$ | which answer | index | the sums $\sum_{i=1}^G$ in R1 (1) and DSM (3) |
| $t$; $o_{i,t}$ | integer, $1\ldots\vert o_i\vert$; one token id | position in answer $i$; the token there | index; random sample | "Compute 𝐴ˆ 𝑖,𝑡 for the 𝑡 -th token of 𝑜𝑖" (Algorithm 1, line 9) |
| $r_i$; $\mathbf r$ | scalar; $G$ numbers | the reward of answer $i$; all $G$ rewards | computed intermediate (from a frozen scorer) | "A reward model is then used to score the outputs, yielding 𝐺 rewards r = {𝑟1 , 𝑟2 , · · · , 𝑟𝐺 } correspondingly." (DSM §4.1.2). R1 uses rules instead: "Accuracy rewards: The accuracy reward model evaluates whether the response is correct." and "Format rewards" (R1 §2.2.2) |
| $r_\phi$ | a network | the reward model of DSM and PPO | frozen model during a step; retrained at Algorithm 1 line 12 | "𝑟𝜑 is the reward model" (DSM §4.1.1, after eq. (2)); "reward models 𝑟𝜑" (Algorithm 1, Input); Figure 4 draws "Reward Model" under "Frozen Models" |
| mean, std | outcome supervision: $G$ numbers → scalar; process supervision: all of $\mathbf R$ → scalar | group average and group standard deviation | operator, given a row because the divisor of std matters | "these rewards are normalized by subtracting the group average and dividing by the group standard deviation" (DSM §4.1.2, outcome supervision); "We also normalize these rewards with the average and the standard deviation" (DSM §4.1.3, process supervision); the divisor is in the list below |
| $A_i$ (R1) | scalar per answer | the advantage of answer $i$: its reward minus the group average, in units of the group's standard deviation | computed intermediate; no gradient flows through it | "𝐴𝑖 is the advantage, computed using a group of rewards {𝑟1 , 𝑟2 , . . . , 𝑟𝐺 } corresponding to the outputs within each group" (R1 §2.2.1, eq. (3)) |
| $\hat A_{i,t}$; $\tilde r_i$ | scalar per token, `[G, T]`; scalar per answer | the advantage of token $t$ of answer $i$; under outcome supervision, the normalized reward $\tilde r_i$, the same for every token of answer $i$ (process supervision: step 3 below) | computed intermediate; no gradient flows through it | "𝐴ˆ 𝑖,𝑡 is the advantage calculated based on relative rewards of the outputs inside each group only, which will be detailed in the following subsections" (DSM §4.1.1); "sets the advantages 𝐴ˆ 𝑖,𝑡 of all tokens in the output as the normalized reward" (DSM §4.1.2, outcome supervision); the formula in DSM §4.1.3 (process supervision). The hat marks an estimate, not a unit vector |
| $\mathbf R$; $r_i^{index(j)}$, $\tilde r_i^{index(j)}$; $index(j)$; $K_i$ (process supervision) | $K_1+\cdots+K_G$ numbers; scalar per step; integer token position; integer | all step rewards of the group; the reward of step $j$ of answer $i$, and its normalized value; where step $j$ ends; how many steps answer $i$ has | computed intermediate, all four (the rewards come from a frozen scorer) | "a process reward model is used to score each step of the outputs, yielding corresponding rewards: R = …, where 𝑖𝑛𝑑𝑒𝑥 ( 𝑗) is the end token index of the 𝑗-th step, and 𝐾𝑖 is the total number of steps in the 𝑖-th output" (DSM §4.1.3); $\tilde r_i^{index(j)}$ by the formula there. The superscript is a label, not a power |
| $\epsilon$ | scalar | how far the ratio may move before the clip stops rewarding it | fixed setting | "𝜀 and 𝛽 are hyper-parameters" (R1 §2.2.1 and DSM §4.1.1); "𝜀 is a clipping-related hyper-parameter introduced in PPO for stabilizing training." (DSM §4.1.1) |
| $\beta$ | scalar | weight of the KL penalty | fixed setting | "𝛽 is the coefficient of the KL penalty" (DSM §4.1.1, after eq. (2)); "The KL coefficient is 0.04." (DSM §4.2) |
| $\mathbb D_{KL}[\pi_\theta\Vert\pi_{ref}]$ | scalar per token, `[G, T]` (DSM); scalar per answer (R1) | an estimate, from the one sampled token (DSM) or the one sampled answer (R1), of how far $\pi_\theta$ has moved from $\pi_{ref}$ | computed intermediate | "we estimate the KL divergence with the following unbiased estimator (Schulman, 2020)" (DSM §4.1.1, eq. (4)); R1 eq. (2). The left side carries no $i,t$, but the right side depends on them |
| $\mathbb E[q\sim P(Q),\{o_i\}_{i=1}^G\sim\pi_{\theta_{old}}(O\vert q)]$ | operator → scalar | average over questions, and over the answers the old copy writes for each one; each $\sim$ means "is sampled from" | operator; the bracket names the sampling, not an argument | "for each question 𝑞, GRPO samples a group of outputs { 𝑜1 , 𝑜2 , · · · , 𝑜𝐺 } from the old policy 𝜋𝜃𝑜𝑙𝑑" (R1 §2.2.1); for $\sim$, the DSM §4.1.1 quote given for $q$, which says "sampled from" where DSM (1) has $\sim$ |
| $\mathcal J_{PPO}(\theta)$ | scalar | PPO's number that training makes larger | computed intermediate (the result of DSM (1)) | "it optimizes LLMs by maximizing the following surrogate objective" (DSM §4.1.1, before eq. (1)) |
| $o$, $o_t$ (PPO) | one answer; its token $t$ | the single answer PPO samples per question | random sample | the DSM §4.1.1 quote given for $q$ |
| $A_t$ (PPO) | scalar per token | PPO's advantage | computed intermediate | "𝐴𝑡 is the advantage, which is computed by applying Generalized Advantage Estimation (GAE) (Schulman et al., 2015), based on the rewards {𝑟 ≥ 𝑡 } and a learned value function 𝑉𝜓" (DSM §4.1.1; the same sentence is in A.1.5) |
| $V_\psi$ | a network | PPO's value model | trained weight | same quote; "a value function needs to be trained alongside the policy model" (DSM §4.1.1) |
| $r_t$ (PPO) | scalar per token | PPO's reward at token $t$ with the KL penalty already subtracted | computed intermediate | "the standard approach is to add a per-token KL penalty from a reference model in the reward at each token" (DSM §4.1.1, before eq. (2)) |
| $\mu$ | integer | number of updates made on one sampled batch | fixed setting | "hyperparameters 𝜀, 𝛽 , 𝜇" and "for GRPO iteration = 1, . . . , 𝜇 do" (Algorithm 1, Input and line 10) |
| $\mathcal D$; $\mathcal D_b$ | set of prompts; a batch of them | the training questions; one batch | input data; random sample | "task prompts D" (Algorithm 1, Input); "Sample a batch D𝑏 from D" (line 5) |
| $I$, $M$ | integers | outer and inner loop counts | fixed setting | only the loop lines: "for iteration = 1, . . . , I do" and "for step = 1, . . . , M do" (Algorithm 1, lines 2 and 4). DSM prints no value. For the iterative-RL experiment it reports "two rounds of iteration" (DSM §5.2.1), and Figure 6 plots Iteration-0 to Iteration-2 (figure legend) |
| $\pi_{\theta_{init}}$ | model | the starting weights | input data | "initial policy model 𝜋𝜃init" (Algorithm 1, Input); "We conduct RL based on DeepSeekMath-Instruct 7B." (DSM §4.2) |

**Not defined in the paper:** I looked in R1 v1 §2.2.1–2.2.2 and in DSM v3
§4.1.1–4.2, Algorithm 1, Figure 4, Table 10 and Appendix A.1.5–A.1.6, and
searched the whole PDF text of both papers for the bar, the double bar and
values.

- $\mid$ in $\pi(o\mid q)$ and $\pi(o_{i,t}\mid q,o_{i,<t})$: no sentence in
  either paper defines it (searched "given" and "condition"; DSM §5.2.3 only
  speaks of "the conditional probability of a certain token"). I read it by
  convention as conditioning: "given".
- $\Vert$ in $\mathbb D_{KL}[\pi_\theta\Vert\pi_{ref}]$ (DSM (3)–(4); R1
  (1)–(2) use round brackets): neither paper defines it. Both use it only
  inside $\mathbb D_{KL}$, and DSM's words for the term, "the KL divergence
  between the trained policy and the reference policy" (§4.1.1), name no
  order. I read it by convention as the separator between the two arguments
  of a KL divergence, trained model first. The right side of DSM (4) and R1
  (2) fixes where each model goes in the estimate, and its average under
  $\pi_\theta$ is the KL with $\pi_\theta$ first (the mismatch under "Where
  the samples come from"), which agrees with this reading.
- $\mathrm{clip}(x,a,b)$: neither paper defines it. DSM credits $\epsilon$ to
  PPO (Schulman et al., 2017), which I did not open. Below I read it as
  clamping $x$ into $[a,b]$, which is what `np.clip` does: "Given an
  interval, values outside the interval are clipped to the interval edges."
  ([NumPy 2.5, `numpy.clip`](https://numpy.org/doc/2.5/reference/generated/numpy.clip.html)).
  That reading is mine.
- $\vert o_i\vert$ (and PPO's $\vert o\vert$): not defined in words. Since
  $t$ runs from 1 to $\vert o_i\vert$ and Algorithm 1 calls $o_{i,t}$ "the
  𝑡 -th token of 𝑜𝑖", I read it as the number of tokens in $o_i$.
- $o_{i,<t}$, $o_{<t}$ and $o_{\le t}$ (PPO (2)): not defined. I read them as
  the tokens before position $t$, and up to and including $t$.
- Capital $Q$ and $O$ in $P(Q)$ and $\pi(O\vert q)$: only lowercase $q$ and
  $o$ get a defining sentence.
- $\pi_\theta(o_i\vert q)$ in R1 (1)–(2): R1 v1 does not say how the
  probability of a whole answer is computed. The check below takes it as the
  product of the token probabilities, which is my reading.
- std: neither paper says whether it divides by the number of values or by
  one less ($G$ or $G-1$ under outcome supervision), nor what happens when it
  is 0.
- $\pi_{ref}$ in R1 v1: it appears only inside equations (1) and (2), so R1 v1
  does not say which model it is or whether it changes during training.
- Values: $\epsilon$ (both papers), $\beta$ and $G$ in R1 v1, and $I$ and $M$
  (for what DSM reports about $I$, see the table). Neither paper prints a number
  for $\mu$; for what DSM says instead, see "What the code must decide".

## Read from the inside out

DSM (3) with (4), for one question. In code, each "per token" quantity is a
`[G, T]` array plus a mask for padding.

1. **Token probabilities.** $\pi_\theta(o_{i,t}\mid q,o_{i,<t})$ is the
   probability that the model gives the token actually written at position
   $t$ of answer $i$, after seeing the question and the earlier tokens of that
   answer. Here $\mid$ means "given" (my reading, see the list above). In
   code: the model's scores before the softmax (the _logits_), shape
   `[G, T, V]`, where $V$ is the vocabulary size → log-softmax over the
   vocabulary axis → at each position, the entry of the token actually
   written → `[G, T]`. Do this three times: with $\pi_\theta$ (gradient
   flows), with $\pi_{\theta_{old}}$ and with $\pi_{ref}$ (both frozen). The subscripts _old_ and _ref_ are labels,
   not powers or indices.
2. **Ratio.** $\pi_\theta/\pi_{\theta_{old}}$ per token, `[G, T]`. It is 1
   when the weights have not moved since sampling, and it measures how much more
   (above 1) or less (below 1) likely that token has become. From the
   log-probabilities of step 1 it is $\exp(\log\pi_\theta-\log\pi_{\theta_{old}})$.
3. **Advantage.** Under outcome supervision (DSM §4.1.2), the $G$ rewards →
   mean and std over the group axis (the $G$ answers to one question, not over the
   batch) → $\tilde r_i$, shape `[G]` → copied to every token of answer $i$:
   $\hat A_{i,t}=\tilde r_i$, `[G, T]`.
   The braces in $\{r_1,\ldots,r_G\}$ list the $G$ values. The mean and std
   run over all $G$ entries, so a value that occurs $k$ times counts $k$ times;
   a Python `set` would keep it once (check 1). Under process supervision (DSM
   §4.1.3), mean and std run over all $K_1+\cdots+K_G$ step rewards in
   $\mathbf R$, giving
   $\tilde r_i^{index(j)}=(r_i^{index(j)}-\mathrm{mean}(\mathbf R))/\mathrm{std}(\mathbf R)$,
   and each token gets the sum over the steps of its answer that end at or
   after it, $\hat A_{i,t}=\sum_{index(j)\ge t}\tilde r_i^{index(j)}$, so
   $\hat A_{i,t}$ changes with $t$ (check 1b).
4. **Clip and min.** Elementwise on `[G, T]`. The unclipped term is
   ratio·$\hat A$. The clipped term clamps the ratio into
   $[1-\epsilon,1+\epsilon]$ first. The $\min$ takes the smaller of the two.
   With $\hat A>0$, the smaller product has the smaller ratio, so the clipped
   term wins once the ratio is above $1+\epsilon$, where clamping lowers it.
   With $\hat A<0$, the smaller product has the larger ratio, so the clipped
   term wins once the ratio is below $1-\epsilon$, where clamping raises it.
   When the clipped term wins, it is a constant, so that token contributes no
   gradient through this term (my derivation). In the other direction (ratio
   below $1-\epsilon$ for $\hat A>0$, above $1+\epsilon$ for $\hat A<0$) the
   unclipped term wins, so a move the advantage disfavours is never capped.
   Inside the range the two terms are equal. Check 2 covers all of these cases.
5. **KL estimate (4).** First, the idea: a _KL divergence_ is a number that
   measures how different two probability distributions are, and it is 0 when
   they are equal. Here the distributions are what $\pi_\theta$ and
   $\pi_{ref}$ predict for the next token. Equation (4) does not sum over the
   vocabulary. It uses only the sampled token: with
   $x=\pi_{ref}/\pi_\theta$ for that token, the estimate is
   $x-\log x-1$, per token `[G, T]`. It is never negative (my derivation):
   $f(x)=x-\log x-1$ has derivative $1-1/x$, which is negative for $x<1$ and
   positive for $x>1$, so $f$ falls until $x=1$, rises after it, and its
   smallest value is $f(1)=0$. So the estimate is 0 exactly when $\pi_{ref}$
   and $\pi_\theta$ give the sampled token the same probability, and positive
   otherwise; check 4 illustrates this on a grid. The $=$ here defines what
   the symbol $\mathbb D_{KL}$ means in this paper: this estimate, not the exact KL. The
   double bar $\Vert$ separates the two models, and the order matters (my
   reading, see the list above). $\log$
   is the natural log.
6. **Per-token term.** $\min[\ldots]-\beta\,\mathbb D_{KL}$, `[G, T]`. The
   curly braces in (3) group this expression; they do not build a set.
7. **Average over each answer's tokens.**
   $\frac1{\vert o_i\vert}\sum_{t=1}^{\vert o_i\vert}$ runs over $t$ along the
   time axis and stops at each answer's own length. Result `[G]`. Each answer
   gets equal weight, so a token of a short answer weighs more than a token of
   a long one (check 2: $1/6$ against $1/4$).
8. **Average over the group.** $\frac1G\sum_{i=1}^G$ runs over the answer
   axis. Result: one number per question.
9. **Expectation.** $\mathbb E[\ldots]$ means the average over random draws.
   The bracket says what is drawn: a question from the dataset, then $G$
   answers written by $\pi_{\theta_{old}}$. Code estimates it with the mean
   over the questions in a batch. Result: a scalar, which training maximizes,
   so the loss for an optimizer that minimizes is $-\mathcal J$.

**R1 (1)–(3)** has the same steps without $t$. The ratio, the clip and the KL
estimate (2) are applied once per answer, to whole-answer probabilities. The
advantage $A_i$ (3) is per answer, and there is no $1/\vert o_i\vert$
average. If a whole-answer probability is the product of its token
probabilities (my reading, see the list above), then the whole-answer ratio is
the product of the token ratios. A product can stay inside the clip range while
single tokens are far outside it (check 3).

**DSM PPO (1)–(2)** has the same clip-and-min, but over one answer $o$
per question. Its $A_t$ comes from GAE (see the table), which DSM cites and
does not spell out. Its KL penalty, $\beta\log(\pi_\theta/\pi_{ref})$, is
subtracted in (2) from the reward model's score $r_\phi(q,o_{\le t})$, giving
$r_t$, before the advantage is computed. Check 7 shows that this per-token
log-ratio can be negative, so it can raise the reward, while the estimate (4)
never is (step 5).

**Kind of statement:** DSM (1), DSM (3) and R1 (1) are **objectives** to
maximize, averaged over the questions and answers of step 9 (one answer per
question in DSM (1)). DSM (2) and R1 (3) are **definitions**: they say what
$r_t$ and $A_i$ stand for. DSM (4) is an **estimator**: a value computed from
samples that stands in for a quantity not computed exactly. DSM introduces it
as one (the $\mathbb D_{KL}$ row): it estimates the KL between $\pi_\theta$
and $\pi_{ref}$ from the one sampled token instead of summing over the
vocabulary (step 5). R1 v1 prints (2) with no sentence about it; it has the
same form applied to the sampled answer, so I read it as the same kind of
one-sample estimate, for whole answers. Both also fix what $\mathbb D_{KL}$
denotes in these papers (step 5). $\mathbb D_{KL}$ and $A_i$ substitute back
into the objective; $r_t$ feeds the GAE that gives PPO's $A_t$. Algorithm 1 is the **update
rule**: it says which copy changes at each line and which stays fixed (next
section).

## Where the samples come from

- **Questions,** $q\sim P(Q)$ and $\mathcal D_b\sim\mathcal D$ (Algorithm 1,
  line 5): a fixed dataset prepared before training. For DeepSeekMath-RL it is
  "chain-of-thought-format questions related to GSM8K and MATH from the SFT
  data, which consists of around 144K questions" (DSM §4.2; the PDF breaks
  "chain-of-thought" across a line). R1 v1 §2.2.1 does
  not describe its question set; §2.2.2 names math problems and LeetCode
  problems as examples of what its rules check.
- **Answers,** $\{o_i\}_{i=1}^G\sim\pi_{\theta_{old}}(O\mid q)$: generated
  during training by $\pi_{\theta_{old}}$, a copy of $\pi_\theta$ taken at
  line 6 just before sampling (line 7). For DSM, "The max length is set to
  1024" (§4.2). Table 10 writes GRPO's data source as
  $\{o_i\}\sim\pi_\theta(O\vert q)$, "the real-time policy model during the
  online training process". This agrees with line 6, because the copy is
  refreshed at every step.
- **Rewards and advantages** (lines 8–9) are computed once per step, before the
  $\mu$ updates of lines 10–11. The reward model (line 12) and $\pi_{ref}$
  (line 3) change only once per outer iteration. DSM §4.1.4 adds that the retraining "incorporates 10% of
  historical data". R1 v1 is silent on when its $\pi_{ref}$ changes (see
  "Not defined in the paper"). A later version,
  [DeepSeek-R1 v2](https://arxiv.org/abs/2501.12948v2), which this reading
  does not pin, gives the interval in §2.1 for DeepSeek-R1-Zero, and again in
  §3.2 for DeepSeek-R1's first RL stage: "Every 400 steps, we replace the
  reference model with the latest policy model."
- **What carries no gradient:** neither paper writes a stop-gradient
  $\mathrm{sg}(\cdot)$. In its GRPO half, Figure 4 draws the reference and reward models as
  "Frozen Models" and only the policy as trained, and $\pi_{\theta_{old}}$ is a
  copy. So $\pi_{\theta_{old}}$, $\pi_{ref}$, $r_i$ and $\hat A_{i,t}$ are
  constants, and $\theta$ enters only through $\pi_\theta$ in the ratio and
  in (4). This is my reading of which
  terms depend on $\theta$.
- **A mismatch, my derivation:** at one position, write
  $x(v)=\pi_{ref}(v)/\pi_\theta(v)$ for each vocabulary token $v$. Averaged
  over $v$ drawn from $\pi_\theta$, $x-1$ gives
  $\sum_v\pi_\theta(v)\frac{\pi_{ref}(v)}{\pi_\theta(v)}-1=\sum_v\pi_{ref}(v)-1=0$,
  and $-\log x$ gives $\sum_v\pi_\theta(v)\log\frac{\pi_\theta(v)}{\pi_{ref}(v)}$,
  which is the standard definition of the exact KL
  $\mathrm{KL}(\pi_\theta\Vert\pi_{ref})$ at that position. So the estimate
  (4) averages to the exact KL when the tokens are drawn from $\pi_\theta$.
  They are drawn from $\pi_{\theta_{old}}$. The two copies are equal on the
  first of the $\mu$ updates and drift apart after it. Then each token is
  weighted by $\pi_{\theta_{old}}(v)$ instead of $\pi_\theta(v)$, neither sum
  above simplifies, and the average need not equal the KL (check 5: 0.16667
  against the KL 0.17329).

## Check with small numbers

The inputs: one question, $G=2$ answers with rewards $(1,0)$, so the
outcome-supervision advantages are $\pm1$ and both signs occur. Answer 1 has 3 tokens and answer 2
has 2, so the $1/\vert o_i\vert$ weights differ. The token ratios 1.5, 1.1 and
0.5 put the ratio above, inside and below $[0.8,1.2]$, so each argument of the
min wins at least once for each sign of the advantage. $\pi_{ref}/\pi_\theta$
takes the values 1, 2 and ½, so the KL term is 0 at the two tokens where the
ratio is 1 and positive at the other three.
The same numbers then go through R1's whole-answer form. The script also checks
the std divisor, repeated rewards, a group with zero spread, the
process-supervision advantage (one answer with two steps, one with one), the "unbiased"
claim, equation (21), and PPO's penalty (2) on answer 2's tokens.
$\epsilon=0.2$ is a test value (see "Not defined in the paper"); $\beta=0.04$
is DSM's value.

```python
import numpy as np

eps = 0.2    # test value; see "Not defined in the paper"
beta = 0.04  # DeepSeekMath §4.2: "The KL coefficient is 0.04"


def kl_est(p_ref, p_th):  # eq (4) per token; R1 eq (2) per output
    x = p_ref / p_th
    return x - np.log(x) - 1


def surrogate(ratio, A):  # the min[...] term of eq (3) / R1 eq (1)
    unclipped = ratio * A
    clipped = np.clip(ratio, 1 - eps, 1 + eps) * A
    return np.minimum(unclipped, clipped), unclipped, clipped


# 1. Advantages (R1 eq (3); DeepSeekMath §4.1.2)
r = np.array([1.0, 0.0])
A = (r - r.mean()) / r.std()                 # default ddof=0: divides by n
A_n1 = (r - r.mean()) / r.std(ddof=1)        # ddof=1: divides by n-1
assert np.allclose(A, [1, -1]) and np.allclose(A_n1, [2**-0.5, -(2**-0.5)])
r64 = np.arange(64.0)                        # G = 64 (DSM §4.2); any rewards with spread
grow = r64.std(ddof=1) / r64.std()           # every advantage is divided by this
assert np.isclose(grow, (64 / 63) ** 0.5) and np.isclose(A[0] / A_n1[0], 2**0.5)
r4 = np.array([1.0, 0, 0, 0])                # a value that occurs k times counts k times
assert r4.mean() == 0.25 and np.mean(list(set(r4))) == 0.5
assert np.allclose((r4 - r4.mean()) / r4.std(), [3**0.5] + [-(3**-0.5)] * 3)
with np.errstate(invalid="ignore"):
    same = np.array([1.0, 1.0])               # every answer gets the same reward
    assert np.isnan((same - same.mean()) / same.std()).all()
print("A (std over n):", A, " A (std over n-1):", A_n1.round(4),
      f" std(n-1)/std(n) at G=64: {grow:.5f}")

# 1b. Process supervision (DeepSeekMath §4.1.3). Output 1 has 3 tokens and two steps
#     ending at tokens 2 and 3; output 2 has 2 tokens and one step ending at token 2.
ends = [[2, 3], [2]]                         # index(j) for each step
R = [[1.0, 0.0], [0.0]]                      # step rewards
flat = np.concatenate(R)                     # mean and std over all K_1 + K_2 = 3 step rewards
R_tilde = [(np.array(Ri) - flat.mean()) / flat.std() for Ri in R]
A_ps = [np.array([sum(rt for e, rt in zip(ends[i], R_tilde[i]) if e >= t)
                  for t in range(1, n + 1)]) for i, n in enumerate([3, 2])]
assert np.allclose(A_ps[0], [2**-0.5, 2**-0.5, -(2**-0.5)])  # changes with t
assert np.allclose(A_ps[1], [-(2**-0.5)] * 2)
print("process supervision: A_1,t =", A_ps[0].round(5), " A_2,t =", A_ps[1].round(5))

# 2. Per-token objective, DeepSeekMath eq (3)-(4). Output 1 has 3 tokens and A=+1,
#    output 2 has 2 tokens and A=-1. Per-token probabilities of the sampled token:
p_old = [np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5])]
p_th = [np.array([0.75, 0.55, 0.25]), np.array([0.75, 0.25])]
p_ref = [np.array([0.75, 0.55, 0.5]), np.array([0.375, 0.5])]
per_output = []
for i in range(2):
    ratio = p_th[i] / p_old[i]
    s, unc, cl = surrogate(ratio, A[i])
    kl = kl_est(p_ref[i], p_th[i])
    per_output.append(np.mean(s - beta * kl))  # (1/|o_i|) sum over t
    print(f"o_{i+1}: ratio={ratio} min-term={s.round(4)} KL={kl.round(5)} "
          f"weight per token={1 / (2 * len(ratio)):.4f}")
    if i == 0:
        assert np.allclose(s, [1.2, 1.1, 0.5])        # clipped wins, equal, unclipped wins
        assert s[0] == cl[0] < unc[0] and s[2] == unc[2] < cl[2]
        assert np.allclose(kl, [0, 0, 1 - np.log(2)])
    else:
        assert np.allclose(s, [-1.5, -0.8])           # unclipped wins, clipped wins
        assert s[0] == unc[0] < cl[0] and s[1] == cl[1] < unc[1]
        assert np.allclose(kl.sum(), 0.5)
J_tok = np.mean(per_output)                            # (1/G) sum over i
print("per-output terms:", np.round(per_output, 5), " J (eq 3) =", round(J_tok, 5))
assert np.isclose(J_tok, (2.8 / 3 - beta * (1 - np.log(2)) / 3 + (-1.15 - beta * 0.25)) / 2)

# 3. R1 eq (1)-(2) read per output, with pi(o_i|q) = product of token
#    probabilities (my reading; R1 v1 does not say how pi(o_i|q) is computed).
J_seq_terms = []
for i in range(2):
    seq_ratio = np.prod(p_th[i]) / np.prod(p_old[i])
    s, _, _ = surrogate(seq_ratio, A[i])
    kl = kl_est(np.prod(p_ref[i]), np.prod(p_th[i]))
    J_seq_terms.append(s - beta * kl)
    print(f"o_{i+1}: sequence ratio={seq_ratio:.4f} min-term={s:.4f} KL={kl:.5f}")
J_seq = np.mean(J_seq_terms)
assert np.isclose(J_seq_terms[0], 0.825 - beta * (1 - np.log(2)))  # inside the clip range
assert np.isclose(J_seq_terms[1], -0.8)                           # clipped, KL estimate 0
print("J (R1 eq 1, sequence form) =", round(J_seq, 5))

# 4. Eq (4) >= 0 on a grid, and 0 at pi_ref = pi_theta (an illustration; the reason is in step 5).
x = np.linspace(0.01, 10, 10_000)
assert (x - np.log(x) - 1 >= 0).all() and kl_est(0.3, 0.3) == 0

# 5. "Unbiased": exact expectation over a 3-token vocabulary.
pt = np.array([0.5, 0.25, 0.25])     # pi_theta
pr = np.array([0.25, 0.5, 0.25])     # pi_ref
po = np.array([1, 1, 1]) / 3         # pi_theta_old, a different copy
true_kl = np.sum(pt * np.log(pt / pr))
e_theta = np.sum(pt * kl_est(pr, pt))  # samples drawn from pi_theta
e_old = np.sum(po * kl_est(pr, pt))    # samples drawn from pi_theta_old
assert np.isclose(e_theta, true_kl) and not np.isclose(e_old, true_kl)
print(f"KL={true_kl:.5f}  E_theta[est]={e_theta:.5f}  E_old[est]={e_old:.5f}")

# 6. Eq (21): d/d(log pi_theta) of one token's term at pi_theta = pi_old
#    equals A + beta*(pi_ref/pi_theta - 1). The ratio is 1 there; its derivative is not 0.
def token_obj(logp, p_old_t, p_ref_t, A_t):
    s, _, _ = surrogate(np.exp(logp) / p_old_t, A_t)
    return s - beta * kl_est(p_ref_t, np.exp(logp))

h = 1e-6
for A_t in (1.0, -1.0):
    l0 = np.log(0.5)
    num = (token_obj(l0 + h, 0.5, 0.25, A_t) - token_obj(l0 - h, 0.5, 0.25, A_t)) / (2 * h)
    gc = A_t + beta * (0.25 / 0.5 - 1)
    assert np.isclose(num, gc, atol=1e-6)
    print(f"A={A_t:+.0f}: finite-difference d/d(log pi)={num:.6f}  eq (21)={gc:.6f}")

# 7. PPO's per-token penalty, DeepSeekMath eq (2), against beta * eq (4), on answer 2's
#    tokens. Usually only the last token gets a reward-model score (DSM §4.1.1); 1 is a test value.
r_phi = np.array([0.0, 1.0])
sub2 = beta * np.log(p_th[1] / p_ref[1])   # what eq (2) subtracts from the reward
r_t = r_phi - sub2
sub4 = beta * kl_est(p_ref[1], p_th[1])    # what eq (3) subtracts from the objective
assert np.allclose(sub2, [beta * np.log(2), -beta * np.log(2)]) and sub2.sum() == 0
assert np.allclose(r_t, [-beta * np.log(2), 1 + beta * np.log(2)])
assert (sub4 > 0).all() and np.isclose(sub4.sum(), beta * 0.5)
print(f"eq (2): subtracted per token={sub2.round(5)} sum={sub2.sum():.5f} r_t={r_t.round(5)}")
print(f"eq (4): beta*estimate per token={sub4.round(5)} sum={sub4.sum():.5f}")
print("all checks passed")
```

Run with `uv run --no-project --with numpy python check.py`.

Output:

```text
A (std over n): [ 1. -1.]  A (std over n-1): [ 0.7071 -0.7071]  std(n-1)/std(n) at G=64: 1.00791
process supervision: A_1,t = [ 0.70711  0.70711 -0.70711]  A_2,t = [-0.70711 -0.70711]
o_1: ratio=[1.5 1.1 0.5] min-term=[1.2 1.1 0.5] KL=[0.      0.      0.30685] weight per token=0.1667
o_2: ratio=[1.5 0.5] min-term=[-1.5 -0.8] KL=[0.19315 0.30685] weight per token=0.2500
per-output terms: [ 0.92924 -1.16   ]  J (eq 3) = -0.11538
o_1: sequence ratio=0.8250 min-term=0.8250 KL=0.30685
o_2: sequence ratio=0.7500 min-term=-0.8000 KL=0.00000
J (R1 eq 1, sequence form) = 0.00636
KL=0.17329  E_theta[est]=0.17329  E_old[est]=0.16667
A=+1: finite-difference d/d(log pi)=0.980000  eq (21)=0.980000
A=-1: finite-difference d/d(log pi)=-1.020000  eq (21)=-1.020000
eq (2): subtracted per token=[ 0.02773 -0.02773] sum=0.00000 r_t=[-0.02773  1.02773]
eq (4): beta*estimate per token=[0.00773 0.01227] sum=0.02000
all checks passed
```

The per-token form (DSM (3)) gives $-0.11538$. The whole-answer form (R1 (1))
gives $0.00636$ on the same tokens. In answer 1, two tokens are outside the clip
range, but the product of the ratios (0.825) is inside it. In answer 2 the
per-token KL estimates add up to 0.5, while the whole-answer estimate is 0. On
answer 2's tokens, PPO's (2) subtracts $\pm0.02773$, a penalty on one token
and a bonus on the other, summing to 0; GRPO's $\beta\times$(4) subtracts
0.00773 and 0.01227, summing to 0.02.

## What the code must decide

The repository that DSM links holds no GRPO training code (see **Source**), so
each choice below is **open**.

- **Std divisor ($G$ or $G-1$):** open (see "Not defined in the paper"). The
  library defaults differ. For `np.std`, "The divisor used in calculations is
  N - ddof, where N represents the number of elements. By default ddof is
  zero."
  ([NumPy 2.5, `numpy.std`](https://numpy.org/doc/2.5/reference/generated/numpy.std.html)),
  so it divides by $G$. `torch.std` divides by $\max(0,N-\delta N)$, with
  $\delta N$ its `correction` argument, which "Defaults to Bessel’s
  correction, correction=1."
  ([PyTorch 2.15, `torch.std`](https://docs.pytorch.org/docs/2.15/generated/torch.std.html)),
  so it divides by $G-1$. Check 1 shows that the choice changes $A$ from
  $\pm1$ to $\pm0.7071$. In general the $G-1$ divisor makes the std
  $\sqrt{G/(G-1)}$ times larger, so every advantage shrinks by that factor
  (my derivation): 1.00791 at DSM's $G=64$ (check 1). Under process
  supervision the same holds with $G$ replaced by the number of step rewards
  in $\mathbf R$.
- **A group whose rewards are all equal:** open. Std is 0 and the advantage is
  0/0 (check 1). A small number added to the denominator makes every advantage
  0, so the group contributes only through the KL term (my derivation). A
  later paper, [DAPO](https://arxiv.org/abs/2503.14476v2) (§3.2, equation
  (11)), removes such groups instead: it proposes to "over-sample and filter
  out prompts with the accuracy equal to 1 and 0", so every question kept has
  both right and wrong answers.
- **Whole-answer probability in R1:** how $\pi_\theta(o_i\mid q)$ in (1)–(2)
  is computed from the token probabilities is open in R1 v1 (see "Not defined
  in the paper" and check 3).
- **Averaging:** (3) fixes the average within an answer ($1/\vert o_i\vert$)
  and over the group ($1/G$). DAPO (§3.3, equation (12)) replaces both with
  one average over all the group's tokens, $1/\sum_{i=1}^G\vert o_i\vert$, so
  that "longer sequences can have more influence on the overall gradient
  update compared to shorter sequences". How questions in a batch are averaged
  is open.
  "the training batch size is 1024" (DSM §4.2) does not say whether it counts
  questions or answers.
- **Padding and truncation:** the sum stops at $\vert o_i\vert$, so padded
  positions must be masked out of both sums. What reward an answer cut off at
  the 1024-token maximum receives is open.
- **What is detached:** $\pi_{\theta_{old}}$, $\pi_{ref}$, $r_i$ and
  $\hat A_{i,t}$ (see "Where the samples come from").
- **Ratio in log space:** computing `exp(logp - logp_old)` rather than
  dividing two probabilities is open, since the paper writes only the ratio.
- **$\mu$:** Algorithm 1 prints no number. DSM §4.2 says "The policy model
  only has a single update following each exploration stage." DSM itself
  reads these words as making the two copies equal: "To simplify the analysis,
  it is assumed that the model only has a single update following each
  exploration stage, thereby ensuring that 𝜋𝜃𝑜𝑙𝑑 = 𝜋𝜃 . In this case, we can
  remove the min and clip operation" (A.1.5, written for PPO). The copies are
  equal when the gradient is taken only if each sampled batch gets one
  gradient step, that is $\mu=1$ (my inference). Then the ratio is 1, the clip
  never acts, and the gradient is (20), whose per-token coefficient is (21)
  (see "The equation").
- **Values of $\epsilon$, $I$, $M$, and $\beta$ and $G$ in R1:** open (see "Not
  defined in the paper").
- **How R1 combines its rewards:** §2.2.2 names accuracy and format rewards
  but not how they add up to $r_i$. That is open in R1 v1 §2.2.

## What the paper claims and what it shows

- **"GRPO foregoes the critic model, instead estimating the baseline from group
  scores, significantly reducing training resources"** (DSM §1 and §1.1; the
  Figure 4 caption says the same with "value model" in place of "critic
  model"), and R1's "In order to save the training
  costs of RL" (§2.2.1): argued from the design. The value model "is typically
  another model of comparable size as the policy model" (DSM §4.1.1). I found
  no measured memory or compute comparison in DSM v3 (searched "memory" and
  "training resources"). The conclusion's "with less memory consumption" is
  stated, not measured.
- **Why a group baseline:** argued, not tested on its own. DSM §4.1.1 says that
  "usually only the last token is assigned a reward score by the reward model,
  which may complicate the training of a value function that is accurate at
  each token", and that the group method "aligns well with the comparative
  nature of rewards models".
- **(4) is "unbiased" (Schulman, 2020):** cited, not derived in the paper.
  For when it holds and when it does not, see the mismatch under "Where the
  samples come from" (my derivation, with check 5).
- **(4) "is guaranteed to be positive":** step 5 of the inside-out reading
  shows why it is never negative and when it is 0, so the exact statement is
  "non-negative".
- **Gradient (20)–(21):** derived under the assumption quoted in "The
  equation", where my derivation of (21) is. As printed, (20) and (21) drop
  $q$ from the conditioning in the ratio $\pi_{ref}/\pi_\theta$.
- **GRPO improves DeepSeekMath-Instruct 7B:** measured, Table 5,
  chain-of-thought: GSM8K 82.9% → 88.2%, MATH 46.8% → 51.7%. These are single
  numbers. I found no run count or interval in the text.
- **GRPO beats Online RFT, and process supervision (GRPO+PS) beats outcome
  supervision (GRPO+OS):** measured on DeepSeekMath-Instruct 1.3B (Figure 5),
  as training curves; in both panels the curves cross at some steps. I found
  no run count or interval in the text. Online
  RFT fine-tunes on the policy's own samples and "does not penalize incorrect
  responses and uniformly reinforces all responses with correct answers at the
  same level of intensity" (DSM §5.2.1). "Outcome supervision only provides a
  reward at the end of each output"; process supervision "provides a reward at
  the end of each reasoning step" (DSM §4.1.3).
- **Why RL helps:** measured Maj@K against Pass@K (Figure 7), then
  interpreted with a hedge. DSM v3 defines neither metric (I searched the PDF
  text for "Pass@", "Maj@" and "majority"; the nearest is Table 5's "majority
  votes with 32 candidates"). I read Maj@K as the accuracy of the majority
  answer among $K$ samples, and Pass@K as the share of questions where at
  least one of $K$ samples is correct; both readings are mine. The hedge: "it
  seems that the improvement is attributed to boosting the correct response
  from TopK rather than the enhancement of fundamental capabilities" (DSM
  §5.2.2). DSM does not define TopK either (searched "TopK" and "Top-K"); I
  read it as the $K$ sampled answers behind Maj@K and Pass@K, which Figure 7's
  axis calls "K: The number of candidates" (my reading).

## Sources

- DeepSeek-AI, DeepSeek-R1, <https://arxiv.org/abs/2501.12948v1>: TeX source
  (`main.tex`) and PDF; §2.2.1–2.2.2 read, PDF text searched for "reference",
  "KL", "penalty", hyperparameters and code links.
- Z. Shao et al., DeepSeekMath, <https://arxiv.org/abs/2402.03300v3>: TeX
  source (`main.tex`) and PDF; §1, §4.1–4.2, §5.2, Table 10, Appendix A.1.5–A.1.6
  read; the rendered Figure 4 and Algorithm 1 (§4.1.1) and Figure 5 viewed.
- DeepSeek-AI, DeepSeek-R1 v2, <https://arxiv.org/abs/2501.12948v2>, a later
  version than the one pinned: TeX source and PDF text searched for "reference
  model"; §2.1 and §3.2 read for how often the reference model is replaced.
- `github.com/deepseek-ai/DeepSeek-Math`, the repository DSM links: the
  commit list, the root directory, `replicate/`, and `evaluation/` at commit
  `b8b0f8c` (15 Apr 2024). None of them contains RL training code.
- Q. Yu et al., DAPO: An Open-Source LLM Reinforcement Learning System at
  Scale, <https://arxiv.org/abs/2503.14476v2>: TeX source and PDF; §3.2–3.3
  read, with equations (11)–(12).
- Library documentation, for the defaults named in "Not defined in the paper"
  and "What the code must decide":
  [NumPy 2.5 `numpy.std`](https://numpy.org/doc/2.5/reference/generated/numpy.std.html)
  and [`numpy.clip`](https://numpy.org/doc/2.5/reference/generated/numpy.clip.html)
  (2.5 is the minor version of NumPy 2.5.4, which the check ran with), and
  [PyTorch 2.15 `torch.std`](https://docs.pytorch.org/docs/2.15/generated/torch.std.html)
  (nothing in scope pins PyTorch and the check does not use it, so I took the
  release that the documentation's "stable" link opens).
- Named but not opened: Schulman et al., 2017 (PPO); Schulman et al., 2015
  (GAE); Schulman, 2020 (KL approximation); Ouyang et al., 2022.
