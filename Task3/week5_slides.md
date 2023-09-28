---
theme: gaia
_class: lead
paginate: true
backgroundColor: #fff
marp: true
---
# **Independent Study Weekly Meeting 5**

#### Re-visit Adpative Stimulus Selection Algorithms

Zion Sheng
Department of ECE
Duke University

---
## Table of Content

1. Part 1: Progess Made This Week
2. Part 2: Understanding the Adpative Stimulus Selection Algorithms
3. Part 3: Questions during Implementation
4. Part 4: Deeper Thoughts

---
## Part 1: Progess Made This Week
- Learn to use MNE
- Practice running jobs on DCC
- Reread Adpative Stimulus Selection Algorithms paper (Mainsah et al., 2018) and implement it in Python

---
## Part 2: Adpative Stimulus Selection Algorithms
<style scoped>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}

section {
  font-size: 25px
}
</style>
### The difficulty of the problem
The difficulty of developping an adaptive stimulus selection algorithm comes from the lack of existing objective functions s with tractable solutions (the "curse of dimensionality") to allow for real-time algorithm implementation.

This paper developed a simple, yet powerful analytical solution to an objective function, which allows for computational efficiency in exploring the high dimensional BCI stimulus space. The objective function is parameterized only by the prior probability mass of a future stimulus under consideration, irrespective of its content.

Mutual information is used here to build this objective function.

---
## Part 2: Adpative Stimulus Selection Algorithms
<style scoped>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}

section {
  font-size: 25px
}
</style>
### What does mutual information indicate?
Mutual information can be evaluated within the context of how much the currently observed data (i.e., the previously observed flash groups, $\mathbf{f_t}$, and classifier scores,$y_t$) reduce the uncertainty about the target character estimate.

### The ultimate form of the objective function
$$
I(Y_{t+1}^h;C^*|y_t, \mathbf{f^h_t}) = \int_{-\infty}^{\infty} \mathcal{I}(z_{t+1}^h)dz_{t+1}^h
$$
$$
\mathcal{I}(z_{t+1}^h) = P_{1t}l1(z_{t+1}^h)\log(\frac{l1(z_{t+1}^h)}{l0(z_{t+1}^h)(1-P_{1t}) + l1(z_{t+1}^h)P_{1t}})
+
(1-P_{1t})l0(z_{t+1}^h)\log(\frac{l0(z_{t+1}^h)}{l0(z_{t+1}^h)(1-P_{1t}) + l1(z_{t+1}^h)P_{1t}})
$$
$P_{1t}(f^h_{t+1})$ is the sum of prior probabilities at time $t$ for characters that are flashed in $f^h_{t+1}$, which we will denote as $P_{1t}$ for simplicity. The objective function is now conveniently parameterized by $P_{1t}$.

---
## Part 3: Questions Encountered during Implementation
<style scoped>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}

section {
  font-size: 25px
}
</style>

The flash group which maximizes the objective function will be selected for presentation.

Here comes the problem:
- Initial choice
The probabilities stuck at the beginning. Need to randomly pick flashed groups at the beginning to jump start.

- Keep flash the row
I expect the row and column of the target should be equally flashed, but only row gets flashed in the simulation. As a result, all characters in the row ends up having the equally high score.

- Analytical solution
Do we have analytical solution for accuracy and EST?