---
theme: gaia
_class: lead
paginate: true
backgroundColor: #fff
marp: true
math: mathjax
---
<style scoped>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}

section {
  font-size: 30px
}
</style>
# **Independent Study Weekly Meeting 11**

#### P300 speller with a simple bi-gram language model

Zion Sheng
Department of ECE
Duke University

---
## Topics

1. Topic 1: Why introduce the bi-gram LM?
2. Topic 2: Results visualization
3. Topic 3: Comparison between with and without the bi-gram LM

---
## Topic 1: Why introduce the bi-gram LM?
<style scoped>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}

section {
  font-size: 30px
}
</style>

The basic observation is that, when we type any English word, the probability distribution of letters given the previous typed ones is not uniform. We can utilize this information to initialize the probability of each character in the hope that this will improve the performance.

For example, suppose we already typed `"pr"`, then the next letter is impossible to be `"z"` because no English word contain this combination.

---
## Topic 1: Why introduce the bi-gram LM?
<style scoped>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}

section {
  font-size: 30px
}
</style>
Under the Markov assumption, we assume that the probability of the next letter is only determined by the previous letter. Such two-letter pairs consisting is called bi-grams.
![width:500px center](images/dist.png)

---
## Topic 2: Results visualization
<style scoped>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}

section {
  font-size: 30px
}
</style>

The participants are arranged in the ascending order of their AUCs. Now we can clearly see that the participants need less flashes if their classifier AUC is high.

![width:550px center](images/2.png)