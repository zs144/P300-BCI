# Task 7.5: Intermediate Step to Investigate EEGNet's Generalizability

## Notes

These results and images whose names ended with `(old)` are for the (EEGNet + simple counting) which we have implemented in Task 7. But knowing that the spelling accuracy is low, we need to find ways to improve it before testing the EEGNet models on unseen users (which will be done in Task 8). One way is to adapt the output to some scores so that we can leverage Bayesian Inference. New results show that this can bring a nearly $10\%$ increment, which is nice but not enough as we want the performance to be competitive with SWLDA.

## Overview

In task 7 (`task7,ipynb`), what we compared is EEGNet + simple counting VS SWLDA + Bayesian inference (enhanced by a 2-gram LM) as the benchmark. The result shows that the SWLDA-based method leads by almost $20\%$ (see `spelling_acc(old).png`). This is somehow unfair for the EEG-based method since simple counting is proved to be less efficient than Bayesian inference. Therefore, in task 7.5, we also tried the other two combinations: SWLDA + simple counting, and EEGNet + Bayesian inference. Specifically, the former is implemented and compared with the benchmark in `check.ipynb`; the comparison between the latter and the benchmark is conduct in `task7.5.ipynb`.

## Results


## TODOs:
- [ ] SWLDA + simple counting ends up with a significantly low spelling accuracy, which warrants deeper investigations, such as plotting the change of scores along trials to see how some nonsensical predictions are made.
- [ ] So far, we haven't tested the generalizability of EEGNet on unseen users. Since we already have two EEGNets trained on `EDFData-StudyA` and `EDFData-StudyD` respectively, we are ready to conduct the ultimate experiment by cross-validate the spelling accuracy on different datasets. However, before doing this, we need to **double-check the code** and **figure out why the EEGNet-based method still performs worse than the benchmark** ($10\%$ gap) given the controversial fact that the two achieves the similar signal-classification accuracy (in fact, EEGNet is even higher on this). **Perhaps, we also need to dive into the scores-VS-trails plot.** But we should be reminded that EEGNet originally outputs a 2-d vector $[a, b]$ with each element measuring the relative likelihood of being the target/non-target. To convert this information to a classifier score (scalar), we use $\frac{e^a}{e^a + e^b}$, inspired by the softmax function. **The goal here is to make sure the distribution of scores between different classes is as separated as possible. If the method we used here is not the best to fit this goal, then it may affect the accuracy of Bayesian inference later.**
- [ ] After finishing the retrospection and possibly some fixing, we will cross-validate EEGNet's spelling accuracy on unseen users in Task 8.