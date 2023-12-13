# Task 7.5: Intermediate Step to the Investigation on EEGNet's Generalizability

## Notes

These results and images whose names ended with `(old)` are for the (EEGNet + simple counting) which we have implemented in Task 7. But knowing that the spelling accuracy is low, we need to find ways to improve it before testing the EEGNet models on unseen users (which will be done in Task 8). One way is to adapt the output to some scores so that we can leverage Bayesian Inference. New results show that this can bring a nearly $10\%$ increment, which is nice but not enough as we want the performance to be competitive with SWLDA.