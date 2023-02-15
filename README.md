## Neural Representations Reveal Distinct Modes of Class Fitting in Residual Convolutional Networks

This repository contains the code for the paper *Neural Representations Reveal Distinct Modes of Class Fitting in Residual Convolutional Networks*, accepted at the AAAI-23 conference [1]. We use Bayesian nonparametric density models [2] to characterise distributions of neural representations in classes. Surprisingly, we find that classes in residual convolutional networks are not fitted in an uniform way. Furthermore, the differences correlate with memorization of input examples and adversarial robustness.

The extended version of the paper, including technical appendix and initial results for non-convolutional architectures with skip-connections (namely,  Vision Transformers and MLP Mixers) is available at: https://arxiv.org/abs/2212.00771

See [Instruction](INSTRUCTION.md) for the detailed instructions on running the experiments.

### References

1. Jamroż, M., & Kurdziel, M. 2023. Neural Representations Reveal Distinct Modes of Class Fitting in Residual Convolutional Networks. *The 37th AAAI Conference on Artificial Intelligence, February 7-14, Washington, DC, USA.*

2. Jamroż, M., Kurdziel, M., & Opala, M. 2020. A Bayesian Nonparametrics View into Deep Representations. In *Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems, December 6-12, virtual,* 1440-1450. Source code available at: https://github.com/mjamroz90/bayesian-view-deep-representations
