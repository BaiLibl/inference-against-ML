# Inference Against Machine Learning

# Contents
- [Surveys and Overviews](#surveys-and-overviews)
- [Papers and Code](#papers-and-code)
  - [Membership inference](#membership-inference)
  - [Attribute/Property inference](#attributeproperty-inference)
  - [Model inversion](#model-inversion)
  - [Model extraction/stealing](#model-extractionstealing)
- [Open Repository](#open-repository)
- [Related Link](#related-link)

# Surveys and Overview

# Papers and Code

## Membership inference
| Year | Title | Adversarial Knowledge | Target Model | Link |
| - | - | - | - | - |
| 21'CCS | [**When Machine Unlearning Jeopardizes Privacy**](https://arxiv.org/abs/2005.02205) | 1 | 1 | 1 | 
| 17'S&P | [**Membership Inference Attacks Against Machine Learning Models**](https://arxiv.org/abs/1610.05820) | 1 | 1 | [code](https://github.com/csong27/membership-inference) |
| 1 | 1 | 1 | 1 | 1 |

## Attribute/Property Inference

## Model Inversion

## Model Extraction/Stealing
| Year | Title | Infer |Adversarial Knowledge | Target Model | Link |
| - | - | - | - | - | - |
| 16'USENIX | [**Stealing Machine Learning Models via Prediction APIs**](https://arxiv.org/abs/1609.02943) | parameter | confidence/label | LR/DT/SVM/NN | [code](https://github.com/ftramer/Steal-ML) |
| 18'ICLR | [**Towards Reverse-Engineering Black-Box Neural Networks**](https://arxiv.org/abs/1711.01768) | hyperparameter | data distribution <br> NN family | Image classification | [code](https://github.com/coallaoh/WhitenBlackBox) |
| 18'CVPR | [**Knockoff Nets: Stealing Functionality of Black-Box Models**](https://arxiv.org/abs/1812.02766) | functionality | public data <br> confidence  | Image classification | [code](https://github.com/tribhuvanesh/knockoffnets) |
| 18'S&P | [**Stealing Hyperparameters in Machine Learning**](https://arxiv.org/abs/1802.05351) | hyperparameter | training dataset <br> objective function <br> model parameter  | LR/SVM/NN |  |
| 1 | 1 | 1 | 1 | 1 | 1 |

# Open Repository
- [**Machine Learning Privacy Meter**](https://github.com/privacytrustlab/ml_privacy_meter) (NUS Data Privacy and Trustworthy Machine Learning Lab)
- [**Adversarial Robustness Toolbox (ART)**](https://github.com/Trusted-AI/adversarial-robustness-toolbox) (IBM)

# Related Link
- [**Membership Inference Attacks and Defenses on Machine Learning Models Literature**](https://github.com/HongshengHu/membership-inference-machine-learning-literature)
- [**Awesome Attacks on Machine Learning Privacy**](https://github.com/stratosphereips/awesome-ml-privacy-attacks#privacy-testing-tools)