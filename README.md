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

## Membership Inference
| Year | Title | Adversarial Knowledge | Target Model | Link |
| - | - | - | - | - |
| 22'USENIX | [**Label Inference Attacks Against Vertical Federated Learning**](https://www.usenix.org/conference/usenixsecurity22/presentation/fu) | auxiliary labeled dataset <br> gradient | vertical FL | |
| 22'USENIX | [**Inference Attacks Against Graph Neural Networks**](https://arxiv.org/abs/2110.02631) | auxiliary dataset <br> graph embedding | Graph NN | [code](https://github.com/Zhangzhk0819/GNN-Embedding-Leaks) |
| 21'ICML | [**Label-Only Membership Inference Attacks**](https://arxiv.org/abs/2007.14321) | auxiliary dataset <br> label | (image) clssification | [code](https://github.com/cchoquette/membership-inference) |
| 21'CCS | [**Membership Inference Attacks Against Recommender Systems**](https://arxiv.org/abs/2109.08045) | auxiliary dataset <br> user's history <br> item list | recommendation |  |
| 21'CCS | [**EncoderMI: Membership Inference against Pre-trained Encoders in Contrastive Learning**](https://arxiv.org/abs/2108.11023) | auxiliary dataset(iid) <br> embedding | (image) pre-trained | |
| 21'CCS | [**Membership Leakage in Label-Only Exposures**](https://arxiv.org/abs/2007.15528) | auxiliary dataset(iid) <br> label | classification | [code](https://github.com/zhenglisec/Decision-based-MIA) |
| 21'CCS | [**TableGAN-MCA: Evaluating Membership Collisions of GAN-Synthesized Tabular Data Releasing**](https://arxiv.org/abs/2107.13190) | synthetic dataset | (Table) GAN |  |
| 21'CCS | [**When Machine Unlearning Jeopardizes Privacy**](https://arxiv.org/abs/2005.02205) | shadow dataset <br> confidence(two) | machine unlearning | [code](https://github.com/MinChen00/UnlearningLeaks) | 
| 17'S&P | [**Membership Inference Attacks Against Machine Learning Models**](https://arxiv.org/abs/1610.05820) | auxiliary dataset <br> confidence | classification | [code](https://github.com/csong27/membership-inference) |


## Attribute/Property Inference
| Year | Title | Adversarial Knowledge | Target Model | Attribute | Link |
| - | - | - | - | - | - |
| 21'CCS | [**Quantifying and Mitigating Privacy Risks of Contrastive Learning**](https://arxiv.org/abs/2102.04140) |  |  |  | [code](https://github.com/xinleihe/ContrastiveLeaks) |
| 21'CCS | [**Honest-but-Curious Nets: Sensitive Attributes of Private Inputs Can Be Secretly Coded into the Classifiers' Outputs**](https://arxiv.org/abs/2105.12049) |  |  |  | [code](https://github.com/mmalekzadeh/honest-but-curious-nets) |
| 21'ICDE | [**Feature Inference Attack on Model Predictions in Vertical Federated Learning**](https://arxiv.org/abs/2010.10152) |  |  |  | [code](https://github.com/xj231/featureinference-vfl) |
| 20'ICML | [**Overlearning Reveals Sensitive Attributes**](https://arxiv.org/abs/1905.11742) |  |  |  |  |
| 19'S&P | [**Exploiting Unintended Feature Leakage in Collaborative Learning**](https://arxiv.org/abs/1805.04049) | | | | [code](https://github.com/csong27/property-inference-collaborative-ml) |
| 18'USENIX | [**AttriGuard: A Practical Defense Against Attribute Inference Attacks via Adversarial Machine Learning**](https://www.usenix.org/system/files/conference/usenixsecurity18/sec18-jia.pdf) |  |  |  | [code](https://github.com/jjy1994/AttriGuard) |


## Model Inversion
| Year | Title | Restore |Adversarial Knowledge | Target Model | Link |
| - | - | - | - | - | - |
| 19'USENIX | [**Updates-Leak: Data Set Inference and Reconstruction Attacks in Online Learning**](https://arxiv.org/abs/1904.01067) | - | - | - | [code](https://github.com/AhmedSalem2/Updates-Leak) |
| 20'arXiv | [**Defending Model Inversion and Membership Inference Attacks via Prediction Purification**](https://arxiv.org/abs/2005.03915) | - | - | - |  |
| 20'CVPR | [**The Secret Revealer: Generative Model-Inversion Attacks Against Deep Neural Networks**](https://arxiv.org/abs/1911.07135) | - | - | - |  |
| 20'CCS | [**Information Leakage in Embedding Models**](https://arxiv.org/abs/2004.00053) | - | - | - | [code](https://github.com/google/embedding-tests) |
| 15'CCS | [**Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures**](https://dl.acm.org/doi/10.1145/2810103.2813677) | - | - | - | [code](https://github.com/yashkant/Model-Inversion-Attack) |
| 14'CCS | [**Privacy in Pharmacogenetics: An End-to-End Case Study of Personalized Warfarin Dosing**](https://www.usenix.org/system/files/conference/usenixsecurity14/sec14-paper-fredrikson-privacy.pdf) | - | - | - | |


## Model Extraction/Stealing
| Year | Title | Infer |Adversarial Knowledge | Target Model | Link |
| - | - | - | - | - | - |
| 16'USENIX | [**Stealing Machine Learning Models via Prediction APIs**](https://arxiv.org/abs/1609.02943) | parameter | confidence/label | LR/DT/SVM/NN | [code](https://github.com/ftramer/Steal-ML) |
| 18'ICLR | [**Towards Reverse-Engineering Black-Box Neural Networks**](https://arxiv.org/abs/1711.01768) | hyperparameter | data distribution <br> NN family | Image classification | [code](https://github.com/coallaoh/WhitenBlackBox) |
| 18'CVPR | [**Knockoff Nets: Stealing Functionality of Black-Box Models**](https://arxiv.org/abs/1812.02766) | functionality | public data <br> confidence  | Image classification | [code](https://github.com/tribhuvanesh/knockoffnets) |
| 18'S&P | [**Stealing Hyperparameters in Machine Learning**](https://arxiv.org/abs/1802.05351) | hyperparameter | training dataset <br> objective function <br> model parameter  | LR/SVM/NN |  |

# Open Repository
- [**Machine Learning Privacy Meter**](https://github.com/privacytrustlab/ml_privacy_meter) (NUS Data Privacy and Trustworthy Machine Learning Lab)
- [**Adversarial Robustness Toolbox (ART)**](https://github.com/Trusted-AI/adversarial-robustness-toolbox) (IBM)

# Related Link
- [**Membership Inference Attacks and Defenses on Machine Learning Models Literature**](https://github.com/HongshengHu/membership-inference-machine-learning-literature)
- [**Awesome Attacks on Machine Learning Privacy**](https://github.com/stratosphereips/awesome-ml-privacy-attacks#privacy-testing-tools)