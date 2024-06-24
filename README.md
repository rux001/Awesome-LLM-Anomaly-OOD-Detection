# Out-of-Distribution/Anomaly Detection with Foundation Models/LLM
## Table of Contents

- [Introduction](#introduction)
- [Tentaive Curated List of Papers](#tentaive-curated-list-of-papers)
  - [Language Foundation Models](#Language-Foundation-models)
  - [Multi-modal Foundation Models](#multi-modal-foundation-models)

## Introduction

## Tentaive Curated List of Papers

### Language Foundation Models

#### Prompting
| Paper | Authors | Backbone Model | Task Category | Dataset Type| Venue | Approach |
|------------------------|--------------|----------------|---------------|------------------|---------|---------|
| [Large language models can be zero-shot anomaly detectors for time series?](https://arxiv.org/abs/2405.14755) | Sarah Alnegheimish, Linh Nguyen, Laure Berti-Equille, Kalyan Veeramachaneni | Mistral-7B-Instruct-v0.2; gpt-3.5-turbo-instruct | Anomaly Detection | Time Series | KDD, 2024 | Prompting |
| [Semantic anomaly detection with large language models](https://arxiv.org/abs/2305.11307) | Amine Elhafsi, Rohan Sinha, Christopher Agia, Edward Schmerling, Issa Nesnas, Marco Pavone | text-davinci-003 | Anomaly Detection | Vision | arXiv, 2023 | Prompting |
| [LogGPT: Exploring ChatGPT for log-based anomaly detection](https://arxiv.org/abs/2309.01189) | Jiaxing Qi, Shaohan Huang, Zhongzhi Luan, Carol Fung, Hailong Yang, Depei Qian | ChatGPT | Anomaly Detection | Log Data | arXiv, 2023 | Prompting |
| [Harnessing large language models for training-free video anomaly detection](https://arxiv.org/abs/2404.01014) | Luca Zanella, Willi Menapace, Massimiliano Mancini, Yiming Wang, Elisa Ricci | Llama-2-13b-chat; ImageBind | Anomaly Detection | Video | CVPR, 2024 | Prompting |
| [Do LLMs understand visual anomalies? Uncovering LLM capabilities in zero-shot anomaly detection](https://arxiv.org/abs/2404.09654) | Jiaqi Zhu, Shaofeng Cai, Fang Deng, Junran Wu | CLIP; GPT-3.5 | Anomaly Detection | Vision | arXiv, 2024 | Prompting |
| [Large language models can deliver accurate and interpretable time series anomaly detection](https://arxiv.org/abs/2405.15370) | Jiaqi Tang, Hao Lu, Ruizheng Wu, Xiaogang Xu, Ke Ma, Cheng Fang, Bin Guo, Jiangbo Lu, Qifeng Chen, Ying-Cong Chen | GPT-4-1106-preview | Anomaly Detection | Time Series | arXiv, 2024 | Prompting |
| [Envisioning outlier exposure by large language models for out-of-distribution detection](https://arxiv.org/abs/2405.15370) | Chentao Cao, Zhun Zhong, Zhanke Zhou, Yang Liu, Tongliang Liu, Bo Han | GPT-3.5-turbo-16k | OOD Detection | Vision | ICML, 2024 | Prompting |

#### Knowledge Distillation
| Paper | Authors | Backbone Model | Task Category | Dataset Type| Venue | Approach |
|------------------------|--------------|----------------|---------------|------------------|---------|---------|
| [Large language model guided knowledge distillation for time series anomaly detection](https://arxiv.org/abs/2401.15123v1) | Chen Liu, Shibo He, Qihang Zhou, Shizhong Li, Wenchao Meng | GPT2 | Anomaly Detection | Time Series | arXiv, 2024 | Knowledge Distillation |
| [Multi-Level Knowledge Distillation for Out-of-Distribution Detection in Text](https://arxiv.org/pdf/2211.11300) | Multi-Level Knowledge Distillation for Out-of-Distribution Detection in Text | GPT-2-small | OOD Detection | Text | ACL, 2023 | Knowledge Distillation |
| [GOLD: Generalized Knowledge Distillation via Out-of-Distribution-Guided Language Data Generation](https://arxiv.org/pdf/2211.11300) | Multi-Level Knowledge Distillation for Out-of-Distribution Detection in Text | LLaMA2 | OOD Detection | Text | arXiv, 2024 | Prompting; Knowledge Distillation |

#### Integrating

| Paper | Authors | Backbone Model | Task Category | Dataset Type| Venue | Approach |
|------------------------|--------------|----------------|---------------|------------------|---------|---------|
| [Your finetuned large language model is already a powerful out-of-distribution detector](https://arxiv.org/abs/2404.08679) | Andi Zhang, Tim Z. Xiao, Weiyang Liu, Robert Bamler, Damon Wischik | Llama; Mistral | OOD Detection | Text | arXiv, 2024 | Integrating |
| [Out-of-distribution detection and selective generation for conditional language models](https://arxiv.org/pdf/2209.15558) | Jie Ren, Jiaming Luo, Yao Zhao, Kundan Krishna, Mohammad Saleh, Balaji Lakshminarayanan, Peter J. Liu | Transformer | OOD Detection | Texts | ICLR, 2023 | Integrating |
| [How Good Are LLMs at Out-of-Distribution Detection?](https://arxiv.org/pdf/2308.10261) | Bo Liu, Liming Zhan, Zexin Lu, Yujie Feng, Lei Xue, Xiao-Ming Wu | LLaMA | OOD Detection | Text | COLING, 2024 | Integrating |
| [Exploring large language models for multi-modal out-of-distribution detection](https://arxiv.org/abs/2310.08027) | Yi Dai, Hao Lang, Kaisheng Zeng, Fei Huang, Yongbin Li | text-davinci-003; CLIP | OOD Detection | Vision | EMNLP, 2023 | Integrating |
| [FiLo: Zero-shot anomaly detection by fine-grained description and high-quality localization](https://arxiv.org/abs/2404.08531) | Zhaopeng Gu, Bingke Zhu, Guibo Zhu, Yingying Chen, Hao Li, Ming Tang, Jinqiao Wang | CLIP | Anomaly Detection | Vision | arXiv, 2024 | Integrating |

### Multi-modal Foundation Models

### Prompting
| Paper | Authors | Backbone Model | Task Category | Dataset Type| Venue | Approach |
|------------------------|--------------|----------------|---------------|------------------|---------|---------|
| [WinCLIP: Zero-/few-shot anomaly classification and segmentation](https://arxiv.org/pdf/2303.14814) | Jongheon Jeong, Yang Zou, Taewan Kim, Dongqing Zhang, Avinash Ravichandran, Onkar Dabeer | CLIP | Anomaly Detection | Vision | CVPR, 2023 | Prompting; Integrating |
| [CLIP-AD: A language-guided staged dual-path model for zero-shot anomaly detection](https://arxiv.org/abs/2311.00453) | Xuhai Chen, Jiangning Zhang, Guanzhong Tian, Haoyang He, Wuhao Zhang, Yabiao Wang, Chengjie Wang, Yunsheng Wu, Yong Liu | CLIP | Anomaly Detection | Vision | arXiv, 2023 | Prompting; Integrating |
| [Exploring grounding potential of VQA-oriented GPT-4V for zero-shot anomaly detection](https://arxiv.org/abs/2311.02612) | Jiangning Zhang, Xuhai Chen, Zhucun Xue, Yabiao Wang, Chengjie Wang, Yong Liu | GPT-4V | Anomaly Detection | Vision | arXiv, 2023 | Prompting; Integrating |
| [Tagfog: Textual anchor guidance and fake outlier generation for visual out-of-distribution detection](https://ojs.aaai.org/index.php/AAAI/article/view/27871) | Jiankang Chen, Tong Zhang, Weishi Zheng, Ruixuan Wang | CLIP; ChatGPT | OOD Detection | Vision | AAAI, 2024 | Prompting; Integrating |

### Prompt-Tuning
| Paper | Authors | Backbone Model | Task Category | Dataset Type | Venue | Approach |
|------------------------|--------------|----------------|---------------|------------------|---------|---------|
| [AnomalyCLIP: Object-agnostic prompt learning for zero-shot anomaly detection](https://openreview.net/forum?id=buC4E91xZE) | Qihang Zhou, Guansong Pang, Yu Tian, Shibo He, Jiming Chen | CLIP | Anomaly Detection | Vision | ICLR, 2024 | Prompt-Tuning |
| [Toward generalist anomaly detection via in-context residual learning with few-shot sample prompts](https://arxiv.org/abs/2403.06495) | Jiawen Zhu, Guansong Pang | CLIP | Anomaly Detection | Vision | CVPR, 2024 | Prompt-Tuning |
| [PromptAD: Learning prompts with only normal samples for few-shot anomaly detection](https://arxiv.org/abs/2404.05231) | Xiaofan Li, Zhizhong Zhang, Xin Tan, Chengwei Chen, Yanyun Qu, Yuan Xie, Lizhuang Ma | CLIP | Anomaly Detection | Vision | CVPR, 2024 | Prompt-Tuning |
| [Text prompt with normality guidance for weakly supervised video anomaly detection](https://arxiv.org/abs/2404.08531) | Zhiwei Yang, Jing Liu, Peng Wu | CLIP | Anomaly Detection | Video | arXiv, 2024 | Prompt-Tuning |
| [LoCoOp: Few-Shot Out-of-Distribution Detection via Prompt Learning](https://arxiv.org/pdf/2306.01293) | Miyai et al. | CLIP | OOD Detection | Vision | NeurIPS, 2023 | Prompt-Tuning |
| [CLIPN for Zero-Shot OOD Detection: Teaching CLIP to Say No](https://arxiv.org/pdf/2308.12213) | Wang et al. | CLIP | OOD Detection | Vision | ICCV, 2023 | Prompt-Tuning |
| [Out-Of-Distribution Detection With Negative Prompts](https://openreview.net/pdf?id=nanyAujl6e) | Nie et al. | CLIP | OOD Detection | Vision | ICLR, 2024 | Prompt-Tuning |
| [ID-like Prompt Learning for Few-Shot Out-of-Distribution Detection](https://arxiv.org/pdf/2311.15243) | Yichen Bai, Zongbo Han, Changqing Zhang, Bing Cao, Xiaoheng Jiang, Qinghua Hu | CLIP | OOD Detection | Vision | CVPR, 2024 | Prompt-Tuning |
| [Learning transferable negative prompts for out-of-distribution detection](https://arxiv.org/abs/2404.03248) | Tianqi Li, Guansong Pang, Xiao Bai, Wenjun Miao, Jin Zheng | CLIP | OOD Detection | Vision | CVPR, 2024 | Prompt-Tuning |

### Adapter-Tuning
| Paper | Authors | Backbone Model | Task Category | Dataset Type | Venue | Approach |
|------------------------|--------------|----------------|---------------|------------------|---------|---------|
| [Adapting visual-language models for generalizable anomaly detection in medical images](https://arxiv.org/abs/2403.12570v1) | Chaoqin Huang, Aofan Jiang, Jinghao Feng, Ya Zhang, Xinchao Wang, Yanfeng Wang | CLIP | Anomaly Detection | Medical Images | CVPR, 2024 | Adapter-Tuning |
| [Adapting Contrastive Language-Image Pretrained (CLIP) Models for Out-of-Distribution Detection](https://openreview.net/pdf?id=YCgX7sJRF1) | Nikolas Adaloglou, Felix Michels, Tim Kaiser, Markus Kollmann | CLIP | OOD Detection | Vision | TMLR, 2024 | Adapter-Tuning |
| [Video anomaly detection and explanation via large language models](https://arxiv.org/abs/2401.05702) | Hui Lv, Qianru Sun | Video-LLaMA | Anomaly Detection | Video | arXiv, 2024 | Adapter-Tuning |
| [VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection](https://arxiv.org/abs/2401.05702) | Peng Wu, Xuerong Zhou, Guansong Pang, Lingru Zhou, Qingsen Yan, Peng Wang, Yanning Zhang | CLIP | Anomaly Detection | Video | AAAI, 2023 | Prompt-Tuning; Adapter-Tuning |

### Knowledge Distillation
| Paper | Authors | Backbone Model | Task Category | Dataset Type | Venue | Approach |
|------------------------|--------------|----------------|---------------|------------------|---------|---------|
| [Distilling out-of-distribution robustness from vision-language foundation models](https://neurips.cc/virtual/2023/poster/70716) | Andy Zhou, Jindong Wang, Yuxiong Wang, Haohan Wang | CLIP | OOD Detection | Vision | NeurIPS, 2023 | Knowledge Distillation |

### Integrating
| Paper | Authors | Backbone Model | Task Category | Dataset Type | Venue | Approach |
|------------------------|--------------|----------------|---------------|------------------|---------|---------|
| [How Does Fine-Tuning Impact Out-of-Distribution Detection for Vision-Language Models?](https://arxiv.org/pdf/2306.06048) | Yifei Ming, Yixuan Li | CLIP | OOD Detection | Vision | IJCV, 2023 | Prompt-Tuning; Integrating |
| [Envisioning outlier exposure by large language models for out-of-distribution detection](https://arxiv.org/abs/2405.15370) | Chentao Cao, Zhun Zhong, Zhanke Zhou, Yang Liu, Tongliang Liu, Bo Han | CLIP | OOD Detection | Vision | ICML, 2024 | Integrating |
| [Delving into Out-of-Distribution Detection with Vision-Language Representations](https://arxiv.org/pdf/2211.13445.pdf) | Ming et al. | CLIP | OOD Detection | Vision | NeurIPS, 2022 | Integrating; Aligning |
| [Text prompt with normality guidance for weakly supervised video anomaly detection](https://arxiv.org/abs/2404.08531) | Zhiwei Yang, Jing Liu, Peng Wu | CLIP | Anomaly Detection | Video | arXiv, 2024 | Prompt-Tuning; Integrating |
| [Negative Label Guided OOD Detection with Pretrained Vision-Language Models](https://arxiv.org/pdf/2403.20078) | Xue Jiang, Feng Liu, Zhen Fang, Hong Chen, Tongliang Liu, Feng Zheng, Bo Han | CLIP; ALIGN; GroupViT; AltCLIP | OOD Detection | Vision | ICLR, 2024 | Integrating; Aligning |
| [Zero-shot out-of-distribution detection based on the pretrained model CLIP](https://arxiv.org/pdf/2109.02748) | Sepideh Esmaeilpour, Bing Liu, Eric Robertson, Lei Shu | CLIP | OOD Detection | Vision | AAAI, 2022 | Integrating; Aligning |
| [Exploring large language models for multi-modal out-of-distribution detection](https://arxiv.org/abs/2310.08027) | Yi Dai, Hao Lang, Kaisheng Zeng, Fei Huang, Yongbin Li | text-davinci-003; CLIP | OOD Detection | Vision | EMNLP, 2023 | Integrating; Aligning |


         
