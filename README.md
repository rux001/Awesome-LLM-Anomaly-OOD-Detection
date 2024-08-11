# Awesome-LLM-OOD-Anomaly

Tracking advancements in "Large Language Models for Anomaly and Out-of-Distribution Detection", based on our detailed survey found at [Large Language Models for Anomaly and Out-of-Distribution Detection: A Survey]().

## Table of Contents

- [Introduction](#introduction)
- [Taxonomy](#taxonomy)
  - [LLMs for Enhancement](#llms-for-enhancement)
  - [LLMs for Detection](#llms-for-detection)
  - [LLMs for Explanation](#llms-for-explanation)
- [Citation](#citation)

## Introduction

![Overview of LLMs for Anomaly and OOD Detection](intro22.png)

Detecting anomalies and out-of-distribution (OOD) samples is crucial for maintaining reliable and trustworthy machine learning systems. Recently, Large Language Models (LLMs) have gained significant attention in natural language processing due to their excellence in language comprehension and summarization. With advancements in multimodal understanding, LLMs can be broadly applied for anomaly and OOD detection, particularly with a focus on vision data. Integrating LLMs for these tasks has transformed traditional approaches, leveraging their potential across various data modalities and datasets. This survey systematically reviews recent works applying LLMs to anomaly and OOD detection, categorizing them based on the role of LLMs: Detection, Enhancement, and Explanation, while highlighting potential avenues for future research.
## Taxonomy

### LLMs for Enhancement

Exploring how LLMs support the enhancement of detection capabilities without being direct detectors.

![LLMs for Enhancement](/path/to/enhancement_image.png)

| Paper | Authors | Backbone Model | Task Category | Dataset Type| Venue |
|-------|---------|----------------|---------------|-------------|---------|
| [Envisioning outlier exposure by large language models for out-of-distribution detection](https://arxiv.org/abs/2406.00806) | Chentao Cao, Zhun Zhong, Zhanke Zhou, Yang Liu, Tongliang Liu, Bo Han | GPT-3.5-turbo-16k; CLIP | OOD Detection | Images | ICML, 2024 |
| [Out-of-Distribution Detection Using Peer-Class Generated by Large Language Model](https://arxiv.org/abs/2403.13324) | K Huang, G Song, Hanwen Su, Jiyan Wang | GPT-3; CLIP | OOD Detection | Images | ArXiv, 2024 |
| [On the Powerfulness of Textual Outlier Exposure for Visual OoD Detection](https://openreview.net/forum?id=090ORrOAPL) | Sangha Park and Jisoo Mok and Dahuin Jung and Saehyung Lee and Sungroh Yoon | BERT; BLIP-2; GPT-3; CLIP | OOD Detection | Images | NeurIPS, 2023 |
| [Contrastive Novelty-Augmented Learning: Anticipating Outliers with Large Language Models](https://aclanthology.org/2023.acl-long.658.pdf) | Albert Xu, Xiang Ren, Robin Jia | GPT-3; GPT-J; BERT | OOD Detection | Texts | ACL, 2023 |
| [Tagfog: Textual anchor guidance and fake outlier generation for visual out-of-distribution detection](https://ojs.aaai.org/index.php/AAAI/article/view/27871) | Jiankang Chen, Tong Zhang, Weishi Zheng, Ruixuan Wang | ChatGPT; CLIP | OOD Detection | Images | AAAI, 2024 |
| [Do LLMs Understand Visual Anomalies? Uncovering LLM Capabilities in Zero-shot Anomaly Detection](https://arxiv.org/pdf/2404.09654) | Jiaqi Zhu, Shaofeng Cai, Fang Deng, Junran Wu | GPT-3.5; CLIP | Anomaly Detection | Images | ArXiv, 2024 |
| [Exploring large language models for multi-modal out-of-distribution detection](https://arxiv.org/abs/2310.08027) | Yi Dai, Hao Lang, Kaisheng Zeng, Fei Huang, Yongbin Li | text-davinci-003; CLIP | OOD Detection | Images | EMNLP, 2023 |
| [LogGPT: Exploring ChatGPT for log-based anomaly detection](https://arxiv.org/pdf/2309.01189) | Jiaxing Qi et al. | ChatGPT | Anomaly Detection | Log Data | IEEE HPCC, 2023 |
| [LogBERT: Log anomaly detection via BERT](https://arxiv.org/pdf/2103.04475) | Haixuan Guo et al. | BERT | Anomaly Detection | Log Data | IJCNN, 2021 |
| [NeuralLog: Natural Language Inference with Joint Neural and Logical Reasoning](https://aclanthology.org/2021.starsem-1.7.pdf) | Zeming Chen et al. | BERT | Log Data | Texts | SEM, 2021 |
| [LogFiT: Log Anomaly Detection Using Fine-Tuned Language Models](https://ieeexplore.ieee.org/document/10414427) | Crispin Almodovar et al. | Various LLMs | Anomaly Detection | Log Data | IEEE TNSM 2024 |
| [How good are LLMs at out-of-distribution detection?](https://aclanthology.org/2024.lrec-main.720.pdf) | Andi Zhang et al. | LLaMA etc. | OOD Detection | Various | COLING, 2024 |
| [Your Finetuned Large Language Model is Already a Powerful Out-of-distribution Detector](https://arxiv.org/abs/2404.08679) | Andi Zhang, Tim Z Xiao, Weiyang Liu, Robert Bamler, Damon Wischik | Various LLMs | OOD Detection | Texts | arXiv, 2024 |
| [Large language model guided knowledge distillation for time series anomaly detection](https://arxiv.org/abs/2401.15123v1) | Chen Liu, Shibo He, Qihang Zhou, Shizhong Li, Wenchao Meng | GPT2 | Anomaly Detection | Time Series | arXiv, 2024 |

### LLMs for Detection

Highlighting how LLMs directly contribute to detecting anomalies and out-of-distribution samples.

![LLMs for Detection](/path/to/detection_image.png)

| Paper | Authors | Backbone Model | Task Category | Dataset Type| Venue |
|-------|---------|----------------|---------------|-------------|---------|
| [WinCLIP: Zero-/few-shot anomaly classification and segmentation](https://arxiv.org/pdf/2303.14814) | Jongheon Jeong, Yang Zou, Taewan Kim, Dongqing Zhang, Avinash Ravichandran, Onkar Dabeer | CLIP | Anomaly Detection | Images | CVPR, 2023 |
| [CLIP-AD: A language-guided staged dual-path model for zero-shot anomaly detection](https://arxiv.org/abs/2311.00453) | Xuhai Chen, Jiangning Zhang, Guanzhong Tian, Haoyang He, Wuhao Zhang, Yabiao Wang, Chengjie Wang, Yunsheng Wu, Yong Liu | CLIP | Anomaly Detection | Images | arXiv, 2023 |
| [Exploring grounding potential of VQA-oriented GPT-4V for zero-shot anomaly detection](https://arxiv.org/abs/2311.02612) | Jiangning Zhang, Xuhai Chen, Zhucun Xue, Yabiao Wang, Chengjie Wang, Yong Liu | GPT-4V | Anomaly Detection | Images | arXiv, 2023 |
| [CLIPScope: Enhancing Zero-Shot OOD Detection with Bayesian Scoring](https://arxiv.org/pdf/2405.14737) | Hao Fu, Naman Patel, Prashanth Krishnamurthy, Farshad Khorrami | CLIP | OOD Detection | Images | ArXiv, 2024 |
| [AnomalyCLIP: Object-agnostic prompt learning for zero-shot anomaly detection](https://openreview.net/forum?id=buC4E91xZE) | Qihang Zhou, Guansong Pang, Yu Tian, Shibo He, Jiming Chen | CLIP | Anomaly Detection | Images | ICLR, 2024 |
| [Toward generalist anomaly detection via in-context residual learning with few-shot sample prompts](https://arxiv.org/abs/2403.06495) | Jiawen Zhu, Guansong Pang | CLIP | Anomaly Detection | Images | CVPR, 2024 |
| [PromptAD: Learning prompts with only normal samples for few-shot anomaly detection](https://arxiv.org/abs/2404.05231) | Xiaofan Li, Zhizhong Zhang, Xin Tan, Chengwei Chen, Yanyun Qu, Yuan Xie, Lizhuang Ma | CLIP | Anomaly Detection | Images | CVPR, 2024 |
| [Text prompt with normality guidance for weakly supervised video anomaly detection](https://arxiv.org/abs/2404.08531) | Zhiwei Yang, Jing Liu, Peng Wu | CLIP | Anomaly Detection | Videos | arXiv, 2024 |
| [LoCoOp: Few-Shot Out-of-Distribution Detection via Prompt Learning](https://arxiv.org/pdf/2306.01293) | Miyai et al. | CLIP | OOD Detection | Images | NeurIPS, 2023 |
| [CLIPN for Zero-Shot OOD Detection: Teaching CLIP to Say No](https://arxiv.org/pdf/2308.12213) | Wang et al. | CLIP | OOD Detection | Images | ICCV, 2023 |
| [Out-Of-Distribution Detection With Negative Prompts](https://openreview.net/pdf?id=nanyAujl6e) | Nie et al. | CLIP | OOD Detection | Images | ICLR, 2024 |
| [ID-like Prompt Learning for Few-Shot Out-of-Distribution Detection](https://arxiv.org/pdf/2311.15243) | Yichen Bai, Zongbo Han, Changqing Zhang, Bing Cao, Xiaoheng Jiang, Qinghua Hu | CLIP | OOD Detection | Images | CVPR, 2024 |
| [Learning transferable negative prompts for out-of-distribution detection](https://arxiv.org/abs/2404.03248) | Tianqi Li, Guansong Pang, Xiao Bai, Wenjun Miao, Jin Zheng | CLIP | OOD Detection | Images | CVPR, 2024 |
| [AnomalyGPT: Detecting Industrial Anomalies Using Large Vision-Language Models](https://arxiv.org/pdf/2308.15366) | Zhaopeng Gu, Bingke Zhu, Guibo Zhu, Yingying Chen, Ming Tang, Jinqiao Wang | ImageBind-Huge; Vicuna-7B | Anomaly Detection | Images | ArXiv, 2024 |
| [Adapting visual-language models for generalizable anomaly detection in medical images](https://arxiv.org/abs/2403.12570v1) | Chaoqin Huang, Aofan Jiang, Jinghao Feng, Ya Zhang, Xinchao Wang, Yanfeng Wang | CLIP | Anomaly Detection | Images | CVPR, 2024 |
| [Adapting Contrastive Language-Image Pretrained (CLIP) Models for Out-of-Distribution Detection](https://openreview.net/pdf?id=YCgX7sJRF1) | Nikolas Adaloglou, Felix Michels, Tim Kaiser, Markus Kollmann | CLIP | OOD Detection | Images | TMLR, 2024 |
| [Video anomaly detection and explanation via large language models](https://arxiv.org/abs/2401.05702) | Hui Lv, Qianru Sun | Video-LLaMA | Anomaly Detection | Videos | arXiv, 2024 |
| [VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection](https://arxiv.org/abs/2401.05702) | Peng Wu, Xuerong Zhou, Guansong Pang, Lingru Zhou, Qingsen Yan, Peng Wang, Yanning Zhang | CLIP | Anomaly Detection | Videos | AAAI, 2023 |
| [Delving into Out-of-Distribution Detection with Vision-Language Representations](https://arxiv.org/pdf/2211.13445.pdf) | Ming et al. | CLIP | OOD Detection | Images | NeurIPS, 2022 |
| [Text prompt with normality guidance for weakly supervised video anomaly detection](https://arxiv.org/abs/2404.08531) | Zhiwei Yang, Jing Liu, Peng Wu | CLIP | Anomaly Detection | Videos | arXiv, 2024 |
| [Negative Label Guided OOD Detection with Pretrained Vision-Language Models](https://arxiv.org/pdf/2403.20078) | Xue Jiang, Feng Liu, Zhen Fang, Hong Chen, Tongliang Liu, Feng Zheng, Bo Han | CLIP; ALIGN; GroupViT; AltCLIP | OOD Detection | Images | ICLR, 2024 
| [Zero-shot out-of-distribution detection based on the pretrained model CLIP](https://arxiv.org/pdf/2109.02748) | Sepideh Esmaeilpour, Bing Liu, Eric Robertson, Lei Shu | CLIP | OOD Detection | Images | AAAI, 2022 |
| [Large language models can be zero-shot anomaly detectors for time series?](https://arxiv.org/abs/2405.14755) | Sarah Alnegheimish, Linh Nguyen, Laure Berti-Equille, Kalyan Veeramachaneni | Mistral-7B-Instruct-v0.2; gpt-3.5-turbo-instruct | Anomaly Detection | Time Series | arXiv, 2024 |
| [Semantic anomaly detection with large language models](https://arxiv.org/abs/2305.11307) | Amine Elhafsi, Rohan Sinha, Christopher Agia, Edward Schmerling, Issa Nesnas, Marco Pavone | text-davinci-003 | Anomaly Detection | Videos | arXiv, 2023 |
| [Harnessing large language models for training-free video anomaly detection](https://arxiv.org/abs/2404.01014) | Luca Zanella, Willi Menapace, Massimiliano Mancini, Yiming Wang, Elisa Ricci | Llama-2-13b-chat; ImageBind | Anomaly Detection | Videos | CVPR, 2024 |
| [Large language models can deliver accurate and interpretable time series anomaly detection](https://arxiv.org/abs/2405.15370) | Jiaqi Tang, Hao Lu, Ruizheng Wu, Xiaogang Xu, Ke Ma, Cheng Fang, Bin Guo, Jiangbo Lu, Qifeng Chen, Ying-Cong Chen | GPT-4-1106-preview | Anomaly Detection | Time Series | arXiv, 2024 |
| [FiLo: Zero-shot anomaly detection by fine-grained description and high-quality localization](https://arxiv.org/abs/2404.08531) | Zhaopeng Gu, Bingke Zhu, Guibo Zhu, Yingying Chen, Hao Li, Ming Tang, Jinqiao Wang | CLIP | Anomaly Detection | Vision | arXiv, 2024 |

### LLMs for Explanation

Detailing how LLMs aid in explaining the detection results, enhancing understanding and trust.

![LLMs for Explanation](/path/to/explanation_image.png)

| Paper | Authors | Backbone Model | Task Category | Dataset Type| Venue |
|-------|---------|----------------|---------------|-------------|---------|
| [Holmes-VAD: Towards Unbiased and Explainable Video Anomaly Detection via Multi-modal LLM](https://arxiv.org/pdf/2406.12235) | Huaxin Zhang, Xiaohao Xu, Xiang Wang, Jialong Zuo, Chuchu Han, Xiaonan Huang, Changxin Gao, Yuehuan Wang, Nong Sang | Video-LLaVA | Anomaly Detection | Videos | ArXiv, 2024 |
| [Follow the Rules: Reasoning for Video Anomaly Detection with Large Language Models](https://arxiv.org/html/2407.10299v1) | Yuchen Yang, Kwonjoon Lee, Behzad Dariush, Yinzhi Cao, Shao-Yuan Lo |  CogVLM-17B; GPT-4; Mistral-7B-Instruct-v0.2 | Anomaly Detection | Videos | ArXiv, 2024 |
| [Video Anomaly Detection and Explanation via Large Language Models](https://arxiv.org/html/2401.05702v1) | Lv et al. | LLaMA | Anomaly Detection | Videos | ICCV, 2024 |
| [Real-Time Anomaly Detection and Reactive Planning with Large Language Models](https://arxiv.org/abs/2406.09876) | Sinha et al. | BERT, Llama 2 etc. | Anomaly Detection | Robotic Data | ArXiv, 2024 |

## Citation

If you find this work useful, please cite our survey paper:

         
