# Out-of-Distribution/Anomaly Detection with Foundation Models/LLM
## Table of Contents

- [Introduction](#introduction)
- [Tentaive Curated List of Papers](#tentaive-curated-list-of-papers)
  - [Language Foundation Models](#Language-Foundation-models)
  - [Vision Foundation Models](#vision-foundation-models)
  - [Multi-modal Foundation Models](#multi-modal-foundation-models)

## Introduction

## Tentaive Curated List of Papers

### Language Foundation Models

## Prompting

| Paper | Authors | Backbone Model | Task Category | Dataset Type| Venue | Approach |
|------------------------|--------------|----------------|---------------|------------------|---------|---------|
| [Semantic anomaly detection with large language models](https://arxiv.org/abs/2305.11307) | Amine Elhafsi, Rohan Sinha, Christopher Agia, Edward Schmerling, Issa Nesnas, Marco Pavone | text-davinci-003 | Anomaly Detection | Vision | arXiv, 2023 | Prompting |
| [LogGPT: Exploring ChatGPT for log-based anomaly detection](https://arxiv.org/abs/2309.01189) | Jiaxing Qi, Shaohan Huang, Zhongzhi Luan, Carol Fung, Hailong Yang, Depei Qian | ChatGPT | Anomaly Detection | Log Data | arXiv, 2023 | Prompting |
| [Exploring large language models for multi-modal out-of-distribution detection](https://arxiv.org/abs/2310.08027) | Yi Dai, Hao Lang, Kaisheng Zeng, Fei Huang, Yongbin Li | text-davinci-003; CLIP | OOD Detection | Vision | EMNLP, 2023 | Prompting |
| [Harnessing large language models for training-free video anomaly detection](https://arxiv.org/abs/2404.01014) | Luca Zanella, Willi Menapace, Massimiliano Mancini, Yiming Wang, Elisa Ricci | Llama-2-13b-chat; ImageBind | Anomaly Detection | Video | CVPR, 2024 | Prompting |
| [Do LLMs understand visual anomalies? Uncovering LLM capabilities in zero-shot anomaly detection](https://arxiv.org/abs/2404.09654) | Jiaqi Zhu, Shaofeng Cai, Fang Deng, Junran Wu | CLIP; GPT-3.5 | Anomaly Detection | Vision | arXiv, 2024 | Prompting |
| [FiLo: Zero-shot anomaly detection by fine-grained description and high-quality localization](https://arxiv.org/abs/2404.08531) | Zhaopeng Gu, Bingke Zhu, Guibo Zhu, Yingying Chen, Hao Li, Ming Tang, Jinqiao Wang | CLIP | Anomaly Detection | Vision | arXiv, 2024 | Prompting |
| [Large language models can deliver accurate and interpretable time series anomaly detection](https://arxiv.org/abs/2405.15370) | Jiaqi Tang, Hao Lu, Ruizheng Wu, Xiaogang Xu, Ke Ma, Cheng Fang, Bin Guo, Jiangbo Lu, Qifeng Chen, Ying-Cong Chen | GPT-4-1106-preview | Anomaly Detection | Time Series | arXiv, 2024 | Prompting |
| [Large language models can be zero-shot anomaly detectors for time series?](https://arxiv.org/abs/2405.14755) | Sarah Alnegheimish, Linh Nguyen, Laure Berti-Equille, Kalyan Veeramachaneni | Mistral-7B-Instruct-v0.2; gpt-3.5-turbo-instruct | Anomaly Detection | Time Series | KDD, 2024 | Prompting |

## Knowledge Distillation

| Paper | Authors | Backbone Model | Task Category | Dataset Type| Venue | Approach |
|------------------------|--------------|----------------|---------------|------------------|---------|---------|
| [Large language model guided knowledge distillation for time series anomaly detection](https://arxiv.org/abs/2401.15123v1) | Chen Liu, Shibo He, Qihang Zhou, Shizhong Li, Wenchao Meng | GPT2 | Anomaly Detection | Time Series | arXiv, 2024 | Knowledge Distillation |

## Integrate

| Paper | Authors | Backbone Model | Task Category | Dataset Type| Venue | Approach |
|------------------------|--------------|----------------|---------------|------------------|---------|---------|
| [Your finetuned large language model is already a powerful out-of-distribution detector](https://arxiv.org/abs/2404.08679) | Andi Zhang, Tim Z. Xiao, Weiyang Liu, Robert Bamler, Damon Wischik | Llama; Mistral | OOD Detection | Text | arXiv, 2024 | Integrate |
| [Out-of-distribution detection and selective generation for conditional language models](https://arxiv.org/pdf/2209.15558) | Jie Ren, Jiaming Luo, Yao Zhao, Kundan Krishna, Mohammad Saleh, Balaji Lakshminarayanan, Peter J. Liu | Transformer | OOD Detection | Texts | ICLR, 2023 | Integrate |
| [How Good Are LLMs at Out-of-Distribution Detection?](https://arxiv.org/pdf/2308.10261) | Bo Liu, Liming Zhan, Zexin Lu, Yujie Feng, Lei Xue, Xiao-Ming Wu | LLaMA | OOD Detection | Text | COLING, 2024 | Integrate |

   
### Vision Foundation Models
## Reconstruction-based
| Paper | Authors | Backbone Model | Task Category | Dataset Type | Venue | Approach |
| --- | --- | --- | --- | --- | --- | --- |
| [Exploring plain ViT reconstruction for multi-class unsupervised anomaly detection](https://arxiv.org/abs/2312.07495) | Jiangning Zhang, Xuhai Chen, Yabiao Wang, Chengjie Wang, Yong Liu, Xiangtai Li, Ming-Hsuan Yang, Dacheng Tao | ViT | Anomaly Detection | Vision | arXiv, 2023 | Reconstruction-based |
| [Multi-contextual predictions with vision transformer for video anomaly detection](https://arxiv.org/abs/2206.08568?context=cs) | Joo-Yeon Lee, Woo-Jeoung Nam, Seong-Whan Lee | ViT | Anomaly Detection | Video | arXiv, 2022 | Reconstruction-based |
| [Generalizable industrial visual anomaly detection with self-induction vision transformer](https://arxiv.org/abs/2211.12311) | Haiming Yao, Xue Wang | CNN; ViT | Anomaly Detection | Industrial vision | arXiv, 2022 | Hybrid; Reconstruction-based |
| [VT-ADL: A vision transformer network for image anomaly detection and localization](https://ieeexplore.ieee.org/abstract/document/9576231) | Pankaj Mishra, Riccardo Verk, Daniele Fornasier, Claudio Piciarelli, Gian Luca Foresti | ViT | Anomaly Detection | Vision | ISIE, 2021 | Reconstruction-based; Patch embedding |
| [Dinomaly: The less is more philosophy in multi-class unsupervised anomaly detection](https://arxiv.org/abs/2405.14325) | Jia Guo, Shuai Lu, Weihang Zhang, Huiqi Li | ViT | Anomaly Detection | Vision | arXiv, 2024 | Reconstruction-based |
| [Rethinking Out-of-Distribution (OOD) Detection: Masked Image Modeling Is All You Need](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Rethinking_Out-of-Distribution_OOD_Detection_Masked_Image_Modeling_Is_All_You_CVPR_2023_paper.pdf) | Li et al. | ViT | OOD Detection | Vision | CVPR, 2023 | Reconstruction-based pretext tasks |
| [Anomaly detection with conditioned denoising diffusion models](https://arxiv.org/abs/2305.15956) | Arian Mousakhan, Thomas Brox, Jawad Tayyub | - | Anomaly Detection | Vision | arXiv, 2023 | Reconstruction-based; Diffusion |
| [Unsupervised out-of-distribution detection with diffusion inpainting](https://openreview.net/forum?id=HiX1ybkFMl) | Zhenzhen Liu, Jin Peng Zhou, Yufan Wang, Kilian Q. Weinberger | - | OOD Detection | Vision | ICML, 2023 | Reconstruction-based; Diffusion |

## Memory-bank-based
| Paper | Authors | Backbone Model | Task Category | Dataset Type | Venue | Approach |
| --- | --- | --- | --- | --- | --- | --- |
| [AnomalyDINO: Boosting patch-based few-shot anomaly detection with DINOv2](https://arxiv.org/abs/2405.14529) | Simon Damm, Mike Laszkiewicz, Johannes Lederer, Asja Fischer | DINOv2 | Anomaly Detection | Vision | arXiv, 2024 | Memory-bank-based |

## Pretraining
| Paper | Authors | Backbone Model | Task Category | Dataset Type | Venue | Approach |
| --- | --- | --- | --- | --- | --- | --- |
| [How to train your ViT for OOD Detection](https://arxiv.org/abs/2405.17447) | Maximilian Mueller, Matthias Hein | ViT | OOD Detection | Vision | arXiv, 2024 | Pretraining |
| [Can pre-trained networks detect familiar out-of-distribution data?](https://arxiv.org/abs/2311.12076) | Atsuyuki Miyai, Qing Yu, Go Irie, Kiyoharu Aizawa | ViT-S; ResNet-50 | OOD Detection | Vision | arXiv, 2023 | Pretraining |

## Fine Tuning
| Paper | Authors | Backbone Model | Task Category | Dataset Type | Venue | Approach |
| --- | --- | --- | --- | --- | --- | --- |
| [Exploring the Limits of Out-of-Distribution Detection](https://arxiv.org/pdf/2106.03004) | Stanislav Fort, Jie Ren, Balaji Lakshminarayanan | ViT | OOD Detection | Vision | NeurIPS, 2021 | Fine Tuning |
| [Towards few-shot out-of-distribution detection](https://arxiv.org/abs/2311.12076) | Jiuqing Dong, Yongbin Gao, Heng Zhou, Jun Cen, Yifan Yao, Sook Yoon, Park Dong Sun | ViT | Few-shot OOD Detection | Vision | arXiv, 2023 | Fine Tuning |

## Similarity score-based
| Paper | Authors | Backbone Model | Task Category | Dataset Type | Venue | Approach |
| --- | --- | --- | --- | --- | --- | --- |
| [OODformer: Out-Of-Distribution Detection Transformer](https://arxiv.org/pdf/2107.08976) | Rajat Koner, Poulami Sinhamahapatra, Karsten Roscher, Stephan Günnemann, Volker Tresp | ViT | OOD Detection | Vision | arXiv, 2021 | Similarity score-based |

## Diffusion
| Paper | Authors | Backbone Model | Task Category | Dataset Type | Venue | Approach |
| --- | --- | --- | --- | --- | --- | --- |
| [AnoDDPM: Anomaly detection with denoising diffusion probabilistic models using simplex noise](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Wyatt_AnoDDPM_Anomaly_Detection_With_Denoising_Diffusion_Probabilistic_Models_Using_Simplex_CVPRW_2022_paper.html) | Julian Wyatt, Adam Leach, Sebastian M. Schmon, Chris G. Willcocks | DDPMs | Anomaly Detection | Vision | CVPR, 2022 | Diffusion |
| [Diffusion models for medical anomaly detection](https://link.springer.com/chapter/10.1007/978-3-031-16452-1_4) | Julia Wolleb, Florentin Bieder, Robin Sandkühler, Philippe C. Cattin | DDPMs | Anomaly Detection | Medical Images | MICCAI, 2022 | Diffusion |
| [DiffusionAD: Denoising diffusion for anomaly detection](https://arxiv.org/abs/2303.08730) | Hui Zhang, Zheng Wang, Zuxuan Wu, Yugang Jiang | - | Anomaly Detection | Vision | arXiv, 2023 | Diffusion |
| [On diffusion modeling for anomaly detection](https://openreview.net/forum?id=lR3rk7ysXz) | Victor Livernoche, Vineet Jain, Yashar Hezaveh, Siamak Ravanbakhsh | DDPMs | Anomaly Detection | Tabular; Image; Text | ICLR, 2024 | Diffusion |
| [Mask, stitch, and re-sample: Enhancing robustness and generalizability in anomaly detection through automatic diffusion models](https://arxiv.org/abs/2305.19643) | Cosmin I. Bercea, Michael Neumayr, Daniel Rueckert, Julia A. Schnabel | DDPMs | Anomaly Detection | Vision | arXiv, 2023 | Diffusion |
| [Unsupervised anomaly detection in medical images using masked diffusion model](https://arxiv.org/abs/2305.19867) | Hasan Iqbal, Umar Khalid, Jing Hua, Chen Chen | DDPMs | Anomaly Detection | Medical | arXiv, 2023 | Diffusion |
| [LafitE: Latent diffusion model with feature editing for unsupervised multi-class anomaly detection](https://arxiv.org/abs/2307.08059) | Haonan Yin, Guanlong Jiao, Qianhui Wu, Borje F. Karlsson, Biqing Huang, Chin Yew Lin | LDM | Anomaly Detection | Vision | arXiv, 2023 | Diffusion |
| [Diffusion models for counterfactual generation and anomaly detection in brain images](https://arxiv.org/abs/2308.02062) | Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, Amos Storkey | DDPMs; DDIMs | Anomaly Detection | Medical Images| arXiv, 2023 | Diffusion |
| [Unsupervised surface anomaly detection with diffusion probabilistic model](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Unsupervised_Surface_Anomaly_Detection_with_Diffusion_Probabilistic_Model_ICCV_2023_paper.pdf) | Matic Fučka, Vitjan Zavrtanik, Danijel Skočaj | LDM | Anomaly Detection | Vision | ICCV, 2023 | Diffusion |
| [Transfusion -- A transparency-based diffusion model for anomaly detection](https://arxiv.org/abs/2311.09999) | Ziyun Liang, Harry Anthony, Felix Wagner, Konstantinos Kamnitsas | - | Anomaly Detection | Vision | arXiv, 2023 | Diffusion |
| [Unsupervised anomaly detection using aggregated normative diffusion](https://arxiv.org/abs/2312.01904) | Alexander Frotscher, Jaivardhan Kapoor, Thomas Wolfers, Christian F. Baumgartner | - | Anomaly Detection | Vision | arXiv, 2023 | Diffusion |
| [Adversarial denoising diffusion model for unsupervised anomaly detection](https://arxiv.org/abs/2312.04382) | Jongmin Yu, Hyeontaek Oh, Jinhong Yang | - | Anomaly Detection | Vision | arXiv, 2023 | Diffusion |
| [Guided reconstruction with conditioned diffusion models for unsupervised anomaly detection in brain MRIs](https://arxiv.org/abs/2312.04215) | Finn Behrendt, Debayan Bhattacharya, Robin Mieling, Lennart Maack, Julia Krüger, Roland Opfer, Alexander Schlaefer | - | Anomaly Detection | Medical | arXiv, 2023 | Diffusion; Reconstruction-based |
| [DiAD: A diffusion-based framework for multi-class anomaly detection](https://arxiv.org/abs/2312.06607) | Haoyang He, Jiangning Zhang, Hongxu Chen, Xuhai Chen, Zhishan Li, Xu Chen, Yabiao Wang, Chengjie Wang, Lei Xie | - | Anomaly Detection | Vision | arXiv, 2023 | Diffusion |
| [Feature prediction diffusion model for video anomaly detection](https://openaccess.thecvf.com/content/ICCV2023/html/Yan_Feature_Prediction_Diffusion_Model_for_Video_Anomaly_Detection_ICCV_2023_paper.html) | Cheng Yan, Shiyu Zhang, Yang Liu, Guansong Pang, Wenjun Wang | - | Anomaly Detection | Video | ICCV, 2023 | Diffusion |
| [Removing anomalies as noises for industrial defect localization](https://openaccess.thecvf.com/content/ICCV2023/html/Lu_Removing_Anomalies_as_Noises_for_Industrial_Defect_Localization_ICCV_2023_paper.html) | Fanbin Lu, Xufeng Yao, Chi-Wing Fu, Jiaya Jia | - | Anomaly Detection | Industrial | ICCV, 2023 | Diffusion |
| [D3AD: Dynamic denoising diffusion probabilistic model for anomaly detection](https://arxiv.org/abs/2401.04463) | Justin Tebbe, Jawad Tayyub | DDPMs | Anomaly Detection | Vision | arXiv, 2024 | Diffusion |
| [TauAD: MRI-free Tau anomaly detection in PET imaging via conditioned diffusion models](https://arxiv.org/abs/2405.13199) | Lujia Zhong, Shuo Huang, Jiaxin Yue, Jianwei Zhang, Zhiwei Deng, Wenhao Chi, Yonggang Shi | - | Anomaly Detection | Medical Images| arXiv, 2024 | Diffusion |
| [DIFFGUARD: Semantic Mismatch-Guided Out-of-Distribution Detection using Pre-trained Diffusion Models](https://openaccess.thecvf.com/content/ICCV2023/papers/Gao_DIFFGUARD_Semantic_Mismatch-Guided_Out-of-Distribution_Detection_Using_Pre-Trained_Diffusion_Models_ICCV_2023_paper.pdf) | Gao et al. | Conditional DM | OOD Detection | Vision | ICCV, 2023 | Diffusion |
| [Deep Feature Deblurring Diffusion for Detecting Out-of-Distribution Objects](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Deep_Feature_Deblurring_Diffusion_for_Detecting_Out-of-Distribution_Objects_ICCV_2023_paper.pdf) | Wu et al. | ResNet-50 | OOD Detection | Vision | ICCV, 2023 | Diffusion |


### Multi-modal Foundation Models

### Prompting
| Paper | Authors | Backbone Model | Task Category | Dataset Type | Venue | Approach |
|------------------------|--------------|----------------|---------------|------------------|---------|---------|
| [WinCLIP: Zero-/few-shot anomaly classification and segmentation](https://arxiv.org/pdf/2303.14814) | Jongheon Jeong, Yang Zou, Taewan Kim, Dongqing Zhang, Avinash Ravichandran, Onkar Dabeer | CLIP | Anomaly Detection | Vision | CVPR, 2023 | Prompting |
| [AnomalyCLIP: Object-agnostic prompt learning for zero-shot anomaly detection](https://openreview.net/forum?id=buC4E91xZE) | Qihang Zhou, Guansong Pang, Yu Tian, Shibo He, Jiming Chen | CLIP | Anomaly Detection | Vision | ICLR, 2024 | Prompting |
| [CLIP-AD: A language-guided staged dual-path model for zero-shot anomaly detection](https://arxiv.org/abs/2311.00453) | Xuhai Chen, Jiangning Zhang, Guanzhong Tian, Haoyang He, Wuhao Zhang, Yabiao Wang, Chengjie Wang, Yunsheng Wu, Yong Liu | CLIP | Anomaly Detection | Vision | arXiv, 2023 | Prompting |
| [Exploring grounding potential of VQA-oriented GPT-4V for zero-shot anomaly detection](https://arxiv.org/abs/2311.02612) | Jiangning Zhang, Xuhai Chen, Zhucun Xue, Yabiao Wang, Chengjie Wang, Yong Liu | GPT-4V | Anomaly Detection | Vision | arXiv, 2023 | Prompting |
| [Toward generalist anomaly detection via in-context residual learning with few-shot sample prompts](https://arxiv.org/abs/2403.06495) | Jiawen Zhu, Guansong Pang | CLIP | Anomaly Detection | Vision | CVPR, 2024 | Prompting |
| [PromptAD: Learning prompts with only normal samples for few-shot anomaly detection](https://arxiv.org/abs/2404.05231) | Xiaofan Li, Zhizhong Zhang, Xin Tan, Chengwei Chen, Yanyun Qu, Yuan Xie, Lizhuang Ma | CLIP | Anomaly Detection | Vision | CVPR, 2024 | Prompting |
| [Text prompt with normality guidance for weakly supervised video anomaly detection](https://arxiv.org/abs/2404.08531) | Zhiwei Yang, Jing Liu, Peng Wu | CLIP | Anomaly Detection | Video | arXiv, 2024 | Prompting; Aligning |
| [LoCoOp: Few-Shot Out-of-Distribution Detection via Prompt Learning](https://arxiv.org/pdf/2306.01293) | Miyai et al. | CLIP | OOD Detection | Vision | NeurIPS, 2023 | Prompting |
| [CLIPN for Zero-Shot OOD Detection: Teaching CLIP to Say No](https://arxiv.org/pdf/2308.12213) | Wang et al. | CLIP | OOD Detection | Vision | ICCV, 2023 | Prompting |
| [How Does Fine-Tuning Impact Out-of-Distribution Detection for Vision-Language Models?](https://arxiv.org/pdf/2306.06048) | Yifei Ming, Yixuan Li | CLIP | OOD Detection | Vision | IJCV, 2023 | Prompting |
| [Out-Of-Distribution Detection With Negative Prompts](https://openreview.net/pdf?id=nanyAujl6e) | Nie et al. | CLIP | OOD Detection | Vision | ICLR, 2024 | Prompting |
| [ID-like Prompt Learning for Few-Shot Out-of-Distribution Detection](https://arxiv.org/pdf/2311.15243) | Yichen Bai, Zongbo Han, Changqing Zhang, Bing Cao, Xiaoheng Jiang, Qinghua Hu | CLIP | OOD Detection | Vision | CVPR, 2024 | Prompting |
| [Learning transferable negative prompts for out-of-distribution detection](https://arxiv.org/abs/2404.03248) | Tianqi Li, Guansong Pang, Xiao Bai, Wenjun Miao, Jin Zheng | CLIP | OOD Detection | Vision | CVPR, 2024 | Prompting |
| [Tagfog: Textual anchor guidance and fake outlier generation for visual out-of-distribution detection](https://ojs.aaai.org/index.php/AAAI/article/view/27871) | Jiankang Chen, Tong Zhang, Weishi Zheng, Ruixuan Wang | CLIP; ChatGPT | OOD Detection | Vision | AAAI, 2024 | Prompting |
| [Exploring grounding potential of VQA-oriented GPT-4V for zero-shot anomaly detection](https://arxiv.org/abs/2311.02612) | Jiangning Zhang, Xuhai Chen, Zhucun Xue, Yabiao Wang, Chengjie Wang, Yong Liu | GPT-4V | Anomaly Detection | Video | arXiv, 2023 | Prompting |
| [Envisioning outlier exposure by large language models for out-of-distribution detection](https://arxiv.org/abs/2405.15370) | Chentao Cao, Zhun Zhong, Zhanke Zhou, Yang Liu, Tongliang Liu, Bo Han | CLIP | OOD Detection | Vision | ICML, 2024 | Prompting |


### Aligning
| Paper | Authors | Backbone Model | Task Category | Dataset Type | Venue | Approach |
|------------------------|--------------|----------------|---------------|------------------|---------|---------|
| [Delving into Out-of-Distribution Detection with Vision-Language Representations](https://arxiv.org/pdf/2211.13445.pdf) | Ming et al. | CLIP | OOD Detection | Vision | NeurIPS, 2022 | Aligning |
| [Text prompt with normality guidance for weakly supervised video anomaly detection](https://arxiv.org/abs/2404.08531) | Zhiwei Yang, Jing Liu, Peng Wu | CLIP | Anomaly Detection | Video | arXiv, 2024 | Prompting; Aligning |
| [Negative Label Guided OOD Detection with Pretrained Vision-Language Models](https://arxiv.org/pdf/2403.20078) | Xue Jiang, Feng Liu, Zhen Fang, Hong Chen, Tongliang Liu, Feng Zheng, Bo Han | CLIP; ALIGN; GroupViT; AltCLIP | OOD Detection | Vision | ICLR, 2024 | Aligning |
| [Zero-shot out-of-distribution detection based on the pretrained model CLIP](https://arxiv.org/pdf/2109.02748) | Sepideh Esmaeilpour, Bing Liu, Eric Robertson, Lei Shu | CLIP | OOD Detection | Vision | AAAI, 2022 | Aligning |
| [Exploring large language models for multi-modal out-of-distribution detection](https://arxiv.org/abs/2310.08027) | Yi Dai, Hao Lang, Kaisheng Zeng, Fei Huang, Yongbin Li | text-davinci-003; CLIP | OOD Detection | Vision | EMNLP, 2023 | Aligning |

### Adapting
| Paper | Authors | Backbone Model | Task Category | Dataset Type | Venue | Approach |
|------------------------|--------------|----------------|---------------|------------------|---------|---------|
| [Adapting visual-language models for generalizable anomaly detection in medical images](https://arxiv.org/abs/2403.12570v1) | Chaoqin Huang, Aofan Jiang, Jinghao Feng, Ya Zhang, Xinchao Wang, Yanfeng Wang | CLIP | Anomaly Detection | Medical Images | CVPR, 2024 | Adapter-based |
| [Adapting Contrastive Language-Image Pretrained (CLIP) Models for Out-of-Distribution Detection](https://openreview.net/pdf?id=YCgX7sJRF1) | Nikolas Adaloglou, Felix Michels, Tim Kaiser, Markus Kollmann | CLIP | OOD Detection | Vision | TMLR, 2024 | Adapter-based |
| [Video anomaly detection and explanation via large language models](https://arxiv.org/abs/2401.05702) | Hui Lv, Qianru Sun | Video-LLaMA | Anomaly Detection | Video | arXiv, 2024 | Adapter-based |

### Knowledge Distillation
| Paper | Authors | Backbone Model | Task Category | Dataset Type | Venue | Approach |
|------------------------|--------------|----------------|---------------|------------------|---------|---------|
| [Distilling out-of-distribution robustness from vision-language foundation models](https://neurips.cc/virtual/2023/poster/70716) | Andy Zhou, Jindong Wang, Yuxiong Wang, Haohan Wang | CLIP | OOD Detection | Vision | NeurIPS, 2023 | Knowledge Distillation |

         
