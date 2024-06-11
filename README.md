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

| Paper | Authors | Backbone Model | Task Category | Dataset Type| Venue | Approach | Summary |
|------------------------|--------------|----------------|---------------|------------------|---------|---------|---------|
| [Semantic anomaly detection with large language models](https://arxiv.org/abs/2305.11307) | Amine Elhafsi, Rohan Sinha, Christopher Agia, Edward Schmerling, Issa Nesnas, Marco Pavone | text-davinci-003 | Semantic Anomaly Detection | Vision | arXiv, 2023 | Few-shot Prompting; Chain-of-thought prompting | - |
| [LogGPT: Exploring ChatGPT for log-based anomaly detection](https://arxiv.org/abs/2309.01189) | Jiaxing Qi, Shaohan Huang, Zhongzhi Luan, Carol Fung, Hailong Yang, Depei Qian | ChatGPT | Log-based Anomaly Detection | Log Data | arXiv, 2023 | Few-shot Prompting | - |
| [Exploring large language models for multi-modal out-of-distribution detection](https://arxiv.org/abs/2310.08027) | Yi Dai, Hao Lang, Kaisheng Zeng, Fei Huang, Yongbin Li | text-davinci-003; CLIP | OOD Detection | Multi-modal; Vision | EMNLP, 2023 | Zero-shot Prompting | - |
| [Overlooked video classification in weakly supervised video anomaly detection](https://openaccess.thecvf.com/content/WACV2024W/RWS/html/Tan_Overlooked_Video_Classification_in_Weakly_Supervised_Video_Anomaly_Detection_WACVW_2024_paper.html) | Weijun Tan, Qi Yao, Jingfeng Liu | - | Anomaly Detection | Video | WACV, 2024 | - | - |
| [Video anomaly detection and explanation via large language models](https://arxiv.org/abs/2401.05702) | Hui Lv, Qianru Sun | Video-LLaMA | Anomaly Detection, Explanation | Video | arXiv, 2024 | Hybrid | - |
| [Large language model guided knowledge distillation for time series anomaly detection](https://arxiv.org/abs/2401.15123v1) | Chen Liu, Shibo He, Qihang Zhou, Shizhong Li, Wenchao Meng | - | Anomaly Detection, Knowledge Distillation | Time Series | arXiv, 2024 | - | - |
| [Harnessing large language models for training-free video anomaly detection](https://arxiv.org/abs/2404.01014) | Luca Zanella, Willi Menapace, Massimiliano Mancini, Yiming Wang, Elisa Ricci | - | Anomaly Detection | Video | CVPR, 2024 | - | - |
| [Your finetuned large language model is already a powerful out-of-distribution detector](https://arxiv.org/abs/2404.08679) | Andi Zhang, Tim Z. Xiao, Weiyang Liu, Robert Bamler, Damon Wischik | Llama; Mistral | OOD Detection | Text | arXiv, 2024 | Fine Tuning; LLM-based likelihood ratio | - |
| [Do LLMs understand visual anomalies? Uncovering LLM capabilities in zero-shot anomaly detection](https://arxiv.org/abs/2404.09654) | Jiaqi Zhu, Shaofeng Cai, Fang Deng, Junran Wu | - | Anomaly Detection | Visual Data | arXiv, 2024 | - | - |
| [FiLo: Zero-shot anomaly detection by fine-grained description and high-quality localization](https://arxiv.org/abs/2404.08531) | Zhaopeng Gu, Bingke Zhu, Guibo Zhu, Yingying Chen, Hao Li, Ming Tang, Jinqiao Wang | - | Anomaly Detection | - | arXiv, 2024 | - | - |
| [Large language models can deliver accurate and interpretable time series anomaly detection](https://arxiv.org/abs/2405.15370) | Jiaqi Tang, Hao Lu, Ruizheng Wu, Xiaogang Xu, Ke Ma, Cheng Fang, Bin Guo, Jiangbo Lu, Qifeng Chen, Ying-Cong Chen | - | Anomaly Detection | Time Series | arXiv, 2024 | - | - |
| [Hawk: Learning to understand open-world video anomalies](https://arxiv.org/abs/2405.15370) | Jun Liu, Chaoyun Zhang, Jiaxu Qian, Minghua Ma, Si Qin, Chetan Bansal, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang, Yixin Liu, Shiyuan Li, Yu Zheng, Qingfeng Chen, Chengqi Zhang, Shirui Pan | - | Anomaly Detection | Video | arXiv, 2024 | - | - |
| [Large language models can be zero-shot anomaly detectors for time series?](https://arxiv.org/abs/2405.14755) | Sarah Alnegheimish, Linh Nguyen, Laure Berti-Equille, Kalyan Veeramachaneni | - | Anomaly Detection | Time Series | KDD, 2024 | - | - |
| [Out-of-distribution detection and selective generation for conditional language models](https://arxiv.org/pdf/2209.15558) | Jie Ren, Jiaming Luo, Yao Zhao, Kundan Krishna, Mohammad Saleh, Balaji Lakshminarayanan, Peter J. Liu | - | OOD Detection | - | ICLR, 2023 | - | - |
| [On the robustness of ChatGPT: An adversarial and out-of-distribution perspective](https://arxiv.org/abs/2302.12095) | Jindong Wang, Xixu Hu, Wenxin Hou, Hao Chen, Runkai Zheng, Yidong Wang, Linyi Yang, Haojun Huang, Wei Ye, Xiubo Geng, Binxin Jiao, Yue Zhang, Xing Xie | ChatGPT | OOD Detection, Adversarial | - | arXiv, 2023 | - | - |
| [How Good Are LLMs at Out-of-Distribution Detection?](https://arxiv.org/pdf/2308.10261) | Bo Liu, Liming Zhan, Zexin Lu, Yujie Feng, Lei Xue, Xiao-Ming Wu | - | OOD Detection | - | COLING, 2024 | - | - |
| [Beyond the Known: Investigating LLMs Performance on Out-of-Domain Intent Detection](https://arxiv.org/pdf/2402.17256) | Pei Wang, Keqing He, Yejie Wang, Xiaoshuai Song, Yutao Mou, Jingang Wang, Yunsen Xian, Xunliang Cai, Weiran Xu | - | OOD Detection | - | COLING, 2024 | - | - |
| [VI-OOD: A Unified Representation Learning Framework for Textual Out-of-distribution Detection](https://arxiv.org/pdf/2404.06217) | Li-Ming Zhan, Bo Liu, Xiao-Ming Wu | - | OOD Detection | Text | arXiv, 2024 | - | - |
| [Exploring Large Language Models for Multi-Modal Out-of-Distribution Detection](https://arxiv.org/pdf/2310.08027) | Yi Dai, Hao Lang, Kaisheng Zeng, Fei Huang, Yongbin Li | - | OOD Detection | Multi-Modal | EMNLP, 2023 | - | - |


1. **Semantic anomaly detection with large language models.** arXiv, 2023. [paper](https://arxiv.org/abs/2305.11307)

   *Amine Elhafsi, Rohan Sinha, Christopher Agia, Edward Schmerling, Issa Nesnas, and Marco Pavone.*

1. **LogGPT: Exploring ChatGPT for log-based anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2309.01189)

   *Jiaxing Qi, Shaohan Huang, Zhongzhi Luan, Carol Fung, Hailong Yang, and Depei Qian.*

1. **HuntGPT: Integrating machine learning-based anomaly detection and explainable AI with large language models (LLMs).** arXiv, 2023. [paper](https://arxiv.org/abs/2309.16021)

   *Tarek Ali and Panos Kostakos.*

1. **Graph neural architecture search with GPT-4.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.01436)

   *Haishuai Wang, Yang Gao, Xin Zheng, Peng Zhang, Hongyang Chen, and Jiajun Bu.* 

1. **Exploring large language models for multi-modal out-of-distribution detection.** EMNLP, 2023. [paper](https://arxiv.org/abs/2310.08027)

   *Yi Dai, Hao Lang, Kaisheng Zeng, Fei Huang, and Yongbin Li.*

1. **Detecting pretraining data from large language models.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.16789)

   *Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, and Luke Zettlemoyer.*

1. **Weakly supervised detection of gallucinations in LLM activations.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.02798)

   *Miriam Rateike, Celia Cintas, John Wamburu, Tanya Akumu, and Skyler Speakman.* 

1. **overlooked video classification in weakly supervised video anomaly detection.** WACV, 2024. [paper](https://openaccess.thecvf.com/content/WACV2024W/RWS/html/Tan_Overlooked_Video_Classification_in_Weakly_Supervised_Video_Anomaly_Detection_WACVW_2024_paper.html)

   *Weijun Tan, Qi Yao, and Jingfeng Liu.* 

1. **Video anomaly detection and explanation via large language models.** arXiv, 2024. [paper](https://arxiv.org/abs/2401.05702)

   *Hui Lv and Qianru Sun.*

1. **Large language model guided knowledge distillation for time series anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2401.15123v1)

   *Chen Liu, Shibo He, Qihang Zhou, Shizhong Li, and Wenchao Meng.*

1. **Harnessing large language models for training-free video anomaly detection.** CVPR, 2024. [paper](https://arxiv.org/abs/2404.01014)

   *Luca Zanella, Willi Menapace, Massimiliano Mancini, Yiming Wang, and Elisa Ricci.*

1. **Your finetuned large language model is already a powerful out-of-distribution detector.** arXiv, 2024. [paper](https://arxiv.org/abs/2404.08679)

   *Andi Zhang, Tim Z. Xiao, Weiyang Liu, Robert Bamler, and Damon Wischik.*

1. **Do LLMs understand visual anomalies? Uncovering LLM capabilities in zero-shot anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2404.09654)

   *Jiaqi Zhu, Shaofeng Cai, Fang Deng, and Junran Wu.*

1. **FiLo: Zero-shot anomaly detection by fine-grained description and high-quality localization.** arXiv, 2024. [paper](https://arxiv.org/abs/2404.08531)

   *Zhaopeng Gu, Bingke Zhu, Guibo Zhu, Yingying Chen, Hao Li, Ming Tang, and Jinqiao Wang.*

1. **Large language models can deliver accurate and interpretable time series anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.15370)

   *Jiaqi Tang, Hao Lu, Ruizheng Wu, Xiaogang Xu, Ke Ma, Cheng Fang, Bin Guo, Jiangbo Lu, Qifeng Chen, and Ying-Cong Chen.*

1. **Hawk: Learning to understand open-world video anomalies.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.15370)

   *Jun Liu, Chaoyun Zhang, Jiaxu Qian, Minghua Ma, Si Qin, Chetan Bansal, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang.* 

   *Yixin Liu, Shiyuan Li, Yu Zheng, Qingfeng Chen, Chengqi Zhang, and Shirui Pan.*

1. **Large language models can be zero-shot anomaly detectors for time series?** KDD, 2024. [paper](https://arxiv.org/abs/2405.14755)

   *Sarah Alnegheimish, Linh Nguyen, Laure Berti-Equille, and Kalyan Veeramachaneni.*

(ICLR 2023) [Out-of-distribution detection and selective generation for conditional language models](https://arxiv.org/pdf/2209.15558) by Jie Ren, Jiaming Luo, Yao Zhao, Kundan Krishna, Mohammad Saleh, Balaji Lakshminarayanan, Peter J. Liu

(arXiv 2023) [On the robustness of ChatGPT: An adversarial and out-of-distribution perspective](https://arxiv.org/abs/2302.12095) by Jindong Wang, Xixu Hu, Wenxin Hou, Hao Chen, Runkai Zheng, Yidong Wang, Linyi Yang, Haojun Huang, Wei Ye, Xiubo Geng, Binxin Jiao, Yue Zhang, Xing Xie

(ACL 2023) [Multi-Level Knowledge Distillation for Out-of-Distribution Detection in Text](https://arxiv.org/pdf/2211.11300) by Qianhui Wu, Huiqiang Jiang, Haonan Yin, Börje F. Karlsson, Chin-Yew Lin

(COLING 2024) [How Good Are LLMs at Out-of-Distribution Detection?](https://arxiv.org/pdf/2308.10261) by Bo Liu, Liming Zhan, Zexin Lu, Yujie Feng, Lei Xue, Xiao-Ming Wu

(COLING 2024) [Beyond the Known: Investigating LLMs Performance on Out-of-Domain Intent Detection](https://arxiv.org/pdf/2402.17256) by Pei Wang, Keqing He, Yejie Wang, Xiaoshuai Song, Yutao Mou, Jingang Wang, Yunsen Xian, Xunliang Cai, Weiran Xu

(arXiv 2024) [VI-OOD: A Unified Representation Learning Framework for Textual Out-of-distribution Detection](https://arxiv.org/pdf/2404.06217) by Li-Ming Zhan, Bo Liu, Xiao-Ming Wu

(EMNLP 2023) [Exploring Large Language Models for Multi-Modal Out-of-Distribution Detection](https://arxiv.org/pdf/2310.08027) by Yi Dai, Hao Lang, Kaisheng Zeng, Fei Huang, Yongbin Li
      
   
### Vision Foundation Models

1. **How to train your ViT for OOD Detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.17447)

   *Maximilian Mueller and Matthias Hein.*

1. **Exploring plain ViT reconstruction for multi-class unsupervised anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.07495)

   *Jiangning Zhang, Xuhai Chen, Yabiao Wang, Chengjie Wang, Yong Liu, Xiangtai Li, Ming-Hsuan Yang, and Dacheng Tao.*

1. **Multi-contextual predictions with vision transformer for video anomaly detection.** arXiv, 2022. [paper](https://arxiv.org/abs/2206.08568?context=cs)

   *Joo-Yeon Lee, Woo-Jeoung Nam, and Seong-Whan Lee.* 

1. **Generalizable industrial visual anomaly detection with self-induction vision transformer.** arXiv, 2022. [paper](https://arxiv.org/abs/2211.12311)

   *Haiming Yao and Xue Wang.* 

1. **VT-ADL: A vision transformer network for image anomaly detection and localization.** ISIE, 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9576231)

   *Pankaj Mishra, Riccardo Verk, Daniele Fornasier, Claudio Piciarelli, and Gian Luca Foresti.*

1. **Dinomaly: The less is more philosophy in multi-class unsupervised anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.14325)

   *Jia Guo, Shuai Lu, Weihang Zhang, and Huiqi Li.*
   
(NeurIPS 2021) [Exploring the Limits of Out-of-Distribution Detection](https://arxiv.org/pdf/2106.03004) by Stanislav Fort, Jie Ren, Balaji Lakshminarayanan

(arXiv 2021) [OODformer: Out-Of-Distribution Detection Transformer](https://arxiv.org/pdf/2107.08976) by Rajat Koner, Poulami Sinhamahapatra, Karsten Roscher, Stephan Günnemann, Volker Tresp

(arXiv 2023) [Can pre-trained networks detect familiar out-of-distribution data?]([https://arxiv.org/abs/2311.12076](https://arxiv.org/abs/2310.00847) by Atsuyuki Miyai, Qing Yu, Go Irie, and Kiyoharu Aizawa

(arXiv 2023) [Towards few-shot out-of-distribution detection](https://arxiv.org/abs/2311.12076) by Jiuqing Dong, Yongbin Gao, Heng Zhou, Jun Cen, Yifan Yao, Sook Yoon, Park Dong Sun

(CVPR 2023) [Rethinking Out-of-Distribution (OOD) Detection: Masked Image Modeling Is All You Need](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Rethinking_Out-of-Distribution_OOD_Detection_Masked_Image_Modeling_Is_All_You_CVPR_2023_paper.pdf) [[Code]](https://github.com/JulietLJY/MOOD) by Li et al.

(ICLR 2024 Workshop) [How to train your ViT for OOD Detection](https://arxiv.org/pdf/2405.17447) by Maximilian Mueller, Matthias Hein

#### Diffusion Model
1. **AnoDDPM: Anomaly detection with denoising diffusion probabilistic models using simplex noise.** CVPR, 2022. [paper](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Wyatt_AnoDDPM_Anomaly_Detection_With_Denoising_Diffusion_Probabilistic_Models_Using_Simplex_CVPRW_2022_paper.html)

   *Julian Wyatt, Adam Leach, Sebastian M. Schmon, and Chris G. Willcocks.* 

1. **Diffusion models for medical anomaly detection.** MICCAI, 2022. [paper](https://link.springer.com/chapter/10.1007/978-3-031-16452-1_4)

   *Julia Wolleb, Florentin Bieder, Robin Sandkühler, and Philippe C. Cattin.* 

1. **DiffusionAD: Denoising diffusion for anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2303.08730)

   *Hui Zhang, Zheng Wang, Zuxuan Wu, Yugang Jiang.* 

1. **Anomaly detection with conditioned denoising diffusion models.** arXiv, 2023. [paper](https://arxiv.org/abs/2305.15956)

   *Arian Mousakhan, Thomas Brox, and Jawad Tayyub.* 

1. **Unsupervised out-of-distribution detection with diffusion inpainting.** ICML, 2023. [paper](https://openreview.net/forum?id=HiX1ybkFMl)

   *Zhenzhen Liu, Jin Peng Zhou, Yufan Wang, and Kilian Q. Weinberger.* 

1. **On diffusion modeling for anomaly detection.** ICLR, 2024. [paper](https://openreview.net/forum?id=lR3rk7ysXz)

   *Victor Livernoche, Vineet Jain, Yashar Hezaveh, and Siamak Ravanbakhsh.* 

1. **Mask, stitch, and re-sample: Enhancing robustness and generalizability in anomaly detection through automatic diffusion models.** arXiv, 2023. [paper](https://arxiv.org/abs/2305.19643)

   *Cosmin I. Bercea, Michael Neumayr, Daniel Rueckert, and Julia A. Schnabel.* 

1. **Unsupervised anomaly detection in medical images using masked diffusion model.** arXiv, 2023. [paper](https://arxiv.org/abs/2305.19867)

   *Hasan Iqbal, Umar Khalid, Jing Hua, and Chen Chen.* 

1. **Unsupervised anomaly detection in medical images using masked diffusion model.** arXiv, 2023. [paper](https://arxiv.org/abs/2305.19867)

   *Hasan Iqbal, Umar Khalid, Jing Hua, and Chen Chen.* 

1. **ImDiffusion: Imputed diffusion models for multivariate time series anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.00754)

   *Yuhang Chen, Chaoyun Zhang, Minghua Ma, Yudong Liu, Ruomeng Ding, Bowen Li, Shilin He, Saravan Rajmohan, Qingwei Lin, and Dongmei Zhang.* 

1. **Multimodal motion conditioned diffusion model for skeleton-based video anomaly detection.** ICCV, 2023. [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Flaborea_Multimodal_Motion_Conditioned_Diffusion_Model_for_Skeleton-based_Video_Anomaly_Detection_ICCV_2023_paper.html)

   *Alessandro Flaborea, Luca Collorone, Guido Maria D’Amely di Melendugno, Stefano D’Arrigo, Bardh Prenkaj, and Fabio Galasso.*

1. **LafitE: Latent diffusion model with feature editing for unsupervised multi-class anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.08059)

   *Haonan Yin, Guanlong Jiao, Qianhui Wu, Borje F. Karlsson, Biqing Huang, and Chin Yew Lin.*

1. **Diffusion models for counterfactual generation and anomaly detection in brain images.** arXiv, 2023. [paper](https://arxiv.org/abs/2308.02062)

   *Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, and Amos Storkey.*

1. **Imputation-based time-series anomaly detection with conditional weight-incremental diffusion models.** KDD, 2023. [paper](https://dl.acm.org/doi/10.1145/3580305.3599391)

   *Chunjing Xiao, Zehua Gou, Wenxin Tai, Kunpeng Zhang, and Fan Zhou.*

1. **MadSGM: Multivariate anomaly detection with score-based generative models.** CIKM, 2023. [paper](https://arxiv.org/abs/2308.15069)

   *Haksoo Lim, Sewon Park, Minjung Kim, Jaehoon Lee, Seonkyu Lim, and Noseong Park.*

1. **Modality cycles with masked conditional diffusion for unsupervised anomaly segmentation in MRI.** MICCAI, 2023. [paper](https://arxiv.org/abs/2308.16150)

   *Ziyun Liang, Harry Anthony, Felix Wagner, and Konstantinos Kamnitsas.*

1. **Controlled graph neural networks with denoising diffusion for anomaly detection.** Expert Systems with Applications, 2023. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417423020353)

   *Xuan Li, Chunjing Xiao, Ziliang Feng, Shikang Pang, Wenxin Tai, and Fan Zhou.*

1. **Unsupervised surface anomaly detection with diffusion probabilistic model.** ICCV, 2023. [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Unsupervised_Surface_Anomaly_Detection_with_Diffusion_Probabilistic_Model_ICCV_2023_paper.pdf)

   *Matic Fučka, Vitjan Zavrtanik, and Danijel Skočaj.*

1. **Transfusion -- A transparency-based diffusion model for anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.09999)

   *Ziyun Liang, Harry Anthony, Felix Wagner, and Konstantinos Kamnitsas.*

1. **Unsupervised anomaly detection using aggregated normative diffusion.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.01904)

   *Alexander Frotscher, Jaivardhan Kapoor, Thomas Wolfers, and Christian F. Baumgartner.*

1. **Adversarial denoising diffusion model for unsupervised anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.04382)

   *Jongmin Yu, Hyeontaek Oh, and Jinhong Yang.*

1. **Guided reconstruction with conditioned diffusion models for unsupervised anomaly detection in brain MRIs.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.04215)

   *Finn Behrendt, Debayan Bhattacharya, Robin Mieling, Lennart Maack, Julia Krüger, Roland Opfer, and Alexander Schlaefer.*

1. **DiAD: A diffusion-based framework for multi-class anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.06607)

   *Haoyang He, Jiangning Zhang, Hongxu Chen, Xuhai Chen, Zhishan Li, Xu Chen, Yabiao Wang, Chengjie Wang, and Lei Xie.*

1. **Feature prediction diffusion model for video anomaly detection.** ICCV, 2023. [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Yan_Feature_Prediction_Diffusion_Model_for_Video_Anomaly_Detection_ICCV_2023_paper.html)

   *Cheng Yan, Shiyu Zhang, Yang Liu, Guansong Pang, and Wenjun Wang.*

1. **Removing anomalies as noises for industrial defect localization.** ICCV, 2023. [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Lu_Removing_Anomalies_as_Noises_for_Industrial_Defect_Localization_ICCV_2023_paper.html)

   *Fanbin Lu, Xufeng Yao, Chi-Wing Fu, and Jiaya Jia.*

1. **DATAELIXIR: Purifying poisoned dataset to mitigate backdoor attacks via diffusion models.** AAAI, 2024. [paper](https://arxiv.org/abs/2312.11057)

   *Jiachen Zhou, Peizhuo Lv, Yibing Lan, Guozhu Meng, Kai Chen, and Hualong Ma.*

1. **Controlled graph neural networks with denoising diffusion for anomaly detection.** Expert Systems with Applications, 2024. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417423020353)

   *Xuan Li, Chunjing Xiao, Ziliang Feng, Shikang Pang, Wenxin Tai, and Fan Zhou.*

1. **D3AD: Dynamic denoising diffusion probabilistic model for anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2401.04463)

   *Justin Tebbe and Jawad Tayyub.*

1. **TauAD: MRI-free Tau anomaly detection in PET imaging via conditioned diffusion models.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.13199)

   *Lujia Zhong, Shuo Huang, Jiaxin Yue, Jianwei Zhang, Zhiwei Deng, Wenhao Chi, and Yonggang Shi.*

(NeurIPS 2022) [Delving into Out-of-Distribution Detection with Vision-Language Representations](https://arxiv.org/pdf/2211.13445.pdf) by Ming et al. [[Code]](https://github.com/deeplearning-wisc/MCM) [[Video]](https://www.youtube.com/watch?v=ZZlxBgGalVA)

(NeurIPS 2023) [Dream the Impossible: Outlier Imagination with Diffusion Models](https://arxiv.org/pdf/2309.13415) by Du et al.

(ICCV 2023) [DIFFGUARD: Semantic Mismatch-Guided Out-of-Distribution Detection using Pre-trained Diffusion Models](https://openaccess.thecvf.com/content/ICCV2023/papers/Gao_DIFFGUARD_Semantic_Mismatch-Guided_Out-of-Distribution_Detection_Using_Pre-Trained_Diffusion_Models_ICCV_2023_paper.pdf) by Gao et al. [[Code]](https://github.com/cure-lab/DiffGuard)

(ICCV 2023) [Deep Feature Deblurring Diffusion for Detecting Out-of-Distribution Objects](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Deep_Feature_Deblurring_Diffusion_for_Detecting_Out-of-Distribution_Objects_ICCV_2023_paper.pdf) by Wu et al. [[Code]](https://github.com/AmingWu/DFDD-OOD)

### Multi-modal Foundation Models

| Paper | Authors | Backbone Model | Task Category | Dataset Type| Venue | Approach | Summary |
|------------------------|--------------|----------------|---------------|------------------|---------|---------|---------|
| [WinCLIP: Zero-/few-shot anomaly classification and segmentation](https://arxiv.org/pdf/2303.14814) | Jongheon Jeong, Yang Zou, Taewan Kim, Dongqing Zhang, Avinash Ravichandran, Onkar Dabeer | CLIP | Anomaly Classification | - | CVPR, 2023 | - | - |
| [AnomalyGPT: Detecting industrial anomalies using large vision-language models](https://arxiv.org/abs/2308.15366) | Zhaopeng Gu, Bingke Zhu, Guibo Zhu, Yingying Chen, Ming Tang, Jinqiao Wang | GPT | Anomaly Detection | Industrial Data | arXiv, 2023 | - | - |
| [AnoVL: Adapting vision-language models for unified zero-shot anomaly localization](https://arxiv.org/abs/2308.15939) | Hanqiu Deng, Zhaoxiang Zhang, Jinan Bao, Xingyu Li | - | Anomaly Localization | - | arXiv, 2023 | - | - |
| [CLIPN for zero-shot OOD detection: Teaching CLIP to say no](https://arxiv.org/abs/2308.12213) | Hualiang Wang, Yi Li, Huifeng Yao, Xiaomeng Li | CLIP | OOD Detection | - | ICCV, 2023 | - | - |
| [Semantic scene difference detection in daily life patrolling by mobile robots using pre-trained large-scale vision-language model](https://arxiv.org/abs/2309.16552) | Yoshiki Obinata, Kento Kawaharazuka, Naoaki Kanazawa, Naoya Yamaguchi, Naoto Tsukamoto, Iori Yanokura, Shingo Kitagawa, Koki Shinjo, Kei Okada, Masayuki Inaba | - | Scene Difference Detection | - | IROS, 2023 | - | - |
| [AnomalyCLIP: Object-agnostic prompt learning for zero-shot anomaly detection](https://openreview.net/forum?id=buC4E91xZE) | Qihang Zhou, Guansong Pang, Yu Tian, Shibo He, Jiming Chen | CLIP | Anomaly Detection | - | ICLR, 2024 | - | - |
| [CLIP-AD: A language-guided staged dual-path model for zero-shot anomaly detection](https://arxiv.org/abs/2311.00453) | Xuhai Chen, Jiangning Zhang, Guanzhong Tian, Haoyang He, Wuhao Zhang, Yabiao Wang, Chengjie Wang, Yunsheng Wu, Yong Liu | CLIP | Anomaly Detection | - | arXiv, 2023 | - | - |
| [Exploring grounding potential of VQA-oriented GPT-4V for zero-shot anomaly detection](https://arxiv.org/abs/2311.02612) | Jiangning Zhang, Xuhai Chen, Zhucun Xue, Yabiao Wang, Chengjie Wang, Yong Liu | GPT-4V | Anomaly Detection | - | arXiv, 2023 | - | - |
| [Open-vocabulary video anomaly detection](https://arxiv.org/abs/2311.07042) | Peng Wu, Xuerong Zhou, Guansong Pang, Yujia Sun, Jing Liu, Peng Wang, Yanning Zhang | - | Anomaly Detection | Video | arXiv, 2023 | - | - |
| [Distilling out-of-distribution robustness from vision-language foundation models](https://neurips.cc/virtual/2023/poster/70716) | Andy Zhou, Jindong Wang, Yuxiong Wang, Haohan Wang | - | OOD Detection | - | NeurIPS, 2023 | - | - |
| [How well does GPT-4V(ision) adapt to distribution shifts? A preliminary investigation](https://arxiv.org/abs/2312.07424) | Zhongyi Han, Guanglin Zhou, Rundong He, Jindong Wang, Xing Xie, Tailin Wu, Yilong Yin, Salman Khan, Lina Yao, Tongliang Liu, Kun Zhang | GPT-4V | Distribution Shift | - | arXiv, 2023 | - | - |
| [Toward generalist anomaly detection via in-context residual learning with few-shot sample prompts](https://arxiv.org/abs/2403.06495) | Jiawen Zhu, Guansong Pang | - | Anomaly Detection | - | CVPR, 2024 | - | - |
| [Adapting visual-language models for generalizable anomaly detection in medical images](https://arxiv.org/abs/2403.12570v1) | Chaoqin Huang, Aofan Jiang, Jinghao Feng, Ya Zhang, Xinchao Wang, Yanfeng Wang | - | Anomaly Detection | Medical Images | CVPR, 2024 | - | - |
| [PromptAD: Learning prompts with only normal samples for few-shot anomaly detection](https://arxiv.org/abs/2404.05231) | Xiaofan Li, Zhizhong Zhang, Xin Tan, Chengwei Chen, Yanyun Qu, Yuan Xie, Lizhuang Ma | - | Anomaly Detection | - | CVPR, 2024 | - | - |
| [Text prompt with normality guidance for weakly supervised video anomaly detection](https://arxiv.org/abs/2404.08531) | Zhiwei Yang, Jing Liu, Peng Wu | - | Anomaly Detection | Video | arXiv, 2024 | - | - |
| [AnomalyDINO: Boosting patch-based few-shot anomaly detection with DINOv2](https://arxiv.org/abs/2405.14529) | Simon Damm, Mike Laszkiewicz, Johannes Lederer, Asja Fischer | DINOv2 | Anomaly Detection | - | arXiv, 2024 | - | - |
| [Exploring the Limits of Out-of-Distribution Detection](https://arxiv.org/pdf/2106.03004) | Stanislav Fort, Jie Ren, Balaji Lakshminarayanan | - | OOD Detection | - | NeurIPS, 2021 | - | - |
| [Delving into Out-of-Distribution Detection with Vision-Language Representations](https://arxiv.org/pdf/2211.13445.pdf) | Ming et al. | - | OOD Detection | - | NeurIPS, 2022 | - | - |
| [Zero-shot out-of-distribution detection based on the pretrained model CLIP](https://arxiv.org/pdf/2109.02748) | Sepideh Esmaeilpour, Bing Liu, Eric Robertson, Lei Shu | CLIP | OOD Detection | - | AAAI, 2022 | - | - |
| [LoCoOp: Few-Shot Out-of-Distribution Detection via Prompt Learning](https://arxiv.org/pdf/2306.01293) | Miyai et al. | - | OOD Detection | - | NeurIPS, 2023 | - | - |
| [CLIPN for Zero-Shot OOD Detection: Teaching CLIP to Say No](https://arxiv.org/pdf/2308.12213) | Wang et al. | CLIP | OOD Detection | - | ICCV, 2023 | Negative Prompts | - |
| [How Does Fine-Tuning Impact Out-of-Distribution Detection for Vision-Language Models?](https://arxiv.org/pdf/2306.06048) | Yifei Ming, Yixuan Li | - | OOD Detection | - | IJCV, 2023 | - | - |
| [Out-Of-Distribution Detection With Negative Prompts](https://openreview.net/pdf?id=nanyAujl6e) | Nie et al. | - | OOD Detection | - | ICLR, 2024 | Negative Prompts | - |
| [Negative Label Guided OOD Detection with Pretrained Vision-Language Models](https://arxiv.org/pdf/2403.20078) | Xue Jiang, Feng Liu, Zhen Fang, Hong Chen, Tongliang Liu, Feng Zheng, Bo Han | - | OOD Detection | - | ICLR, 2024 | Negative Label Guidance | - |
| [ID-like Prompt Learning for Few-Shot Out-of-Distribution Detection](https://arxiv.org/pdf/2311.15243) | Yichen Bai, Zongbo Han, Changqing Zhang, Bing Cao, Xiaoheng Jiang, Qinghua Hu | - | OOD Detection | - | CVPR, 2024 | - | - |
| [Learning transferable negative prompts for out-of-distribution detection](https://arxiv.org/abs/2404.03248) | Tianqi Li, Guansong Pang, Xiao Bai, Wenjun Miao, Jin Zheng | - | OOD Detection | - | CVPR, 2024 | - | - |
| [Adapting Contrastive Language-Image Pretrained (CLIP) Models for Out-of-Distribution Detection](https://openreview.net/pdf?id=YCgX7sJRF1) | Nikolas Adaloglou, Felix Michels, Tim Kaiser, Markus Kollmann | CLIP | OOD Detection | - | TMLR, 2024 | - | - |
| [Tagfog: Textual anchor guidance and fake outlier generation for visual out-of-distribution detection](https://ojs.aaai.org/index.php/AAAI/article/view/27871) | Jiankang Chen, Tong Zhang, Weishi Zheng, Ruixuan Wang | - | OOD Detection | - | AAAI, 2024 | Textual Anchor Guidance | - |
| [Domain aligned CLIP for few-shot classification](https://arxiv.org/abs/2311.09191) | Muhammad Waleed Gondal, Jochen Gast, Inigo Alonso Ruiz, Richard Droste, Tommaso Macri, Suren Kumar, Luitpold Staudigl | CLIP | Few-shot Classification | - | WACV, 2024 | - | - |
         
