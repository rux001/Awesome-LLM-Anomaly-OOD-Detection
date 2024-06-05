# Out-of-Distribution/Anomaly Detection with Foundation Models/LLM
## Table of Contents

- [Introduction](#introduction)
- [Tentaive Curated List of Papers](#tentaive-curated-list-of-papers)
  - [OOD Detection](#ood-detection)
    - [OOD Detection for NLP](#ood-detection-for-nlp)
    - [OOD Detection for CV](#ood-detection-for-CV)
    - [OOD Detection for Multimodal Models](#ood-detection-for-multimodal-models)
  - [Anomaly Detection](#anomaly-detection)
    - [AutoEncoder](#content)
    - [GAN](#content)
    - [Flow](#content)
    - [Diffusion Model](#content)
    - [Transformer](#content)
    - [Convolution](#content)
    - [GNN](#content)
    - [Time Series](#content)
    - [Tabular](#content)
    - [Large Model](#content)
    - [Reinforcement Learning](#content)
    - [Representation Learning](#content)
    - [Nonparametric Approach](#content)

## Introduction

## Tentaive Curated List of Papers

### OOD Detection for NLP

#### Pre-trained Language Models
(arXiv 2020) [Pretrained Transformers Improve Out-of-Distribution Robustness](https://arxiv.org/pdf/2004.06100) by Dan Hendrycks, Xiaoyuan Liu, Eric Wallace, Adam Dziedzic, Rishabh Krishnan, Dawn Song
** Transformers Based/BERT

(ACL 2021) [Unsupervised Out-of-Domain Detection via Pre-trained Transformers](https://arxiv.org/pdf/2106.00948) by Keyang Xu, Tongzheng Ren, Shikun Zhang, Yihao Feng, Caiming Xiong
** Transformers Based/BERT/RoBERTa

(ACL 2021) [Contrastive Out-of-Distribution Detection for Pretrained Transformers](https://arxiv.org/pdf/2104.08812) by Wenxuan Zhou, Fangyu Liu, Muhao Chen
** Transformers Based

(ACL 2023) [Is Fine-tuning Needed? Pre-trained Language Models Are Near Perfect for Out-of-Domain Detection](https://aclanthology.org/2023.acl-long.717.pdf) by Rheeya Uppaal, Junjie Hu, Yixuan Li

#### Large Language Models

(ICLR 2023) [Out-of-distribution detection and selective generation for conditional language models](https://arxiv.org/pdf/2209.15558) by Jie Ren, Jiaming Luo, Yao Zhao, Kundan Krishna, Mohammad Saleh, Balaji Lakshminarayanan, Peter J. Liu

(ACL 2023) [Multi-Level Knowledge Distillation for Out-of-Distribution Detection in Text](https://arxiv.org/pdf/2211.11300) by Qianhui Wu, Huiqiang Jiang, Haonan Yin, Börje F. Karlsson, Chin-Yew Lin

(COLING 2024) [How Good Are LLMs at Out-of-Distribution Detection?](https://arxiv.org/pdf/2308.10261) by Bo Liu, Liming Zhan, Zexin Lu, Yujie Feng, Lei Xue, Xiao-Ming Wu

(COLING 2024) [Beyond the Known: Investigating LLMs Performance on Out-of-Domain Intent Detection](https://arxiv.org/pdf/2402.17256) by Pei Wang, Keqing He, Yejie Wang, Xiaoshuai Song, Yutao Mou, Jingang Wang, Yunsen Xian, Xunliang Cai, Weiran Xu

(arXiv 2024) [VI-OOD: A Unified Representation Learning Framework for Textual Out-of-distribution Detection](https://arxiv.org/pdf/2404.06217) by Li-Ming Zhan, Bo Liu, Xiao-Ming Wu

### OOD Detection for CV

#### Vision Transformers
(NeurIPS 2021) [Exploring the Limits of Out-of-Distribution Detection](https://arxiv.org/pdf/2106.03004) by Stanislav Fort, Jie Ren, Balaji Lakshminarayanan

(arXiv 2021) [OODformer: Out-Of-Distribution Detection Transformer](https://arxiv.org/pdf/2107.08976) by Rajat Koner, Poulami Sinhamahapatra, Karsten Roscher, Stephan Günnemann, Volker Tresp

(CVPR 2023) [Rethinking Out-of-Distribution (OOD) Detection: Masked Image Modeling Is All You Need](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Rethinking_Out-of-Distribution_OOD_Detection_Masked_Image_Modeling_Is_All_You_CVPR_2023_paper.pdf) [[Code]](https://github.com/JulietLJY/MOOD) by Li et al.

(ICLR 2024 Workshop) [How to train your ViT for OOD Detection](https://arxiv.org/pdf/2405.17447) by Maximilian Mueller, Matthias Hein

### OOD Detection for Multimodal Models

#### Vision-Language Models
(NeurIPS 2021) [Exploring the Limits of Out-of-Distribution Detection](https://arxiv.org/pdf/2106.03004) by Stanislav Fort, Jie Ren, Balaji Lakshminarayanan

(NeurIPS 2022) [Delving into Out-of-Distribution Detection with Vision-Language Representations](https://arxiv.org/pdf/2211.13445.pdf) by Ming et al. [[Code]](https://github.com/deeplearning-wisc/MCM) [[Video]](https://www.youtube.com/watch?v=ZZlxBgGalVA)

(AAAI 2022) [Zero-shot out-of-distribution detection based on the pretrained model clip](https://arxiv.org/pdf/2109.02748) by Sepideh Esmaeilpour, Bing Liu, Eric Robertson, and Lei Shu.

(NeurIPS 2023) [LoCoOp: Few-Shot Out-of-Distribution Detection via Prompt Learning](https://arxiv.org/pdf/2306.01293) by Miyai et al. [[Code]](https://github.com/AtsuMiyai/LoCoOp)

(ICCV 2023) [CLIPN for Zero-Shot OOD Detection: Teaching CLIP to Say No](https://arxiv.org/pdf/2308.12213) by Wang et al. [[Code]](https://github.com/xmed-lab/CLIPN)

(IJCV 2023) [How Does Fine-Tuning Impact Out-of-Distribution Detection for Vision-Language Models?](https://arxiv.org/pdf/2306.06048) by Yifei Ming, Yixuan Li

(ICLR 2024) [Out-Of-Distribution Detection With Negative Prompts](https://openreview.net/pdf?id=nanyAujl6e) by Nie et al.

(ICLR 2024) [Negative Label Guided OOD Detection with Pretrained Vision-Language Models](https://arxiv.org/pdf/2403.20078) by Xue Jiang, Feng Liu, Zhen Fang, Hong Chen, Tongliang Liu, Feng Zheng, Bo Han

(CVPR 2024) [ID-like Prompt Learning for Few-Shot Out-of-Distribution Detection](https://arxiv.org/pdf/2311.15243) by Yichen Bai, Zongbo Han, Changqing Zhang, Bing Cao, Xiaoheng Jiang, Qinghua Hu

(TMLR 2024) [Adapting Contrastive Language-Image Pretrained (CLIP) Models for Out-of-Distribution Detection](https://openreview.net/pdf?id=YCgX7sJRF1) by Nikolas Adaloglou, Felix Michels, Tim Kaiser, Markus Kollmann

#### Diffusion Models

(NeurIPS 2022) [Delving into Out-of-Distribution Detection with Vision-Language Representations](https://arxiv.org/pdf/2211.13445.pdf) by Ming et al. [[Code]](https://github.com/deeplearning-wisc/MCM) [[Video]](https://www.youtube.com/watch?v=ZZlxBgGalVA)

(NeurIPS 2023) [Dream the Impossible: Outlier Imagination with Diffusion Models](https://arxiv.org/pdf/2309.13415) by Du et al.

(ICCV 2023) [DIFFGUARD: Semantic Mismatch-Guided Out-of-Distribution Detection using Pre-trained Diffusion Models](https://openaccess.thecvf.com/content/ICCV2023/papers/Gao_DIFFGUARD_Semantic_Mismatch-Guided_Out-of-Distribution_Detection_Using_Pre-Trained_Diffusion_Models_ICCV_2023_paper.pdf) by Gao et al. [[Code]](https://github.com/cure-lab/DiffGuard)

(ICCV 2023) [Deep Feature Deblurring Diffusion for Detecting Out-of-Distribution Objects](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Deep_Feature_Deblurring_Diffusion_for_Detecting_Out-of-Distribution_Objects_ICCV_2023_paper.pdf) by Wu et al. [[Code]](https://github.com/AmingWu/DFDD-OOD)

#### Generative Models
(ICLR 2023) [The Tilted Variational Autoencoder: Improving Out-of-Distribution Detection](https://openreview.net/pdf?id=YlGsTZODyjz) [[Code]](https://github.com/anonconfsubaccount/tilted_prior) by Floto et al.



#### Large Language Models

(EMNLP 2023) [Exploring Large Language Models for Multi-Modal Out-of-Distribution Detection](https://arxiv.org/pdf/2310.08027) by Yi Dai, Hao Lang, Kaisheng Zeng, Fei Huang, Yongbin Li

### Anomaly Detection
#### AutoEncoder
1. **Graph regularized autoencoder and its application in unsupervised anomaly detection.** TPAMI, 2022. [paper](https://ieeexplore.ieee.org/document/9380495)

   *Imtiaz Ahmed, Travis Galoppo, Xia Hu, and Yu Ding.* 

1. **Innovations autoencoder and its application in one-class anomalous sequence detection.** JMLR, 2022. [paper](https://www.jmlr.org/papers/volume23/21-0735/21-0735.pdf)

   *Xinyi Wang and Lang Tong.* 

1. **Autoencoders-A comparative analysis in the realm of anomaly detection.** CVPR, 2022. [paper](https://openaccess.thecvf.com/content/CVPR2022W/WiCV/html/Schneider_Autoencoders_-_A_Comparative_Analysis_in_the_Realm_of_Anomaly_CVPRW_2022_paper.html)

   *Sarah Schneider, Doris Antensteiner, Daniel Soukup, and Matthias Scheutz.* 

1. **Attention guided anomaly localization in images.** ECCV, 2020. [paper](https://link.springer.com/chapter/10.1007/978-3-030-58520-4_29)

   *Shashanka Venkataramanan, Kuan-Chuan Peng, Rajat Vikram Singh, and Abhijit Mahalanobis.* 

1. **Latent space autoregression for novelty detection.** CVPR, 2018. [paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Abati_Latent_Space_Autoregression_for_Novelty_Detection_CVPR_2019_paper.html)

   *Davide Abati, Angelo Porrello, Simone Calderara, and Rita Cucchiara.*

1. **Anomaly detection in time series with robust variational quasi-recurrent autoencoders.** ICDM, 2018. [paper](https://ieeexplore.ieee.org/abstract/document/9835268)

   *Tung Kieu, Bin Yang, Chenjuan Guo, Razvan-Gabriel Cirstea, Yan Zhao, Yale Song, and Christian S. Jensen.*

1. **Robust and explainable autoencoders for unsupervised time series outlier detection.** ICDE, 2022. [paper](https://ieeexplore.ieee.org/document/9835554)

   *Tung Kieu, Bin Yang, Chenjuan Guo, Christian S. Jensen, Yan Zhao, Feiteng Huang, and Kai Zheng.*

1. **Latent feature learning via autoencoder training for automatic classification configuration recommendation.** KBS, 2022. [paper](https://www.sciencedirect.com/science/article/pii/S0950705122013144)

   *Liping Deng and Mingqing Xiao.*

1. **Deep autoencoding Gaussian mixture model for unsupervised anomaly detection.** ICLR, 2018. [paper](https://openreview.net/forum?id=BJJLHbb0-)

   *Bo Zongy, Qi Songz, Martin Renqiang Miny, Wei Chengy, Cristian Lumezanuy, Daeki Choy, and Haifeng Chen.* 

1. **Anomaly detection with robust deep autoencoders.** KDD, 2017. [paper](https://dl.acm.org/doi/10.1145/3097983.3098052)

   *Chong Zhou and Randy C. Paffenroth.* 

1. **Unsupervised anomaly detection via variational auto-encoder for seasonal KPIs in web applications.** WWW, 2018. [paper](https://dl.acm.org/doi/abs/10.1145/3178876.3185996)

   *Haowen Xu, Wenxiao Chen, Nengwen Zhao,Zeyan Li, Jiahao Bu, Zhihan Li, Ying Liu, Youjian Zhao, Dan Pei, Yang Feng, Jie Chen, Zhaogang Wang, and Honglin Qiao.* 

1. **Spatio-temporal autoencoder for video anomaly detection.** MM, 2017. [paper](https://dl.acm.org/doi/abs/10.1145/3123266.3123451)

   *Yiru Zhao, Bing Deng, Chen Shen, Yao Liu, Hongtao Lu, and Xiansheng Hua.* 

1. **Learning discriminative reconstructions for unsupervised outlier removal.** ICCV, 2015. [paper](https://ieeexplore.ieee.org/document/7410534)

   *Yan Xia, Xudong Cao, Fang Wen, Gang Hua, and Jian Sun.* 

1. **Outlier detection with autoencoder ensembles.** ICDM, 2017. [paper](https://research.ibm.com/publications/outlier-detection-with-autoencoder-ensembles)

   *Jinghui Chen, Saket Sathey, Charu Aggarwaly, and Deepak Turaga.*

1. **A study of deep convolutional auto-encoders for anomaly detection in videos.** Pattern Recognition Letters, 2018. [paper](https://www.sciencedirect.com/science/article/pii/S0167865517302489)

   *Manassés Ribeiro, AndréEugênio Lazzaretti, and Heitor Silvério Lopes.*

1. **Classification-reconstruction learning for open-set recognition.** CVPR, 2019. [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yoshihashi_Classification-Reconstruction_Learning_for_Open-Set_Recognition_CVPR_2019_paper.pdf)

   *Ryota Yoshihashi, Shaodi You, Wen Shao, Makoto Iida, Rei Kawakami, and Takeshi Naemura.*

1. **Making reconstruction-based method great again for video anomaly detection.** ICDM, 2022. [paper](https://ieeexplore.ieee.org/abstract/document/10027694/)

   *Yizhou Wang, Can Qin, Yue Bai, Yi Xu, Xu Ma, and Yun Fu.*

1. **Two-stream decoder feature normality estimating network for industrial snomaly fetection.** ICASSP, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10027694/)

   *Chaewon Park, Minhyeok Lee, Suhwan Cho, Donghyeong Kim, and Sangyoun Lee.*

1. **Synthetic pseudo anomalies for unsupervised video anomaly detection: A simple yet efficient framework based on masked autoencoder.** ICASSP, 2023. [paper](https://arxiv.org/abs/2303.05112)

   *Xiangyu Huang, Caidan Zhao, Chenxing Gao, Lvdong Chen, and Zhiqiang Wu.*

1. **Deep autoencoding one-class time series anomaly detection.** ICASSP, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10095724)

   *Xudong Mou, Rui Wang, Tiejun Wang, Jie Sun, Bo Li, Tianyu Wo, and Xudong Liu.*

1. **Reconstruction error-based anomaly detection with few outlying examples.** arXiv, 2023. [paper](https://arxiv.org/abs/2305.10464)

   *Fabrizio Angiulli, Fabio Fassetti, and Luca Ferragina.*

1. **LARA: A light and anti-overfitting retraining approach for unsupervised anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.05668)

   *Feiyi Chen, Zhen Qing, Yingying Zhang, Shuiguang Deng, Yi Xiao, Guansong Pang, and Qingsong Wen.*

1. **FMM-Head: Enhancing autoencoder-based ECG anomaly detection with prior knowledge.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.05848)

   *Giacomo Verardo, Magnus Boman, Samuel Bruchfeld, Marco Chiesa, Sabine Koch, Gerald Q. Maguire Jr., and Dejan Kostic.*

1. **Online multi-view anomaly detection with disentangled product-of-experts modeling.** MM, 2023. [paper](https://arxiv.org/abs/2310.18728)

   *Hao Wang, Zhiqi Cheng, Jingdong Sun, Xin Yang, Xiao Wu, Hongyang Chen, and Yan Yang.*

1. **Fast particle-based anomaly detection algorithm with variational autoencoder.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.17162)

   *Ryan Liu, Abhijith Gandrakota, Jennifer Ngadiuba, Maria Spiropulu, and Jean-Roch Vlimant.*

1. **Dynamic erasing network based on multi-scale temporal features for weakly supervised video anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.01764)

   *Chen Zhang, Guorong Li, Yuankai Qi, Hanhua Ye, Laiyun Qing, Ming-Hsuan Yang, and Qingming Huang.*

1. **ACVAE: A novel self-adversarial variational auto-encoder combined with contrast learning for time series anomaly detection.** Neural Networks, 2023. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608023007281)

   *Xiaoxia Zhang, Shang Shi, HaiChao Sun, Degang Chen, Guoyin Wang, and Kesheng Wu.*

1. **Dual-constraint autoencoder and adaptive weighted similarity spatial attention for unsupervised anomaly detection.** TII, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10504620)

   *Ruifan Zhang, Hao Wang, Mingyao Feng, Yikun Liu, and Gongping Yang.*

#### [GAN]
1. **Stabilizing adversarially learned one-class novelty detection using pseudo anomalies.** TIP, 2022. [paper](https://ieeexplore.ieee.org/abstract/document/9887825)

   *Muhammad Zaigham Zaheer, Jin-Ha Lee, Arif Mahmood, Marcella Astri, and Seung-Ik Lee.* 

1. **GAN ensemble for anomaly detection.** AAAI, 2021. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/16530)

   *Han, Xu, Xiaohui Chen, and Liping Liu.* 

1. **Generative cooperative learning for unsupervised video anomaly detection.** CVPR, 2022. [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Zaheer_Generative_Cooperative_Learning_for_Unsupervised_Video_Anomaly_Detection_CVPR_2022_paper.html)

   *Zaigham Zaheer, Arif Mahmood, M. Haris Khan, Mattia Segu, Fisher Yu, and Seung-Ik Lee.* 

1. **GAN-based anomaly detection in imbalance problems.** ECCV, 2020. [paper](https://link.springer.com/chapter/10.1007/978-3-030-65414-6_11)

   *Junbong Kim, Kwanghee Jeong, Hyomin Choi, and Kisung Seo.* 

1. **Old is gold: Redefining the adversarially learned one-class classifier training paradigm.** CVPR, 2020. [paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Zaheer_Old_Is_Gold_Redefining_the_Adversarially_Learned_One-Class_Classifier_Training_CVPR_2020_paper.html)

   *Muhammad Zaigham Zaheer, Jin-ha Lee, Marcella Astrid, and Seung-Ik Lee.* 

1. **Unsupervised anomaly detection with generative adversarial networks to guide marker discovery.** IPMI, 2017. [paper](https://link.springer.com/chapter/10.1007/978-3-319-59050-9_12)

   *Thomas Schlegl, Philipp Seeböck, Sebastian M. Waldstein, Ursula Schmidt-Erfurth, and Georg Langs.* 

1. **Adversarially learned anomaly detection.** ICDM, 2018. [paper](https://ieeexplore.ieee.org/document/8594897)

   *Houssam Zenati, Manon Romain, Chuan-Sheng Foo, Bruno Lecouat, and Vijay Chandrasekhar.* 

1. **BeatGAN: Anomalous rhythm detection using adversarially generated time series.** IJCAI, 2019. [paper](https://www.ijcai.org/proceedings/2019/616)

   *Bin Zhou, Shenghua Liu, Bryan Hooi, Xueqi Cheng, and Jing Ye.* 

1. **Convolutional transformer based dual discriminator generative adversarial networks for video anomaly detection.** MM, 2021. [paper](https://dl.acm.org/doi/abs/10.1145/3474085.3475693)

   *Xinyang Feng, Dongjin Song, Yuncong Chen, Zhengzhang Chen, Jingchao Ni, and Haifeng Chen.* 

1. **USAD: Unsupervised anomaly detection on multivariate time series.** KDD, 2020. [paper](https://dl.acm.org/doi/abs/10.1145/3394486.3403392)

   *Julien Audibert, Pietro Michiardi, Frédéric Guyard, Sébastien Marti, and Maria A. Zuluaga.* 

1. **Anomaly detection with generative adversarial networks for multivariate time series.** ICLR, 2018. [paper](https://arxiv.org/abs/1809.04758)

   *Dan Li, Dacheng Chen, Jonathan Goh, and See-kiong Ng.* 

1. **Efficient GAN-based anomaly detection.** ICLR, 2018. [paper](https://arxiv.org/abs/1802.06222)

   *Houssam Zenati, Chuan Sheng Foo, Bruno Lecouat, Gaurav Manek, and Vijay Ramaseshan Chandrasekhar.* 

1. **GANomaly: Semi-supervised anomaly detection via adversarial training.** ACCV, 2019. [paper](https://link.springer.com/chapter/10.1007/978-3-030-20893-6_39)

   *Akcay, Samet, Amir Atapour-Abarghouei, and Toby P. Breckon.* 

1. **f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks.** Medical Image Analysis, 2019. [paper](https://www.sciencedirect.com/science/article/pii/S1361841518302640)

   *Thomas Schlegl, Philipp Seeböck, Sebastian M. Waldstein, Georg Langs, and Ursula Schmidt-Erfurth.* 

1. **OCGAN: One-class novelty detection using GANs with constrained latent representations.** CVPR, 2019. [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Perera_OCGAN_One-Class_Novelty_Detection_Using_GANs_With_Constrained_Latent_Representations_CVPR_2019_paper.pdf)

   *Pramuditha Perera, Ramesh Nallapati, and Bing Xiang.* 

1. **Adversarially learned one-class classifier for novelty detection.** CVPR, 2018. [paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sabokrou_Adversarially_Learned_One-Class_CVPR_2018_paper.pdf)

   *Mohammad Sabokrou, Mohammad Khalooei, Mahmood Fathy, and Ehsan Adeli.* 

1. **Generative probabilistic novelty detection with adversarial autoencoders.** NIPS, 2018. [paper](https://dl.acm.org/doi/10.5555/3327757.3327787)

   *Stanislav Pidhorskyi, Ranya Almohsen, Donald A. Adjeroh, and Gianfranco Doretto.* 

1. **Image anomaly detection with generative adversarial networks.** ECML PKDD, 2018. [paper](https://link.springer.com/chapter/10.1007/978-3-030-10925-7_1)

   *Lucas Deecke, Robert Vandermeulen, Lukas Ruff, Stephan Mandt, and Marius Kloft.*

1. **RGI: Robust GAN-inversion for mask-free image inpainting and unsupervised pixel-wise anomaly detection.** ICLR, 2023. [paper](https://openreview.net/forum?id=1UbNwQC89a)

   *Shancong Mou, Xiaoyi Gu, Meng Cao, Haoping Bai, Ping Huang, Jiulong Shan, and Jianjun Shi.*

1. **Truncated affinity maximization: One-class homophily modeling for graph anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2306.00006)

   *Qiao Hezhe and Pang Guansong.*

1. **Anomaly detection under contaminated data with contamination-immune bidirectional GANs.** TKDE, 2024. [paper](https://www.computer.org/csdl/journal/tk/5555/01/10536641/1X9vdwpnhO8)

   *Qinliang Su, Bowen Tian, Hai Wan, and Jian Yin.*

### [Flow]
1. **OneFlow: One-class flow for anomaly detection based on a minimal volume region.** TPAMI, 2022. [paper](https://ieeexplore.ieee.org/abstract/document/9525256)

   *Lukasz Maziarka, Marek Smieja, Marcin Sendera, Lukasz Struski, Jacek Tabor, and Przemyslaw Spurek.* 

1. **Comprehensive regularization in a bi-directional predictive network for video anomaly detection.** AAAI, 2022. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/19898)

   *Chengwei Chen, Yuan Xie, Shaohui Lin, Angela Yao, Guannan Jiang, Wei Zhang, Yanyun Qu, Ruizhi Qiao, Bo Ren, and Lizhuang Ma.* 

1. **Future frame prediction network for video anomaly detection.** TPAMI, 2022. [paper](https://ieeexplore.ieee.org/abstract/document/9622181/)

   *Weixin Luo, Wen Liu, Dongze Lian, and Shenghua Gao.* 

1. **Graph-augmented normalizing flows for anomaly detection of multiple time series.** ICLR, 2022. [paper](https://openreview.net/forum?id=45L_dgP48Vd)

   *Enyan Dai and Jie Chen.* 

1. **Cloze test helps: Effective video anomaly detection via learning to complete video events.** MM, 2020. [paper](https://dl.acm.org/doi/abs/10.1145/3394171.3413973)

   *Guang Yu, Siqi Wang, Zhiping Cai, En Zhu, Chuanfu Xu, Jianping Yin, and Marius Kloft.* 

1. **A modular and unified framework for detecting and localizing video anomalies.** WACV, 2022. [paper](https://openaccess.thecvf.com/content/WACV2022/html/Doshi_A_Modular_and_Unified_Framework_for_Detecting_and_Localizing_Video_WACV_2022_paper.html)

   *Keval Doshi and Yasin Yilmaz.*

1. **Video anomaly detection with compact feature sets for online performance.** TIP, 2017. [paper](https://ieeexplore.ieee.org/abstract/document/7903693)

   *Roberto Leyva, Victor Sanchez, and Chang-Tsun Li.*

1. **U-Flow: A U-shaped normalizing flow for anomaly detection with unsupervised threshold.** arXiv, 2017. [paper](https://arxiv.org/abs/2211.12353)

   *Matías Tailanian, Álvaro Pardo, and Pablo Musé.*

1. **Bi-directional frame interpolation for unsupervised video anomaly detection.** WACV, 2023. [paper](https://arxiv.org/abs/2211.12353)

   *Hanqiu Deng, Zhaoxiang Zhang, Shihao Zou, and Xingyu Li.*

1. **AE-FLOW: Autoencoders with normalizing flows for medical images anomaly detection.** ICLR, 2023. [paper](https://openreview.net/forum?id=9OmCr1q54Z)

   *Yuzhong Zhao, Qiaoqiao Ding, and Xiaoqun Zhang.*

1. **A video anomaly detection framework based on appearance-motion semantics representation consistency.** ICASSP, 2023. [paper](https://arxiv.org/abs/2303.05109)

   *Xiangyu Huang, Caidan Zhao, and Zhiqiang Wu.*

1. **Fully convolutional cross-scale-flows for image-based defect detection.** WACV, 2022. [paper](https://openaccess.thecvf.com/content/WACV2022/html/Rudolph_Fully_Convolutional_Cross-Scale-Flows_for_Image-Based_Defect_Detection_WACV_2022_paper.html)

   *Marco Rudolph, Tom Wehrbein, Bodo Rosenhahn, and Bastian Wandt.*

1. **CFLOW-AD: Real-time unsupervised anomaly detection with localization via conditional normalizing flows.** WACV, 2022. [paper](https://openaccess.thecvf.com/content/WACV2022/html/Gudovskiy_CFLOW-AD_Real-Time_Unsupervised_Anomaly_Detection_With_Localization_via_Conditional_Normalizing_WACV_2022_paper.html)

   *Denis Gudovskiy, Shun Ishizaka, and Kazuki Kozuka.*

1. **Same same but DifferNet: Semi-supervised defect detection with normalizing flows.** WACV, 2021. [paper](https://openaccess.thecvf.com/content/WACV2021/html/Rudolph_Same_Same_but_DifferNet_Semi-Supervised_Defect_Detection_With_Normalizing_Flows_WACV_2021_paper.html)

   *Marco Rudolph, Bastian Wandt, and Bodo Rosenhahn.*

1. **Normalizing flow based feature synthesis for outlier-aware object detection.** CVPR, 2023. [paper](https://arxiv.org/abs/2302.07106v2)

   *Nishant Kumar, Siniša Šegvić, Abouzar Eslami, and Stefan Gumhold.*

1. **DyAnNet: A scene dynamicity guided self-trained video anomaly detection network.** WACV, 2023. [paper](https://openaccess.thecvf.com/content/WACV2023/html/Thakare_DyAnNet_A_Scene_Dynamicity_Guided_Self-Trained_Video_Anomaly_Detection_Network_WACV_2023_paper.html)

   *Kamalakar Vijay Thakare, Yash Raghuwanshi, Debi Prosad Dogra, Heeseung Choi, and Ig-Jae Kim.*

1. **Multi-scale spatial-temporal interaction network for video anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2306.10239)

   *Zhiyuan Ning, Zhangxun Li, and Liang Song.*

1. **MSFlow: Multi-scale flow-based framework for unsupervised anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2308.15300)

   *Yixuan Zhou, Xing Xu, Jingkuan Song, Fumin Shen, and Hengtao Shen.*

1. **PyramidFlow: High-resolution defect contrastive localization using pyramid normalizing flow.** CVPR, 2023. [paper](https://ieeexplore.ieee.org/document/10204306)

   *Jiarui Lei, Xiaobo Hu, Yue Wang, and Dong Liu.*

1. **Topology-matching normalizing flows for out-of-distribution detection in robot learning.** CoRL, 2023. [paper](https://openreview.net/forum?id=BzjLaVvr955)

   *Jianxiang Feng, Jongseok Lee, Simon Geisler, Stephan Günnemann, and Rudolph Triebel.*

1. **Video anomaly detection via spatio-temporal pseudo-anomaly generation : A unified approach.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.16514)

   *Ayush K. Rai, Tarun Krishna, Feiyan Hu, Alexandru Drimbarean, Kevin McGuinness, Alan F. Smeaton, and Noel E. O'Connor.*

1. **Self-supervised normalizing flows for image anomaly detection and localization.** CVPR, 2023. [paper](https://openaccess.thecvf.com/content/CVPR2023W/VAND/html/Chiu_Self-Supervised_Normalizing_Flows_for_Image_Anomaly_Detection_and_Localization_CVPRW_2023_paper.html)

   *Li-Ling Chiu and Shang-Hong Lai.*

1. **Normalizing flows for human pose anomaly detection.** ICCV, 2023. [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Hirschorn_Normalizing_Flows_for_Human_Pose_Anomaly_Detection_ICCV_2023_paper.html)

   *Or Hirschorn and Shai Avidan.*

1. **Hierarchical Gaussian mixture normalizing flow modeling for unified anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2403.13349v1)

   *Xincheng Yao, Ruoqi Li, Zefeng Qian, Lu Wang, and Chongyang Zhang.*

#### [Diffusion Model]
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

#### [Transformer]
1. **Video anomaly detection via prediction network with enhanced spatio-temporal memory exchange.** ICASSP, 2022. [paper](https://ieeexplore.ieee.org/document/9747376)

   *Guodong Shen, Yuqi Ouyang, and Victor Sanchez.* 

1. **TranAD: Deep transformer networks for anomaly detection in multivariate time series data.** VLDB, 2022. [paper](https://dl.acm.org/doi/abs/10.14778/3514061.3514067)

   *Shreshth Tuli, Giuliano Casale, and Nicholas R. Jennings.* 

1. **Pixel-level anomaly detection via uncertainty-aware prototypical transformer.** MM, 2022. [paper](https://dl.acm.org/doi/abs/10.1145/3503161.3548082)

   *Chao Huang, Chengliang Liu, Zheng Zhang, Zhihao Wu, Jie Wen, Qiuping Jiang, and Yong Xu.* 

1. **AddGraph: Anomaly detection in dynamic graph using attention-based temporal GCN.** IJCAI, 2019. [paper](https://www.ijcai.org/proceedings/2019/614)

   *Li Zheng, Zhenpeng Li, Jian Li, Zhao Li, and Jun Gao.* 

1. **Anomaly transformer: Time series anomaly detection with association discrepancy.** ICLR, 2022. [paper](https://openreview.net/pdf?id=LzQQ89U1qm_)

   *Jiehui Xu, Haixu Wu, Jianmin Wang, and Mingsheng Long.* 

1. **Constrained adaptive projection with pretrained features for anomaly detection.** IJCAI, 2022. [paper](https://www.ijcai.org/proceedings/2022/0286.pdf)

   *Xingtai Gui, Di Wu, Yang Chang, and Shicai Fan.* 

1. **Self-training multi-sequence learning with transformer for weakly supervised video anomaly detection.** AAAI, 2022. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/20028)

   *Shuo Li, Fang Liu, and Licheng Jiao.* 

1. **Beyond outlier detection: Outlier interpretation by attention-guided triplet deviation network.** WWW, 2021. [paper](https://dl.acm.org/doi/10.1145/3442381.3449868)

   *Hongzuo Xu, Yijie Wang, Songlei Jian, Zhenyu Huang, Yongjun Wang, Ning Liu, and Fei Li.* 

1. **Framing algorithmic recourse for anomaly detection.** KDD, 2022. [paper](https://dl.acm.org/doi/abs/10.1145/3534678.3539344)

   *Debanjan Datta, Feng Chen, and Naren Ramakrishnan.* 

1. **Inpainting transformer for anomaly detection.** ICIAP, 2022. [paper](https://link.springer.com/chapter/10.1007/978-3-031-06430-2_33)

   *Jonathan Pirnay and Keng Chai.* 

1. **Self-supervised and interpretable anomaly detection using network transformers.** arXiv, 2022. [paper](https://arxiv.org/abs/2202.12997)

   *Daniel L. Marino, Chathurika S. Wickramasinghe, Craig Rieger, and Milos Manic.* 

1. **Anomaly detection in surveillance videos using transformer based attention model.** arXiv, 2022. [paper](https://arxiv.org/abs/2206.01524)

   *Kapil Deshpande, Narinder Singh Punn, Sanjay Kumar Sonbhadra, and Sonali Agarwal.* 

1. **Multi-contextual predictions with vision transformer for video anomaly detection.** arXiv, 2022. [paper](https://arxiv.org/abs/2206.08568?context=cs)

   *Joo-Yeon Lee, Woo-Jeoung Nam, and Seong-Whan Lee.* 

1. **Transformer based models for unsupervised anomaly segmentation in brain MR images.** arXiv, 2022. [paper](https://arxiv.org/abs/2207.02059)

   *Ahmed Ghorbel, Ahmed Aldahdooh, Shadi Albarqouni, and Wassim Hamidouche.* 

1. **HaloAE: An HaloNet based local transformer auto-encoder for anomaly detection and localization.** arXiv, 2022. [paper](https://arxiv.org/abs/2208.03486)

   *E. Mathian, H. Liu, L. Fernandez-Cuesta, D. Samaras, M. Foll, and L. Chen.* 

1. **Generalizable industrial visual anomaly detection with self-induction vision transformer.** arXiv, 2022. [paper](https://arxiv.org/abs/2211.12311)

   *Haiming Yao and Xue Wang.* 

1. **VT-ADL: A vision transformer network for image anomaly detection and localization.** ISIE, 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9576231)

   *Pankaj Mishra, Riccardo Verk, Daniele Fornasier, Claudio Piciarelli, and Gian Luca Foresti.* 

1. **Video event restoration based on keyframes for video anomaly detection.** CVPR, 2023. [paper](https://arxiv.org/abs/2304.05112)

   *Zhiwei Yang, Jing Liu, Zhaoyang Wu, Peng Wu, and Xiaotao Liu.* 

1. **AnomalyBERT: Self-supervised Transformer for time series anomaly detection using data degradation scheme.** ICLR, 2023. [paper](https://arxiv.org/abs/2305.04468)

   *Yungi Jeong, Eunseok Yang, Jung Hyun Ryu, Imseong Park, and Myungjoo Kang.* 

1. **HAN-CAD: Hierarchical attention network for context anomaly detection in multivariate time series.** WWW, 2023. [paper](https://link.springer.com/article/10.1007/s11280-023-01171-1)

   *Haicheng Tao, Jiawei Miao, Lin Zhao, Zhenyu Zhang, Shuming Feng, Shu Wang, and Jie Cao.* 

1. **DCdetector: Dual attention contrastive representation learning for time series anomaly detection.** KDD, 2023. [paper](https://arxiv.org/abs/2306.10347)

   *Yiyuan Yang, Chaoli Zhang, Tian Zhou, Qingsong Wen, and Liang Sun.* 

1. **SelFormaly: Towards task-agnostic unified anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.12540)

   *Yujin Lee, Harin Lim, and Hyunsoo Yoon.* 

1. **MIM-OOD: Generative masked image modelling for out-of-distribution detection in medical images.** MICCAI, 2023. [paper](https://arxiv.org/abs/2307.14701)

   *Sergio Naval Marimont, Vasilis Siomos, and Giacomo Tarroni.* 

1. **Focus the discrepancy: Intra- and Inter-correlation learning for image anomaly detection.** ICCV, 2023. [paper](https://arxiv.org/abs/2308.02983)

   *Xincheng Yao, Ruoqi Li, Zefeng Qian, Yan Luo, and Chongyang Zhang.* 

1. **Sparse binary Transformers for multivariate time series modeling.** KDD, 2023. [paper](https://arxiv.org/abs/2308.04637)

   *Matt Gorbett, Hossein Shirazi, and Indrakshi Ray.* 

1. **ADFA: Attention-augmented differentiable top-k feature adaptation for unsupervised medical anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2308.15280)

   *Yiming Huang, Guole Liu, Yaoru Luo, and Ge Yang.* 

1. **Mask2Anomaly: Mask Transformer for universal open-set segmentation.** arXiv, 2023. [paper](https://arxiv.org/abs/2309.04573)

   *Shyam Nandan Rai, Fabio Cermelli, Barbara Caputo, and Carlo Masone.* 

1. **Hierarchical vector quantized Transformer for multi-class unsupervised anomaly detection.** NIPS, 2023. [paper](https://openreview.net/forum?id=clJTNssgn6)

   *Ruiying Lu, YuJie Wu, Long Tian, Dongsheng Wang, Bo Chen, Xiyang Liu, and Ruimin Hu.* 

1. **Attention modules improve image-level anomaly detection for industrial inspection: A DifferNet case study.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.02747)

   *Andre Luiz Vieira e Silva, Francisco Simoes, Danny Kowerko2 Tobias Schlosser, Felipe Battisti, and Veronica Teichrieb.* 

1. **Exploring plain ViT reconstruction for multi-class unsupervised anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.07495)

   *Jiangning Zhang, Xuhai Chen, Yabiao Wang, Chengjie Wang, Yong Liu, Xiangtai Li, Ming-Hsuan Yang, and Dacheng Tao.* 

1. **Self-supervised masked convolutional transformer block for anomaly detection.** TPAMI, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10273635)

   *Neelu Madan, Nicolae-Cătălin Ristea, Radu Tudor Ionescu, Kamal Nasrollahi, Fahad Shahbaz Khan, Thomas B. Moeslund, and Mubarak Shah.* 

1. **Transformer-based multivariate time series anomaly detection using inter-variable attention mechanism.** KBS, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10273635)

   *Hyeongwon Kang and Pilsung Kang.* 

1. **Sub-adjacent Transformer: Improving time series anomaly detection with reconstruction error from sub-adjacent neighborhoods.** IJCAI, 2024. [paper](https://arxiv.org/abs/2404.18948)

   *Wenzhen Yue, Xianghua Ying, Ruohao Guo, DongDong Chen, Ji Shi, Bowei Xing, Yuqing Zhu, and Taiyan Chen.* 

1. **Dinomaly: The less is more philosophy in multi-class unsupervised anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.14325)

   *Jia Guo, Shuai Lu, Weihang Zhang, and Huiqi Li.* 

1. **How to train your ViT for OOD Detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.17447)

   *Maximilian Mueller and Matthias Hein.* 

#### [Convolution]
1. **Self-supervised predictive convolutional attentive block for anomaly detection.** CVPR, 2022. [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Ristea_Self-Supervised_Predictive_Convolutional_Attentive_Block_for_Anomaly_Detection_CVPR_2022_paper.html)

   *Nicolae-Catalin Ristea, Neelu Madan, Radu Tudor Ionescu, Kamal Nasrollahi, Fahad Shahbaz Khan, Thomas B. Moeslund, and Mubarak Shah.* 

1. **Catching both gray and black swans: Open-set supervised anomaly detection.** CVPR, 2022. [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Ding_Catching_Both_Gray_and_Black_Swans_Open-Set_Supervised_Anomaly_Detection_CVPR_2022_paper.html)

   *Choubo Ding, Guansong Pang, and Chunhua Shen.* 

1. **Learning memory-guided normality for anomaly detection.** CVPR, 2020. [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Park_Learning_Memory-Guided_Normality_for_Anomaly_Detection_CVPR_2020_paper.pdf)

   *Hyunjong Park, Jongyoun No, and Bumsub Ham.* 

1. **CutPaste: Self-supervised learning for anomaly detection and localization.** CVPR, 2021. [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_CutPaste_Self-Supervised_Learning_for_Anomaly_Detection_and_Localization_CVPR_2021_paper.pdf)

   *Chunliang Li, Kihyuk Sohn, Jinsung Yoon, and Tomas Pfister.* 

1. **Object-centric auto-encoders and dummy anomalies for abnormal event detection in video.** CVPR, 2019. [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ionescu_Object-Centric_Auto-Encoders_and_Dummy_Anomalies_for_Abnormal_Event_Detection_in_CVPR_2019_paper.pdf)

   *Radu Tudor Ionescu, Fahad Shahbaz Khan, Mariana-Iuliana Georgescu, and Ling Shao.* 

1. **Mantra-Net: Manipulation tracing network for detection and localization of image forgeries with anomalous features.** CVPR, 2019. [paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_ManTra-Net_Manipulation_Tracing_Network_for_Detection_and_Localization_of_Image_CVPR_2019_paper.html)

   *Yue Wu, Wael AbdAlmageed, and Premkumar Natarajan.* 

1. **Grad-CAM: Visual explanations from deep networks via gradient-based localization.** ICCV, 2017. [paper](https://ieeexplore.ieee.org/document/8237336)

   *Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra.* 

1. **A deep neural network for unsupervised anomaly detection and diagnosis in multivariate time series data.** AAAI, 2019. [paper](https://dl.acm.org/doi/10.1609/aaai.v33i01.33011409)

   *Chuxu Zhang, Dongjin Song, Yuncong Chen, Xinyang Feng, Cristian Lumezanu, Wei Cheng, Jingchao Ni, Bo Zong, Haifeng Chen, and Nitesh V. Chawla.* 

1. **Real-world anomaly detection in surveillance videos.** CVPR, 2018. [paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.pdf)

   *Waqas Sultani, Chen Chen, and Mubarak Shah.* 

1. **FastAno: Fast anomaly detection via spatio-temporal patch transformation.** WACV, 2022. [paper](https://openaccess.thecvf.com/content/WACV2022/papers/Park_FastAno_Fast_Anomaly_Detection_via_Spatio-Temporal_Patch_Transformation_WACV_2022_paper.pdf)

   *Chaewon Park, MyeongAh Cho, Minhyeok Lee, and Sangyoun Lee.* 

1. **Object class aware video anomaly detection through image translation.** CRV, 2022. [paper](https://www.computer.org/csdl/proceedings-article/crv/2022/977400a090/1GeCy7y5kgU)

   *Mohammad Baradaran and Robert Bergevin.* 

1. **Anomaly detection in video sequence with appearance-motion correspondence.** ICCV, 2019. [paper](https://ieeexplore.ieee.org/document/9009067)

   *Trong-Nguyen Nguyen and Jean Meunier.* 

1. **Joint detection and recounting of abnormal events by learning deep generic knowledge.** ICCV, 2017. [paper](https://openaccess.thecvf.com/content_iccv_2017/html/Hinami_Joint_Detection_and_ICCV_2017_paper.html)

   *Ryota Hinami, Tao Mei, and Shin’ichi Satoh.* 

1. **Deep-cascade: Cascading 3D deep neural networks for fast anomaly detection and localization in crowded scenes.** TIP, 2017. [paper](https://ieeexplore.ieee.org/abstract/document/7858798)

   *Mohammad Sabokrou, Mohsen Fayyaz, Mahmood Fathy, and Reinhard Klette.* 

1. **Towards interpretable video anomaly detection.** WACV, 2023. [paper](https://openaccess.thecvf.com/content/WACV2023/html/Doshi_Towards_Interpretable_Video_Anomaly_Detection_WACV_2023_paper.html)

   *Keval Doshi and Yasin Yilmaz.* 

1. **Lossy compression for robust unsupervised time-series anomaly detection.** CVPR, 2023. [paper](https://arxiv.org/abs/2212.02303)

   *Christopher P. Ley and Jorge F. Silva.* 

1. **Learning second order local anomaly for general face forgery detection.** CVPR, 2022. [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Fei_Learning_Second_Order_Local_Anomaly_for_General_Face_Forgery_Detection_CVPR_2022_paper.html)

   *Jianwei Fei, Yunshu Dai, Peipeng Yu, Tianrun Shen, Zhihua Xia, and Jian Weng.* 

#### [GNN]
1. **Graph convolutional label noise cleaner: Train a plug-and-play action classifier for anomaly detection.** CVPR, 2019. [paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhong_Graph_Convolutional_Label_Noise_Cleaner_Train_a_Plug-And-Play_Action_Classifier_CVPR_2019_paper.html)

   *Jiaxing Zhong, Nannan Li, Weijie Kong, Shan Liu, Thomas H. Li, and Ge Li.* 

1. **Towards open set video anomaly detection.** ECCV, 2019. [paper](https://link.springer.com/chapter/10.1007/978-3-031-19830-4_23)

   *Yuansheng Zhu, Wentao Bao, and Qi Yu.* 

1. **Decoupling representation learning and classification for GNN-based anomaly detection.** SIGIR, 2021. [paper](https://dl.acm.org/doi/10.1145/3404835.3462944)

   *Yanling Wan,, Jing Zhang, Shasha Guo, Hongzhi Yin, Cuiping Li, and Hong Chen.* 

1. **Crowd-level abnormal behavior detection via multi-scale motion consistency learning.** AAAI, 2023. [paper](https://arxiv.org/abs/2212.00535)

   *Linbo Luo, Yuanjing Li, Haiyan Yin, Shangwei Xie, Ruimin Hu, and Wentong Cai.* 

1. **Rethinking graph neural networks for anomaly detection.** ICML, 2022. [paper](https://proceedings.mlr.press/v162/tang22b/tang22b.pdf)

   *Jianheng Tang, Jiajin Li, Ziqi Gao, and Jia Li.* 

1. **Cross-domain graph anomaly detection via anomaly-aware contrastive alignment.** AAAI, 2023. [paper](https://arxiv.org/abs/2212.01096)

   *Qizhou Wang, Guansong Pang, Mahsa Salehi, Wray Buntine, and Christopher Leckie.* 

1. **A causal inference look at unsupervised video anomaly detection.** AAAI, 2022. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/20053)

   *Xiangru Lin, Yuyang Chen, Guanbin Li, and Yizhou Yu.* 

1. **NetWalk: A flexible deep embedding approach for anomaly detection in dynamic networks.** KDD, 2018. [paper](https://dl.acm.org/doi/10.1145/3219819.3220024)

   *Wenchao Yu, Wei Cheng, Charu C. Aggarwal, Kai Zhang, Haifeng Chen, and Wei Wang.* 

1. **LUNAR: Unifying local outlier detection methods via graph neural networks.** AAAI, 2022. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/20629)

   *Adam Goodge, Bryan Hooi, See-Kiong Ng, and Wee Siong Ng.* 

1. **Series2Graph: Graph-based subsequence anomaly detection for time series.** VLDB, 2022. [paper](https://dl.acm.org/doi/10.14778/3407790.3407792)

   *Paul Boniol and Themis Palpanas.* 

1. **Graph embedded pose clustering for anomaly detection.** CVPR, 2020. [paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Markovitz_Graph_Embedded_Pose_Clustering_for_Anomaly_Detection_CVPR_2020_paper.html)

   *Amir Markovitz, Gilad Sharir, Itamar Friedman, Lihi Zelnik-Manor, and Shai Avidan.* 

1. **Fast memory-efficient anomaly detection in streaming heterogeneous graphs.** KDD, 2016. [paper](https://dl.acm.org/doi/abs/10.1145/2939672.2939783)

   *Emaad Manzoor, Sadegh M. Milajerdi, and Leman Akoglu.*

1. **Raising the bar in graph-level anomaly detection.** IJCAI, 2022. [paper](https://www.ijcai.org/proceedings/2022/0305.pdf)

   *Chen Qiu, Marius Kloft, Stephan Mandt, and Maja Rudolph.*

1. **SpotLight: Detecting anomalies in streaming graphs.** KDD, 2018. [paper](https://dl.acm.org/doi/abs/10.1145/3219819.3220040)

   *Dhivya Eswaran, Christos Faloutsos, Sudipto Guha, and Nina Mishra.* 

1. **Graph anomaly detection via multi-scale contrastive learning networks with augmented view.** AAAI, 2023. [paper](https://arxiv.org/abs/2212.00535)

   *Jingcan Duan, Siwei Wang, Pei Zhang, En Zhu, Jingtao Hu, Hu Jin, Yue Liu, and Zhibin Dong.* 

1. **Counterfactual graph learning for anomaly detection on attributed networks.** TKDE, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10056298)

   *Chunjing Xiao, Xovee Xu, Yue Lei, Kunpeng Zhang, Siyuan Liu, and Fan Zhou.*

1. **Deep variational graph convolutional recurrent network for multivariate time series anomaly detection.** ICML, 2022. [paper](https://proceedings.mlr.press/v162/chen22x.html)

   *Wenchao Chen, Long Tian, Bo Chen, Liang Dai, Zhibin Duan, and Mingyuan Zhou.*

1. **SAD: Semi-supervised anomaly detection on dynamic graphs.** arXiv, 2023. [paper](https://arxiv.org/abs/2305.13573)

   *Sheng Tian, Jihai Dong, Jintang Li, Wenlong Zhao, Xiaolong Xu, Baokun wang, Bowen Song, Changhua Meng, Tianyi Zhang, and Liang Chen.*

1. **Improving generalizability of graph anomaly detection models via data augmentation.** TKDE, 2023. [paper](https://arxiv.org/abs/2306.10534)

   *Shuang Zhou, Xiao Huang, Ninghao Liu, Huachi Zhou, Fu-Lai Chung, and Long-Kai Huang.*

1. **Anomaly detection in networks via score-based generative models.** ICML, 2023. [paper](https://arxiv.org/abs/2306.15324)

   *Dmitrii Gavrilev and Evgeny Burnaev.*

1. **Generated graph detection.** ICML, 2023. [paper](https://openreview.net/forum?id=OoTa4H6Bnz)

   *Yihan Ma, Zhikun Zhang, Ning Yu, Xinlei He, Michael Backes, Yun Shen, and Yang Zhang.*

1. **Graph-level anomaly detection via hierarchical memory networks.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.00755)

   *Chaoxi Niu, Guansong Pang, and Ling Chen.*

1. **CSCLog: A component subsequence correlation-aware log anomaly detection method.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.03359)

   *Ling Chen, Chaodu Song, Xu Wang, Dachao Fu, and Feifei Li.*

1. **A survey on graph neural networks for time series: Forecasting, classification, imputation, and anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.03759)

   *Ming Jin, Huan Yee Koh, Qingsong Wen, Daniele Zambon, Cesare Alippi, Geoffrey I. Webb, Irwin King, and Shirui Pan.*

1. **Correlation-aware spatial-temporal graph learning for multivariate time-series anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.08390)

   *Yu Zheng, Huan Yee Koh, Ming Jin, Lianhua Chi, Khoa T. Phan, Shirui Pan, Yi-Ping Phoebe Chen, and Wei Xiang.*

1. **Graph anomaly detection at group level: A topology pattern enhanced unsupervised approach.** arXiv, 2023. [paper](https://arxiv.org/abs/2308.01063)

   *Xing Ai, Jialong Zhou, Yulin Zhu, Gaolei Li, Tomasz P. Michalak, Xiapu Luo, and Kai Zhou.*

1. **HRGCN: Heterogeneous graph-level anomaly detection with hierarchical relation-augmented graph neural networks.** arXiv, 2023. [paper](https://arxiv.org/abs/2308.14340)

   *Jiaxi Li, Guansong Pang, Ling Chen, Mohammad-Reza and Namazi-Rad.*

1. **Revisiting adversarial attacks on graph neural networks for graph classification.** TKDE, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10243054)

   *Xin Wang, Heng Chang, Beini Xie, Tian Bian, Shiji Zhou, Daixin Wang, Zhiqiang Zhang, and Wenwu Zhu.*

1. **Normality learning-based graph anomaly detection via multi-scale contrastive learning.** MM, 2023. [paper](https://arxiv.org/abs/2309.06034)

   *Jingcan Duan, Pei Zhang, Siwei Wang, Jingtao Hu, Hu Jin, Jiaxin Zhang, Haifang Zhou, and Haifang Zhou.*

1. **GLAD: Content-aware dynamic graphs for log anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2309.05953)

   *Yufei Li, Yanchi Liu, Haoyu Wang, Zhengzhang Chen, Wei Cheng, Yuncong Chen, Wenchao Yu, Haifeng Chen, and Cong Liu.*

1. **ARISE: Graph anomaly detection on attributed networks via substructure awareness.** TNNLS, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10258476)

   *Jingcan Duan, Bin Xiao, Siwei Wang, Haifang Zhou, and Xinwang Liu.*

1. **Rayleigh quotient graph neural networks for graph-level anomaly detection.** ICLR, 2024. [paper](https://openreview.net/forum?id=4UIBysXjVq)

   *Xiangyu Dong, Xingyi Zhang, and Sibo Wang.*

1. **Self-discriminative modeling for anomalous graph detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.06261)

   *Jinyu Cai, Yunhe Zhang, and Jicong Fan.*

1. **CVTGAD: Simplified transformer with cross-view attention for unsupervised graph-level anomaly detection.** ECML PKDD, 2023. [paper](https://link.springer.com/chapter/10.1007/978-3-031-43412-9_11)

   *Jindong Li, Qianli Xing, Qi Wang, and Yi Chang.*

1. **PREM: A simple yet effective approach for node-level graph anomaly detection.** ICDM, 2023. [paper](https://arxiv.org/abs/2310.11676)

   *Junjun Pan, Yixin Liu, Yizhen Zheng, and Shirui Pan.*

1. **THGNN: An embedding-based model for anomaly detection in dynamic heterogeneous social networks.** CIKM, 2023. [paper](https://dl.acm.org/doi/10.1145/3583780.3615079)

   *Yilin Li, Jiaqi Zhu, Congcong Zhang, Yi Yang, Jiawen Zhang, Ying Qiao, and Hongan Wang.*

1. **Learning node abnormality with weak supervision.** CIKM, 2023. [paper](https://dl.acm.org/doi/abs/10.1145/3583780.3614950)

   *Qinghai Zhou, Kaize Ding, Huan Liu, and Hanghang Tong.*

1. **RustGraph: Robust anomaly detection in dynamic graphs by jointly learning structural-temporal dependency.** TKDE, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10301657)

   *Jianhao Guo, Siliang Tang, Juncheng Li, Kaihang Pan, and Lingfei Wu.*

1. **An efficient adaptive multi-kernel learning with safe screening rule for outlier detection.** TKDE, 2023. [paper](https://ieeexplore.ieee.org/document/10310242)

   *Xinye Wang, Lei Duan, Chengxin He, Yuanyuan Chen, and Xindong Wu.*

1. **Anomaly detection in continuous-time temporal provenance graphs.** NIPS, 2023. [paper](https://nips.cc/virtual/2023/76336)

   *Jakub Reha, Giulio Lovisotto, Michele Russo, Alessio Gravina, and Claas Grohnfeldt.*

1. **Open-set graph anomaly detection via normal structure regularisation.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.06835)

   *Qizhou Wang, Guansong Pang, Mahsa Salehi, Wray Buntine, and Christopher Leckie.* 

1. **ADAMM: Anomaly detection of attributed multi-graphs with metadata: A unified neural network approach.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.07355)

   *Konstantinos Sotiropoulos, Lingxiao Zhao, Pierre Jinghong Liang, and Leman Akoglu.* 

1. **Deep joint adversarial learning for anomaly detection on attribute networks.** Information Sciences, 2023. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0020025523014251)

   *Haoyi Fan, Ruidong Wang, Xunhua Huang, Fengbin Zhang, Zuoyong Li, and Shimei Su.* 

1. **Few-shot message-enhanced contrastive learning for graph anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.10370)

   *Fan Xu, Nan Wang, Xuezhi Wen, Meiqi Gao, Chaoqun Guo, and Xibin Zhao.* 

1. **OCGEC: One-class graph embedding classification for DNN backdoor detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.01585)

   *Haoyu Jiang, Haiyang Yu, Nan Li, and Ping Yi.* 

1. **Reinforcement neighborhood selection for unsupervised graph anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.05526)

   *Yuanchen Bei, Sheng Zhou, Qiaoyu Tan, Hao Xu, Hao Chen, Zhao Li, and Jiajun Bu.* 

1. **ADA-GAD: Anomaly-denoised autoencoders for graph anomaly detection.** AAAI, 2024. [paper](https://arxiv.org/abs/2312.14535)

   *Junwei He, Qianqian Xu, Yangbangyan Jiang, Zitai Wang, and Qingming Huang.* 

1. **Boosting graph anomaly detection with adaptive message passing.** ICLR, 2024. [paper](https://openreview.net/forum?id=CanomFZssu)

   *Anonymous authors.* 

1. **Frequency domain-oriented complex graph neural networks for graph classification.** TNNLS, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10409552)

   *Youfa Liu and Bo Du.* 

1. **FGAD: Self-boosted knowledge distillation for an effective federated graph anomaly detection framework.** arXiv, 2024. [paper](https://arxiv.org/abs/2402.12761)

   *Jinyu Cai, Yunhe Zhang, Zhoumin Lu, Wenzhong Guo, and See-kiong Ng.* 

1. **Generative semi-supervised graph anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2402.11887)

   *Hezhe Qiao, Qingsong Wen, Xiaoli Li, Ee-Peng Lim, and Guansong Pang.* 

1. **Graph structure reshaping against adversarial attacks on graph neural networks.** TKDE, 2024. [paper](https://www.computer.org/csdl/journal/tk/5555/01/10538390/1XcOSbOdJD2)

   *Haibo Wang, Chuan Zhou, Xin Chen, Jia Wu, Shirui Pan, Zhao Li, Jilong Wang, and Philip S. Yu.* 

1. **SmoothGNN: Smoothing-based GNN for unsupervised node anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.17525)

   *Xiangyu Dong, Xingyi Zhang, Yanni Sun, Lei Chen, Mingxuan Yuan, and Sibo Wang.* 

1. **Learning-based link anomaly detection in continuous-time dynamic graphs.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.18050)

   *Tim Poštuvan, Claas Grohnfeldt, Michele Russo, and Giulio Lovisotto.* 

#### [Time Series]
1. **Variational LSTM enhanced anomaly detection for industrial big data.** TII, 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9195000)

   *Xiaokang Zhou, Yiyong Hu, Wei Liang, Jianhua Ma, and Qun Jin.* 

1. **Robust anomaly detection for multivariate time series through stochastic recurrent neural network.** KDD, 2019. [paper](https://dl.acm.org/doi/10.1145/3292500.3330672)

   *Ya Su, Youjian Zhao, Chenhao Niu, Rong Liu, Wei Sun, and Dan Pei.* 

1. **DeepLog: Anomaly detection and diagnosis from system logs through deep learning.** CCS, 2017. [paper](https://dl.acm.org/doi/10.1145/3133956.3134015)

   *Min Du, Feifei Li, Guineng Zheng, and Vivek Srikumar.* 

1. **Unsupervised anomaly detection with LSTM neural networks.** TNNLS, 2019. [paper](https://ieeexplore.ieee.org/abstract/document/8836638)

   *Tolga Ergen and Suleyman Serdar Kozat.* 

1. **LogAnomaly: Unsupervised detection of sequential and quantitative anomalies in unstructured logs.** IJCAI, 2019. [paper](https://www.ijcai.org/proceedings/2019/658)

   *Weibin Meng, Ying Liu, Yichen Zhu, Shenglin Zhang, Dan Pei, Yuqing Liu, Yihao Chen, Ruizhi Zhang, Shimin Tao, Pei Sun, and Rong Zhou.* 

1. **Outlier detection for time series with recurrent autoencoder ensembles.** IJCAI, 2019. [paper](https://dl.acm.org/doi/abs/10.5555/3367243.3367418)

   *Tung Kieu, Bin Yang, Chenjuan Guo, and Christian S. Jensen.* 

1. **Learning regularity in skeleton trajectories for anomaly detection in videos.** CVPR, 2019. [paper](https://dl.acm.org/doi/abs/10.5555/3367243.3367418)

   *Romero Morais, Vuong Le, Truyen Tran, Budhaditya Saha, Moussa Mansour, and Svetha Venkatesh.* 

1. **LSTM-based encoder-decoder for multi-sensor anomaly detection.** arXiv, 2016. [paper](https://arxiv.org/abs/1607.00148)

   *Pankaj Malhotra, Anusha Ramakrishnan, Gaurangi Anand, Lovekesh Vig, Puneet Agarwal, and Gautam Shroff.* 

1. **CrossFuN: Multi-view joint cross fusion network for time series anomaly detection.** TIM, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10254685)

   *Yunfei Bai, Jing Wang, Xueer Zhang, Xiangtai Miao, and Youfang Linf.* 

1. **Unsupervised anomaly detection by densely contrastive learning for time series data.** Neural Networks, 2023. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608023005385)

   *Wei Zhu, Weijian Li, E. Ray Dorsey, and Jiebo Luo.* 

1. **Algorithmic recourse for anomaly detection in multivariate time series.** arXiv, 2023. [paper](https://arxiv.org/abs/2309.16896)

   *Xiao Han, Lu Zhang, Yongkai Wu, and Shuhan Yuan.* 

1. **Unravel anomalies: An end-to-end seasonal-trend decomposition approach for time series anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.00268)

   *Zhenwei Zhang, Ruiqi Wang, Ran Ding, and Yuantao Gu.*

1. **MAG: A novel approach for effective anomaly detection in spacecraft telemetry data.** TII, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10269707)

   *Bing Yu, Yang Yu, Jiakai Xu, Gang Xiang, and Zhiming Yang.*

1. **Duogat: Dual time-oriented graph attention networks for accurate, efficient and explainable anomaly detection on time-series.** CIKM, 2023. [paper](https://dl.acm.org/doi/abs/10.1145/3583780.3614857)

   *Jongsoo Lee, Byeongtae Park, and Dong-Kyu Chae.*

1. **An enhanced spatio-temporal constraints network for anomaly detection in multivariate time series.** KBS, 2023. [paper](https://www.sciencedirect.com/science/article/abs/pii/S095070512300919X)

   *Di Ge, Zheng Dong, Yuhang Cheng, and Yanwen Wu.*

1. **Asymmetric autoencoder with SVD regularization for multivariate time series anomaly detection.** Neural Networks, 2023. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608023006469)

   *Yueyue Yao, Jianghong Ma, Shanshan Feng, and Yunming Ye.*

1. **Unraveling the anomaly in time series anomaly detection: A self-supervised tri-domain solution.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.11235)

   *Yuting Sun, Guansong Pang, Guanhua Ye, Tong Chen, Xia Hu, and Hongzhi Yin.*

1. **A filter-augmented auto-encoder with learnable normalization for robust multivariate time series anomaly detection.** Neural Networks, 2023. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608023006706)

   *Jiahao Yu, Xin Gao, Baofeng Li, Feng Zhai, Jiansheng Lu, Bing Xue, Shiyuan Fu, and Chun Xiao.*

1. **MEMTO: Memory-guided Transformer for multivariate time series anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.02530)

   *Junho Song, Keonwoo Kim, Jeonglyul Oh, and Sungzoon Cho.*

1. **Entropy causal graphs for multivariate time series anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.09478)

   *Falih Gozi Febrinanto, Kristen Moore, Chandra Thapa, Mujie Liu, Vidya Saikrishna, Jiangang Ma, and Feng Xia.*

1. **Label-free multivariate time series anomaly detection.** TKDE, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10380724)

   *Qihang Zhou, Shibo He, Haoyu Liu, Jiming Chen, and Wenchao Meng.*

1. **Quantile-long short term memory: A robust, time series anomaly detection method.** TAI, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10398596)

   *Snehanshu Saha, Jyotirmoy Sarkar, Soma Dhavala, Preyank Mota, and Santonu Sarkar.*

1. **PatchAD: Patch-based mlp-mixer for time series anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2401.09793)

   *Zhijie Zhong, Zhiwen Yu, Yiyuan Yang, Weizheng Wang, and Kaixiang Yang.*

1. **MELODY: Robust semi-supervised hybrid model for entity-level online anomaly detection with multivariate time series.** arXiv, 2024. [paper](https://arxiv.org/abs/2401.10338v1)

   *Jingchao Ni, Gauthier Guinet, Peihong Jiang, Laurent Callot, and Andrey Kan.*

1. **Understanding time series anomaly state detection through one-class classification.** arXiv, 2024. [paper](https://arxiv.org/abs/2402.02007v1)

   *Hanxu Zhou, Yuan Zhang, Guangjie Leng, Ruofan Wang, and Zhi-Qin John Xu.*

1. **Asymptotic consistent graph structure learning for multivariate time-series anomaly detection.** TIM, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10445316)

   *Huaxin Pang, Shikui Wei, Youru Li, Ting Liu, Huaqi Zhang, Ying Qin, and Yao Zhao.*

1. **Anomaly detection via graph attention networks-augmented mask autoregressive flow for multivariate time series.** IoT, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10453361)

   *Hao Liu, Wang Luo, Lixin Han, Peng Gao, Weiyong Yang, and Guangjie Han.*

1. **From chaos to clarity: Time series anomaly detection in astronomical observations.** arXiv, 2024. [paper](https://arxiv.org/abs/2403.10220)

   *Xinli Hao, Yile Chen, Chen Yang, Zhihui Du, Chaohong Ma, Chao Wu, and Xiaofeng Meng.*

1. **DACAD: Domain adaptation contrastive learning for anomaly detection in multivariate time series.** arXiv, 2024. [paper](https://arxiv.org/abs/2404.11269)

   *Zahra Zamanzadeh Darban, Geoffrey I. Webb, and Mahsa Salehi.*

1. **Variate associated domain adaptation for unsupervised multivariate time series anomaly detection.** TKDD, 2024. [paper](https://dl.acm.org/doi/10.1145/3663573)

   *Yifan He, Yatao Bian, Xi Ding, Bingzhe Wu, Jihong Guan, Ji Zhang, and Shuigeng Zhou.*

1. **Quo vadis, unsupervised time series anomaly detection?** ICML, 2024. [paper](https://arxiv.org/abs/2405.02678)

   *M. Saquib Sarfraz, Meiyen Chen, Lukas Layer, Kunyu Peng, and Marios Koulakis.*

1. **SiET: Spatial information enhanced transformer for multivariate time series anomaly detection.** KBS, 2024. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705124005628)

   *Weixuan Xiong, Peng Wang, Xiaochen Sun, and Jun Wang.*

1. **Disentangled anomaly detection for multivariate time seriesn.** WWW, 2024. [paper](https://dl.acm.org/doi/abs/10.1145/3589335.3651492)

   *Xin Jie, Xixi Zhou, Chanfei Su, Zijun Zhou, Yuqing Yuan, Jiajun Bu, and Haishuai Wang.*

1. **PATE: Proximity-aware time series anomaly evaluation.** KDD, 2024. [paper](https://arxiv.org/abs/2405.12096)

   *Ramin Ghorbani, Marcel J.T. Reinders, and David M.J. Tax.*

1. **Large language models can be zero-shot anomaly detectors for time series?** KDD, 2024. [paper](https://arxiv.org/abs/2405.14755)

   *Sarah Alnegheimish, Linh Nguyen, Laure Berti-Equille, and Kalyan Veeramachaneni.*

1. **LARA: A light and anti-overfitting retraining approach for unsupervised time series anomaly detection.** WWW, 2024. [paper](https://dl.acm.org/doi/abs/10.1145/3589334.3645472)

   *Feiyi Chen, Zhen Qin, Mengchu Zhou, Yingying Zhang, Shuiguang Deng, Lunting Fan, Guansong Pang, and Qingsong Wen.*

1. **Variate associated domain adaptation for unsupervised multivariate time series anomaly detection.** TKDD, 2024. [paper](https://dl.acm.org/doi/abs/10.1145/3663573)

   *Yifan He, Yatao Bian, Xi Ding, Bingzhe Wu, Jihong Guan, Ji Zhang, and Shuigeng Zhou.*

1. **Uni-directional graph structure learning-based multivariate time series anomaly detection with dynamic prior knowledge.** International Journal of Machine Learning and Cybernetics, 2024. [paper](https://link.springer.com/article/10.1007/s13042-024-02212-5)

   *Shiming He, Genxin Li, Jin Wang, Kun Xie, and Pradip Kumar Sharma.*

1. **Towards a general time series anomaly detector with adaptive bottlenecks and dual adversarial decoders.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.15273)

   *Qichao Shentu, Beibu Li, Kai Zhao, Yang shu, Zhongwen Rao, Lujia Pan, Bin Yang, and Chenjuan Guo.*

1. **USD: Unsupervised soft contrastive learning for fault detection in multivariate time series.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.16258)

   *Hong Liu, Xiuxiu Qiu, Yiming Shi, and Zelin Zang.*

#### [Tabular]
1. **Beyond individual input for deep anomaly detection on tabular data.** arXiv, 2023. [paper](https://arxiv.org/abs/2305.15121)

   *Hugo Thimonier, Fabrice Popineau, Arpad Rimmel, and Bich-Liên Doan.* 

1. **Fascinating supervisory signals and where to find them: Deep anomaly detection with scale learning.** ICML, 2023. [paper](https://arxiv.org/abs/2305.16114)

   *Hongzuo Xu, Yijie Wang, Juhui Wei, Songlei Jian, Yizhou Li, and Ning Liu.* 

1. **TabADM: Unsupervised tabular anomaly detection with diffusion models.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.12336)

   *Guy Zamberg, Moshe Salhov, Ofir Lindenbaum, and Amir Averbuch.* 

1. **ATDAD: One-class adversarial learning for tabular data anomaly detection.** Computers & Security, 2023. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0167404823003590)

   *Xiaohui Yang and Xiang Li.* 

1. **Understanding the limitations of self-supervised learning for tabular anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2309.08374)

   *Kimberly T. Mai, Toby Davies, and Lewis D. Griffin.* 

1. **Unmasking the chameleons: A benchmark for out-of-distribution detection in medical tabular data.** arXiv, 2023. [paper](https://arxiv.org/abs/2309.16220)

   *Mohammad Azizmalayeri, Ameen Abu-Hanna, and Giovanni Ciná.* 

1. **TDeLTA: A light-weight and robust table detection method based on learning text arrangement.** AAAI, 2024. [paper](https://arxiv.org/abs/2309.16220)

   *Yang Fan, Xiangping Wu, Qingcai Chen, Heng Li, Yan Huang, Zhixiang Cai, and Qitian Wu.* 

1. **How to overcome curse-of-dimensionality for out-of-distribution detection?** AAAI, 2024. [paper](https://arxiv.org/abs/2312.11043)

   *Soumya Suvra Ghosal, Yiyou Sun, and Yixuan Li.* 

1. **MCM: Masked cell modeling for anomaly detection in tabular data.** ICLR, 2024. [paper](https://openreview.net/forum?id=lNZJyEDxy4)

   *Anonymous authors.*
   

#### [Large Model]
1. **WinCLIP: Zero-/few-shot anomaly classification and segmentation.** CVPR, 2023. [paper](https://arxiv.org/abs/2303.05047)

   *Jongheon Jeong, Yang Zou, Taewan Kim, Dongqing Zhang, Avinash Ravichandran, and Onkar Dabeer.* 

1. **Semantic anomaly detection with large language models.** arXiv, 2023. [paper](https://arxiv.org/abs/2305.11307)

   *Amine Elhafsi, Rohan Sinha, Christopher Agia, Edward Schmerling, Issa Nesnas, and Marco Pavone.* 

1. **AnomalyGPT: Detecting industrial anomalies using large vision-language models.** arXiv, 2023. [paper](https://arxiv.org/abs/2308.15366)

   *Zhaopeng Gu, Bingke Zhu, Guibo Zhu, Yingying Chen, Ming Tang, and Jinqiao Wang.* 

1. **AnoVL: Adapting vision-language models for unified zero-shot anomaly localization.** arXiv, 2023. [paper](https://arxiv.org/abs/2308.15939)

   *Hanqiu Deng, Zhaoxiang Zhang, Jinan Bao, and Xingyu Li.* 

1. **LogGPT: Exploring ChatGPT for log-based anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2309.01189)

   *Jiaxing Qi, Shaohan Huang, Zhongzhi Luan, Carol Fung, Hailong Yang, and Depei Qian.* 

1. **CLIPN for zero-shot OOD detection: Teaching CLIP to say no.** ICCV, 2023. [paper](https://arxiv.org/abs/2308.12213)

   *Hualiang Wang, Yi Li, Huifeng Yao, and Xiaomeng Li.* 

1. **LogGPT: Log anomaly detection via GPT.** arXiv, 2023. [paper](https://arxiv.org/abs/2309.14482)

   *Xiao Han, Shuhan Yuan, and Mohamed Trabelsi.* 

1. **Semantic scene difference detection in daily life patroling by mobile robots using pre-trained large-scale vision-language model.** IROS, 2023. [paper](https://arxiv.org/abs/2309.16552)

   *Yoshiki Obinata, Kento Kawaharazuka, Naoaki Kanazawa, Naoya Yamaguchi, Naoto Tsukamoto, Iori Yanokura, Shingo Kitagawa, Koki Shinjo, Kei Okada, and Masayuki Inaba.* 

1. **HuntGPT: Integrating machine learning-based anomaly detection and explainable AI with large language models (LLMs).** arXiv, 2023. [paper](https://arxiv.org/abs/2309.16021)

   *Tarek Ali and Panos Kostakos.* 

1. **Graph neural architecture search with GPT-4.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.01436)

   *Haishuai Wang, Yang Gao, Xin Zheng, Peng Zhang, Hongyang Chen, and Jiajun Bu.* 

1. **Exploring large language models for multi-modal out-of-distribution detection.** EMNLP, 2023. [paper](https://arxiv.org/abs/2310.08027)

   *Yi Dai, Hao Lang, Kaisheng Zeng, Fei Huang, and Yongbin Li.* 

1. **Detecting pretraining data from large language models.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.16789)

   *Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, and Luke Zettlemoyer.* 

1. **AnomalyCLIP: Object-agnostic prompt learning for zero-shot anomaly detection.** ICLR, 2024. [paper](https://openreview.net/forum?id=buC4E91xZE)

   *Qihang Zhou, Guansong Pang, Yu Tian, Shibo He, and Jiming Chen.* 

1. **CLIP-AD: A language-guided staged dual-path model for zero-shot anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.00453)

   *Xuhai Chen, Jiangning Zhang, Guanzhong Tian, Haoyang He, Wuhao Zhang, Yabiao Wang, Chengjie Wang, Yunsheng Wu, and Yong Liu.* 

1. **Exploring grounding potential of VQA-oriented GPT-4V for zero-shot anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.02612)

   *Jiangning Zhang, Xuhai Chen, Zhucun Xue, Yabiao Wang, Chengjie Wang, and Yong Liu.* 

1. **Open-vocabulary video anomaly detection.** arXiv, 2023. [paper](https://arxiv.org/abs/2311.07042)

   *Peng Wu, Xuerong Zhou, Guansong Pang, Yujia Sun, Jing Liu, Peng Wang, and Yanning Zhang.* 

1. **Distilling out-of-distribution robustness from vision-language foundation models.** NIPS, 2023. [paper](https://neurips.cc/virtual/2023/poster/70716)

   *Andy Zhou, Jindong Wang, Yuxiong Wang, and Haohan Wang.* 

1. **Weakly supervised detection of gallucinations in LLM activations.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.02798)

   *Miriam Rateike, Celia Cintas, John Wamburu, Tanya Akumu, and Skyler Speakman.* 

1. **How well does GPT-4V(ision) adapt to distribution shifts? A preliminary investigation.** arXiv, 2023. [paper](https://arxiv.org/abs/2312.07424)

   *Zhongyi Han, Guanglin Zhou, Rundong He, Jindong Wang, Xing Xie, Tailin Wu, Yilong Yin, Salman Khan, Lina Yao, Tongliang Liu, and Kun Zhang.* 

1. **overlooked video classification in weakly supervised video anomaly detection.** WACV, 2024. [paper](https://openaccess.thecvf.com/content/WACV2024W/RWS/html/Tan_Overlooked_Video_Classification_in_Weakly_Supervised_Video_Anomaly_Detection_WACVW_2024_paper.html)

   *Weijun Tan, Qi Yao, and Jingfeng Liu.* 

1. **Video anomaly detection and explanation via large language models.** arXiv, 2024. [paper](https://arxiv.org/abs/2401.05702)

   *Hui Lv and Qianru Sun.* 

1. **OVOR: OnePrompt with virtual outlier regularization for rehearsal-free class-incremental learning.** ICLR, 2024. [paper](https://openreview.net/forum?id=FbuyDzZTPt)

   *Weicheng Huang, Chunfu Chen, and Hsiang Hsu.* 

1. **Large language model guided knowledge distillation for time series anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2401.15123v1)

   *Chen Liu, Shibo He, Qihang Zhou, Shizhong Li, and Wenchao Meng.* 

1. **Toward generalist anomaly detection via in-context residual learning with few-shot sample prompts.** CVPR, 2024. [paper](https://arxiv.org/abs/2403.06495)

   *Jiawen Zhu and Guansong Pang.* 

1. **Adapting visual-language models for generalizable anomaly detection in medical images.** CVPR, 2024. [paper](https://arxiv.org/abs/2403.12570v1)

   *Chaoqin Huang, Aofan Jiang, Jinghao Feng, Ya Zhang, Xinchao Wang, and Yanfeng Wang.* 

1. **Harnessing large language models for training-free video anomaly detection.** CVPR, 2024. [paper](https://arxiv.org/abs/2404.01014)

   *Luca Zanella, Willi Menapace, Massimiliano Mancini, Yiming Wang, and Elisa Ricci.* 

1. **Collaborative learning of anomalies with privacy (CLAP) for unsupervised video anomaly detection: A new baseline.** CVPR, 2024. [paper](https://arxiv.org/abs/2404.00847)

   *Anas Al-lahham, Muhammad Zaigham Zaheer, Nurbek Tastan, and Karthik Nandakumar.* 

1. **PromptAD: Learning prompts with only normal samples for few-shot anomaly detection.** CVPR, 2024. [paper](https://arxiv.org/abs/2404.05231)

   *Xiaofan Li, Zhizhong Zhang, Xin Tan, Chengwei Chen, Yanyun Qu, Yuan Xie, and Lizhuang Ma.* 

1. **Dynamic distinction learning: Adaptive pseudo anomalies for video anomaly detection.** CVPR, 2024. [paper](https://arxiv.org/abs/2404.04986)

   *Demetris Lappas, Vasileios Argyriou, and Dimitrios Makris.* 

1. **Your finetuned large language model is already a powerful out-of-distribution detector.** arXiv, 2024. [paper](https://arxiv.org/abs/2404.08679)

   *Andi Zhang, Tim Z. Xiao, Weiyang Liu, Robert Bamler, and Damon Wischik.* 

1. **Do LLMs understand visual anomalies? Uncovering LLM capabilities in zero-shot anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2404.09654)

   *Jiaqi Zhu, Shaofeng Cai, Fang Deng, and Junran Wu.* 

1. **Text prompt with normality guidance for weakly supervised video anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2404.08531)

   *Zhiwei Yang, Jing Liu, and Peng Wu.* 

1. **FiLo: Zero-shot anomaly detection by fine-grained description and high-quality localization.** arXiv, 2024. [paper](https://arxiv.org/abs/2404.08531)

   *Zhaopeng Gu, Bingke Zhu, Guibo Zhu, Yingying Chen, Hao Li, Ming Tang, and Jinqiao Wang.* 

1. **AnomalyDINO: Boosting patch-based few-shot anomaly detection with DINOv2.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.14529)

   *Simon Damm, Mike Laszkiewicz, Johannes Lederer, and Asja Fischer.* 

1. **Large language models can deliver accurate and interpretable time series anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.15370)

   *Jiaqi Tang, Hao Lu, Ruizheng Wu, Xiaogang Xu, Ke Ma, Cheng Fang, Bin Guo, Jiangbo Lu, Qifeng Chen, and Ying-Cong Chen.* 

1. **Hawk: Learning to understand open-world video anomalies.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.15370)

   *Jun Liu, Chaoyun Zhang, Jiaxu Qian, Minghua Ma, Si Qin, Chetan Bansal, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang.* 

1. **ARC: A generalist graph anomaly detector with in-context learning.** arXiv, 2024. [paper](https://arxiv.org/abs/2405.16771)

   *Yixin Liu, Shiyuan Li, Yu Zheng, Qingfeng Chen, Chengqi Zhang, and Shirui Pan.* 

#### [Reinforcement Learning]
1. **Towards experienced anomaly detector through reinforcement learning.** AAAI, 2018. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/12130)

   *Chengqiang Huang, Yulei Wu, Yuan Zuo, Ke Pei, and Geyong Min.* 

1. **Sequential anomaly detection using inverse reinforcement learning.** KDD, 2019. [paper](https://dl.acm.org/doi/10.1145/3292500.3330932)

   *Min-hwan Oh and Garud Iyengar.* 

1. **Toward deep supervised anomaly detection: Reinforcement learning from partially labeled anomaly data.** KDD, 2021. [paper](https://dl.acm.org/doi/10.1145/3447548.3467417)

   *Guansong Pang, Anton van den Hengel, Chunhua Shen, and Longbing Cao.* 

1. **Automated anomaly detection via curiosity-guided search and self-imitation learning.** TNNLS, 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9526875)

   *Yuening Li, Zhengzhang Chen, Daochen Zha, Kaixiong Zhou, Haifeng Jin, Haifeng Chen, and Xia Hu.* 

1. **Meta-AAD: Active anomaly detection with deep reinforcement learning.** ICDM, 2020. [paper](https://ieeexplore.ieee.org/document/9338270)

   *Daochen Zha, Kwei-Herng Lai, Mingyang Wan, and Xia Hu.* 

1. **Semi-supervised learning via DQN for log anomaly detection.** arXiv, 2024. [paper](https://arxiv.org/abs/2401.03151)

   *Yingying He, Xiaobing Pei, and Lihong Shen.* 

1. **OIL-AD: An anomaly detection framework for sequential decision sequences.** arXiv, 2024. [paper](https://arxiv.org/abs/2402.04567)

   *Chen Wang, Sarah Erfani, Tansu Alpcan, and Christopher Leckie.* 

### [Representation Learning]
1. **Localizing anomalies from weakly-labeled videos.** TIP, 2021. [paper](https://ieeexplore.ieee.org/document/9408419)

   *Hui Lv, Chuanwei Zhou, Zhen Cui, Chunyan Xu, Yong Li, and Jian Yang.* 

1. **PAC-Wrap: Semi-supervised PAC anomaly detection.** KDD, 2022. [paper](https://arxiv.org/abs/2205.10798)

   *Shuo Li, Xiayan Ji, Edgar Dobriban, Oleg Sokolsky, and Insup Lee.* 

1. **Effective end-to-end unsupervised outlier detection via inlier priority of discriminative network.** NIPS, 2019. [paper](https://proceedings.neurips.cc/paper/2019/hash/6c4bb406b3e7cd5447f7a76fd7008806-Abstract.html)

   *Siqi Wang, Yijie Zeng, Xinwang Liu, En Zhu, Jianping Yin, Chuanfu Xu, and Marius Kloft.* 

1. **AnomalyHop: An SSL-based image anomaly localization method.** ICVCIP, 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9675385)

   *Kaitai Zhang, Bin Wang, Wei Wang, Fahad Sohrab, Moncef Gabbouj, and C.-C. Jay Kuo.* 

1. **Learning representations of ultrahigh-dimensional data for random distance-based outlier detection.** KDD, 2018. [paper](https://dl.acm.org/doi/abs/10.1145/3219819.3220042)

   *Guansong Pang, Longbing Cao, Ling Chen, and Huan Liu.* 

1. **Federated disentangled representation learning for unsupervised brain anomaly detection.** NMI, 2022. [paper](https://www.nature.com/articles/s42256-022-00515-2)

   *Cosmin I. Bercea, Benedikt Wiestler, Daniel Rueckert, and Shadi Albarqouni.* 

1. **DSR–A dual subspace re-projection network for surface anomaly detection.** ECCV, 2022. [paper](https://link.springer.com/chapter/10.1007/978-3-031-19821-2_31)

   *Vitjan Zavrtanik, Matej Kristan, and Danijel Skočaj.* 

1. **LGN-Net: Local-global normality network for video anomaly detection.** arXiv, 2022. [paper](https://arxiv.org/abs/2211.07454)

   *Mengyang Zhao, Yang Liu, Jing Liu, Di Li, and Xinhua Zeng.* 

1. **Glancing at the patch: Anomaly localization with global and local feature comparison.** CVPR, 2021. [paper](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Glancing_at_the_Patch_Anomaly_Localization_With_Global_and_Local_CVPR_2021_paper.html)

   *Shenzhi Wang, Liwei Wu, Lei Cui, and Yujun Shen.* 

1. **SPot-the-difference self-supervised pre-training for anomaly detection and segmentation.** ECCV, 2022. [paper](https://link.springer.com/chapter/10.1007/978-3-031-20056-4_23)

   *Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, and Onkar Dabeer.* 

1. **SSD: A unified framework for self-supervised outlier detection.** ICLR, 2021. [paper](https://openreview.net/forum?id=v5gjXpmR8J)

   *Vikash Sehwag, Mung Chiang, and Prateek Mittal.* 

1. **NETS: Extremely fast outlier detection from a data stream via set-based processing.** VLDB, 2019. [paper](https://openreview.net/forum?id=v5gjXpmR8J)

   *Susik Yoon, Jae-Gil Lee, and Byung Suk Lee.* 

1. **XGBOD: Improving supervised outlier detection with unsupervised representation learning.** IJCNN, 2018. [paper](https://ieeexplore.ieee.org/abstract/document/8489605)

   *Yue Zhao and Maciej K. Hryniewicki.* 

1. **Red PANDA: Disambiguating anomaly detection by removing nuisance factors.** ICLR, 2023. [paper](https://openreview.net/forum?id=z37tDDHHgi)

   *Niv Cohen, Jonathan Kahana, and Yedid Hoshen.* 

1. **TimesNet: Temporal 2D-variation modeling for general time series analysis.** ICLR, 2023. [paper](https://openreview.net/forum?id=ju_Uqw384Oq)

   *Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long.* 

1. **SimpleNet: A simple network for image anomaly detection and localization.** CVPR, 2023. [paper](https://arxiv.org/abs/2303.15140)

   *Zhikang Liu, Yiming Zhou, Yuansheng Xu, and Zilei Wang.* 

1. **Unsupervised anomaly detection via nonlinear manifold learning.** arXiv, 2023. [paper](https://arxiv.org/abs/2306.09441)

   *Amin Yousefpour, Mehdi Shishehbor, Zahra Zanjani Foumani, and Ramin Bostanabad.* 

1. **Representation learning in anomaly detection: Successes, limits and a grand challenge.** arXiv, 2023. [paper](https://arxiv.org/abs/2307.11085v1)

   *Yedid Hoshen.* 

1. **A lightweight video anomaly detection model with weak supervision and adaptive instance selection.** arXiv, 2023. [paper](https://arxiv.org/abs/2310.05330)

   *Yang Wang, Jiaogen Zhou, and Jihong Guan.* 

1. **MGFN: Magnitude-contrastive glance-and-focus network for weakly-supervised video anomaly detection.** AAAI, 2023. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/25112)

   *Yingxian Chen, Zhengzhe Liu, Baoheng Zhang, Wilton Fok, Xiaojuan Qi, and Yik-Chung Wu.* 

1. **TeD-SPAD: Temporal distinctiveness for self-supervised privacy-preservation for video anomaly detection.** ICCV, 2023. [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Fioresi_TeD-SPAD_Temporal_Distinctiveness_for_Self-Supervised_Privacy-Preservation_for_Video_Anomaly_Detection_ICCV_2023_paper.html)

   *Joseph Fioresi, Ishan Rajendrakumar Dave, and Mubarak Shah.* 

1. **Deep orthogonal hypersphere compression for anomaly detection.** ICLR, 2024. [paper](https://openreview.net/forum?id=cJs4oE4m9Q)

   *Yunhe Zhang, Yan Sun, Jinyu Cai, and Jicong Fan.* 

1. **VI-OOD: A unified representation learning framework for textual out-of-distribution detection.** COLING, 2024. [paper](https://arxiv.org/abs/2404.06217)

   *Yunhe Zhang, Yan Sun, Jinyu Cai, and Jicong Fan.* 

#### [Nonparametric Approach]
1. **Real-time nonparametric anomaly detection in high-dimensional settings.** TPAMI, 2021. [paper](https://ieeexplore.ieee.org/abstract/document/8976215/)

   *Mehmet Necip Kurt, Yasin Yılmaz, and Xiaodong Wang.* 

1. **Neighborhood structure assisted non-negative matrix factorization and its application in unsupervised point anomaly detection.** JMLR, 2021. [paper](https://dl.acm.org/doi/abs/10.5555/3546258.3546292)

   *Imtiaz Ahmed, Xia Ben Hu, Mithun P. Acharya, and Yu Ding.* 

1. **Bayesian nonparametric submodular video partition for robust anomaly detection.** CVPR, 2022. [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Sapkota_Bayesian_Nonparametric_Submodular_Video_Partition_for_Robust_Anomaly_Detection_CVPR_2022_paper.html)

   *Hitesh Sapkota and Qi Yu.* 

1. **Making parametric anomaly detection on tabular data non-parametric again.** arXiv, 2024. [paper](https://arxiv.org/abs/2401.17052)

   *Hugo Thimonier, Fabrice Popineau, Arpad Rimmel, and Bich-Liên Doan.* 


