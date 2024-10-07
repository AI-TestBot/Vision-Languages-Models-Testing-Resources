# Vision Languages Models (VLMs) Testing Resources

## 📒Introduction
Vision Languages Models (VLMs) Testing Resources: A curated list of Awesome VLMs Testing Papers with Codes, check [📖Contents](#paperlist) for more details. This repo is still updated frequently ~ 👨‍💻‍ **Welcome to star ⭐️ or submit a PR to this repo! I will review and merge it.**

## 📖Contents 
* 📖[Review](#Review)
* 📖[General](#General)
  * 📖[Comprehensive](#Comprehensive)
  * 📖[Understanding](#Understanding)
    * 📖[Image](#Image)
    * 📖[Video](#Video)
  * 📖[Generation](#Generation)
    * 📖[Text-to-Image](#Text-to-Image)
    * 📖[Text-to-Video](#Text-to-Video)
  * 📖[VQA](#VQA)
  * 📖[Reasoning](#Reasoning)
  * 📖[Multilingual](#Multilingual)
  * 📖[Instruction-Following](#Instruction-Following)
  * 📖[High-Level-Vision](#High-Level-Vision)
    * 📖[OCR](#OCR)
    * 📖[Aesthetics](#Aesthetics)
  * 📖[Low-Level-Vision](#Low-Level-Vision)
  * 📖[Reliable](#Reliable)
  * 📖[Robust](#Robust)
* 📖[Security](#Security)
* 📖[Application](#Application)
    * 📖[Search](#Search)
    * 📖[Agent](#Agent)
* 📖[Industry](#Industry)
    * 📖[Medical](#Medical)
* 📖[Human-Machine-Interaction](#Human-Machine-Interaction)
* 📖[Testing-Methods](#Testing-Methods)
* 📖[Testing-Tools](#Testing-Tools)
* 📖[Challenges](#Challenges)

## 📖Review  

**From GPT-4 to Gemini and Beyond: Assessing the Landscape of MLLMs on Generalizability, Trustworthiness and Causality through Four Modalities.**<br>
*C Lu, C Qian, G Zheng, H Fan, H Gao, J Zhang, J Shao, J Deng, J Fu, K Huang, K Li, L Li, et al.*<br>
ArXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2401.15071.pdf)]
[[Github](https://openlamm.github.io/Leaderboards)]

**A Survey on Benchmarks of Multimodal Large Language Models.**<br>
*J Li, W Lu.*<br>
ArXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2408.08632)]
[[Github](https://github.com/swordlidev/Evaluation-Multimodal-LLMs-Survey)]

**A Survey on Evaluation of Multimodal Large Language Models.**<br>
*J Huang, J Zhang.*<br>
arxiv:2408.15769, 2024.
[[ArXiv](https://arxiv.org/pdf/2408.15769)]

## 📖General 

### Comprehensive

**Holistic evaluation of text-to-image models.**<br>
*T Lee, M Yasunaga, C Meng, Y Mai, JS Park, A Gupta, Y Zhang, D Narayanan, H Teufel, et al.*<br>
Advances in Neural Information Processing Systems, 2024.
[[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/dd83eada2c3c74db3c7fe1c087513756-Paper-Datasets_and_Benchmarks.pdf)]
[[Github](https://github.com/stanford-crfm/helm)]

**MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models.**<br>
*Fu, Chaoyou and Chen, Peixian and Shen, Yunhang and Qin, Yulei and Zhang, Mengdan and Lin, Xu and Yang, Jinrui and Zheng, Xiawu and Li, Ke and Sun, Xing and others.*<br>
arXiv:2306.13394, 2023.

**MME-RealWorld: Could Your Multimodal LLM Challenge High-Resolution Real-World Scenarios that are Difficult for Humans?**<br>
*YF Zhang, H Zhang, H Tian, C Fu, S Zhang, J Wu, F Li, K Wang, Q Wen, Z Zhang, L Wang, et al.*<br>
arXiv:2408.13257, 2024.
[[ArXiv](https://arxiv.org/pdf/2408.13257)]

**Lvlm-ehub: A comprehensive evaluation benchmark for large vision-language models.**<br>
*P Xu, W Shao, K Zhang, P Gao, S Liu, M Lei, F Meng, S Huang, Y Qiao, P Luo.*<br>
arXiv:2306.09265, 2023.
[[ArXiv](https://arxiv.org/pdf/2306.09265)]

**Mmbench: Is your multi-modal model an all-around player?**<br>
*Y Liu, H Duan, Y Zhang, B Li, S Zhang, W Zhao, Y Yuan, J Wang, C He, Z Liu, K Chen, D Lin.*<br>
arXiv:2307.06281, 2023.
[[ArXiv](https://arxiv.org/pdf/2307.06281.pdf)]
[[Github](https://mmbench.opencompass.org.cn/home)]

**Mm-vet: Evaluating large multimodal models for integrated capabilities.**<br>
*W Yu, Z Yang, L Li, J Wang, K Lin, Z Liu, X Wang, L Wang.*<br>
arXiv:2308.02490, 2023.
[[ArXiv](https://arxiv.org/pdf/2308.02490)]
[[Github](https://github.com/yuweihao/MM-Vet)]

**mplug-owl: Modularization empowers large language models with multimodality.**<br>
*Q Ye, H Xu, G Xu, J Ye, M Yan, Y Zhou, J Wang, A Hu, P Shi, Y Shi, C Li, Y Xu, H Chen, et al.*<br>
arXiv:2304.14178, 2023.
[[ArXiv](https://arxiv.org/pdf/2304.14178)]

**[Seed-bench] Seed-bench: Benchmarking multimodal llms with generative comprehension.**<br>
*B Li, R Wang, G Wang, Y Ge, Y Ge, Y Shan.*<br>
arXiv:2307.16125, 2023.
[[ArXiv](https://arxiv.org/pdf/2307.16125)]
[[Github](https://github.com/AILab-CVC/SEED-Bench)]

**Touchstone: Evaluating vision-language models by language models.**<br>
*S Bai, S Yang, J Bai, P Wang, X Zhang, J Lin, X Wang, C Zhou, J Zhou.*<br>
arXiv:2308.16890, 2023.
[[ArXiv](https://arxiv.org/pdf/2308.16890)]
[[Github](https://github.com/OFA-Sys/TouchStone)]

**Visit-bench: A benchmark for vision-language instruction following inspired by real-world use.**<br>
*Y Bitton, H Bansal, J Hessel, R Shao, W Zhu, A Awadalla, J Gardner, R Taori, L Schimdt.*<br>
arXiv:2308.06595, 2023.
[[ArXiv](https://arxiv.org/pdf/2308.06595)]
[[Github](https://visit-bench.github.io/)]

**Towards an Exhaustive Evaluation of Vision-Language Foundation Models.**<br>
*E Salin, S Ayache, B Favre.*<br>
ICCV, 2023.
[[Paper](https://openaccess.thecvf.com/content/ICCV2023W/MMFM/papers/Salin_Towards_an_Exhaustive_Evaluation_of_Vision-Language_Foundation_Models_ICCVW_2023_paper.pdf)]

**[MMMU] Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi.**<br>
*X Yue, Y Ni, K Zhang, T Zheng, R Liu, G Zhang, S Stevens, D Jiang, W Ren, Y Sun, C Wei, et al.*<br>
CVPR, 2024.
[[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Yue_MMMU_A_Massive_Multi-discipline_Multimodal_Understanding_and_Reasoning_Benchmark_for_CVPR_2024_paper.pdf)]
[[Github](https://mmmu-benchmark.github.io/)]

**[Vbench] Vbench: Comprehensive benchmark suite for video generative models.**<br>
*Z Huang, Y He, J Yu, F Zhang, C Si, Y Jiang, Y Zhang, T Wu, Q Jin, N Chanpaisit, Y Wang, et al.*<br>
CVPR, 2024.
[[ArXiv](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.pdf)]
[[Github](https://vchitect.github.io/VBench-project/)]

**Beyond task performance: Evaluating and reducing the flaws of large multimodal models with in-context learning.**<br>
*M Shukor, A Rame, C Dancette, M Cord.*<br>
ICLR, 2024.
[[ArXiv](https://arxiv.org/pdf/2310.00647)]
[[Github](https://github.com/mshukor/EvALign-ICL)]

**[HR-Bench] Divide, Conquer and Combine: A Training-Free Framework for High-Resolution Image Perception in Multimodal Large Language Models.**<br>
*W Wang, L Ding, M Zeng, X Zhou, L Shen, Y Luo, D Tao.*<br>
ArXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2408.15556)]
[[Github](https://github.com/DreamMr/HR-Bench)]

**[MuirBench] MuirBench: A Comprehensive Benchmark for Robust Multi-image Understanding.**<br>
*F Wang, X Fu, JY Huang, Z Li, Q Liu, X Liu, MD Ma, N Xu, W Zhou, K Zhang, TL Yan, WJ Mo, et al.*<br>
arXiv:2406.09411, 2024.
[[ArXiv](https://arxiv.org/pdf/2406.09411)]
[[Github](https://github.com/muirbench/MuirBench)]
[[HuggingFace](https://huggingface.co/datasets/MUIRBENCH/MUIRBENCH)]

**[Mmt-bench] Mmt-bench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask agi.**<br>
*K Ying, F Meng, J Wang, Z Li, H Lin, Y Yang, et al.*<br>
arXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2404.16006)]
[[Github](https://mmt-bench.github.io/)]
[[HuggingFace](https://huggingface.co/datasets/Kaining/MMT-Bench)]

### Understanding

|Date|Task|Title|Paper|HomePage|Github|DataSets|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|2023| Content | MM-BigBench: Evaluating Multimodal Models on Multimodal Content Comprehension Tasks.|[[ArXiv]](https://arxiv.org/pdf/2310.09036) |-|[[Github]](https://github.com/declare-lab/MM-InstructEval)|-|
|2024| Dialog | [MMDU] MMDU: A Multi-Turn Multi-Image Dialog Understanding Benchmark and Instruction-Tuning Dataset for LVLM.|[[ArXiv]](https://arxiv.org/pdf/2406.11833) |-|[[Github]](https://github.com/Liuziyu77/MMDU/)|-|
|2023| Image | Journeydb: A benchmark for generative image understanding.|[[NeurIPS]](https://proceedings.neurips.cc/paper_files/paper/2023/file/9bc59aff4685e39e1a8175d5303248a1-Paper-Datasets_and_Benchmarks.pdf) |-|[[Github]](https://journeydb.github.io/)|-|
|2024| Image | MMIU: Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models.|[[ArXiv]](https://arxiv.org/pdf/2408.02718) |-|[[Github]](https://mmiu-bench.github.io/)|-|
|2024| Image | MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations.|[[ArXiv]](https://arxiv.org/pdf/2407.01523) |-|[[Github]](https://github.com/mayubo2333/MMLongBench-Doc)|-|
|2024| Relation | [CRPE] The all-seeing project v2: Towards general relation comprehension of the open world.|[[ArXiv]](https://arxiv.org/pdf/2402.19474) |-|[[Github]](https://github.com/OpenGVLab/all-seeing)|[[HuggingFace]](https://huggingface.co/datasets/OpenGVLab/CRPE)|
|2024| Video | MVBench: A Comprehensive Multi-modal Video Understanding Benchmark.|[[CVPR]](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_MVBench_A_Comprehensive_Multi-modal_Video_Understanding_Benchmark_CVPR_2024_paper.pdf) |-|[[Github]](https://github.com/OpenGVLab/Ask-Anything)|-|
|2024| Video | MLVU: A Comprehensive Benchmark for Multi-Task Long Video Understanding.|[[ArXiv]](https://arxiv.org/pdf/2406.04264) |-|[[Github]](https://github.com/JUNJIE99/MLVU)|-|

### Generation

#### Text-to-Image

|Date|Task|Title|Paper|HomePage|Github|DataSets|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|2023| Text-to-Image | Let's ViCE! Mimicking Human Cognitive Behavior in Image Generation Evaluation.|[[ACMMM]](https://dl.acm.org/doi/pdf/10.1145/3581783.3612706) |-|-|-|
|2023| Text-to-Image | Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis.|[[ArXiv]](https://arxiv.org/pdf/2306.09341) |-|[[Github]](https://github.com/tgxs002/HPSv2)|-|
|2023| Text-to-Image | Pku-i2iqa: An image-to-image quality assessment database for ai generated images.|[[ArXiv]](https://arxiv.org/pdf/2311.15556) |-|[[Github]](https://github.com/jiquan123/I2IQA)|-|
|2023| Text-to-Image | Toward verifiable and reproducible human evaluation for text-to-image generation.|[[CVPR]](https://openaccess.thecvf.com/content/CVPR2023/papers/Otani_Toward_Verifiable_and_Reproducible_Human_Evaluation_for_Text-to-Image_Generation_CVPR_2023_paper.pdf) |-|-|-|
|2023| Text-to-Image | Hrs-bench: Holistic, reliable and scalable benchmark for text-to-image models.|[[ICCV]](https://openaccess.thecvf.com/content/ICCV2023/papers/Bakr_HRS-Bench_Holistic_Reliable_and_Scalable_Benchmark_for_Text-to-Image_Models_ICCV_2023_paper.pdf) |-|[[Github]](https://eslambakr.github.io/hrsbench.github.io/)|-|
|2023| Text-to-Image | T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation.|[[NeurIPS]](https://proceedings.neurips.cc/paper_files/paper/2023/file/f8ad010cdd9143dbb0e9308c093aff24-Paper-Datasets_and_Benchmarks.pdf) |-|[[Github]](https://karine-h.github.io/T2I-CompBench/)|-|
|2023| Text-to-Image | Agiqa-3k: An open database for ai-generated image quality assessment.|[[TCSVT]](https://ieeexplore.ieee.org/document/10262331/?denied=) |-|[[Github]](https://github.com/lcysyzxdxc/AGIQA-3k-Database)|-|
|2024| Text-to-Image | Evaluating Text to Image Synthesis: Survey and Taxonomy of Image Quality Metrics.|[[ArXiv]](https://arxiv.longhoe.net/pdf/2403.11821) |-|[[Github]](https://github.com/linzhiqiu/t2v_metrics)|-|
|2024| Text-to-Image | Evaluating text-to-visual generation with image-to-text generation.|[[ArXiv]](https://arxiv.org/pdf/2404.01291) |-|-|-|
|2024| Text-to-Image | UNIAA: A Unified Multi-modal Image Aesthetic Assessment Baseline and Benchmark.|[[ArXiv]](https://arxiv.org/pdf/2404.09619) |-|-|-|
|2024| Text-to-Image | Aigiqa-20k: A large database for ai-generated image quality assessment.|[[CVPRW]](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Li_AIGIQA-20K_A_Large_Database_for_AI-Generated_Image_Quality_Assessment_CVPRW_2024_paper.pdf) |-|-|[[DataSets]](https://www.modelscope.cn/datasets/lcysyzxdxc/AIGCQA-30K-Image)|
|2024| Text-to-Image | Imagereward: Learning and evaluating human preferences for text-to-image generation.|[[NeurIPS]](https://proceedings.neurips.cc/paper_files/paper/2023/file/33646ef0ed554145eab65f6250fab0c9-Paper-Conference.pdf) |-|-|-|
|2024| Text-to-Image | Llmscore: Unveiling the power of large language models in text-to-image synthesis evaluation.|[[NeurIPS]](https://so2.cljtscd.com/scholar?hl=zh-CN&as_sdt=0%2C5&q=LLMScore%3A+Unveiling+the+Power+of+Large+Language+Models+in+Text-to-Image+Synthesis+Evaluation&btnG=) |-|[[Github]](https://github.com/YujieLu10/LLMScore)|-|
|2024| Text-to-Image | EVALALIGN: Supervised Fine-Tuning Multimodal LLMs with Human-Aligned Data for Evaluating Text-to-Image Models.|[[ArXiv]](https://arxiv.org/pdf/2406.16562) |-|[[Github]](https://github.com/SAIS-FUXI/EvalAlign)|-|
|2024| Text-to-Image | PhyBench: A Physical Commonsense Benchmark for Evaluating Text-to-Image Models.|[[ArXiv]](https://arxiv.org/pdf/2406.11802) |-|[[Github]](https://github.com/OpenGVLab/PhyBench)|-|
|2024| Text-to-Image | PTlTScore: Towards Long-Tail Effects in Text-to-Visual Evaluation with Generative Foundation Models.|[[CVPR]](https://openaccess.thecvf.com/content/CVPR2024W/EvGenFM/papers/Ji_TlTScore_Towards_Long-Tail_Effects_in_Text-to-Visual_Evaluation_with_Generative_Foundation_CVPRW_2024_paper.pdf) |-|-|-|
|2024| Text-to-Image | FlashEval: Towards Fast and Accurate Evaluation of Text-to-image Diffusion Generative Models.|[[CVPR]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_FlashEval_Towards_Fast_and_Accurate_Evaluation_of_Text-to-image_Diffusion_Generative_CVPR_2024_paper.pdf) |-|[[Github]](https://github.com/thu-nics/FlashEval)|-|

#### Text-to-Video
|Date|Task|Title|Paper|HomePage|Github|DataSets|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|2023| Text-to-Video | Fetv: A benchmark for fine-grained evaluation of open-domain text-to-video generation.|[[NeurIPS]](https://proceedings.neurips.cc/paper_files/paper/2023/file/c481049f7410f38e788f67c171c64ad5-Paper-Datasets_and_Benchmarks.pdf) |-|[[Github]](https://github.com/llyx97/FETV)|-|
|2024| Text-to-Video | AIGCBench: Comprehensive Evaluation of Image-to-Video Content Generated by AI.|[[ArXiv]](https://arxiv.org/pdf/2401.01651) |-|[[Github]](https://github.com/BenchCouncil/AIGCBench)|-|
|2024| Text-to-Video | Evaluation of Text-to-Video Generation Models: A Dynamics Perspective.|[[ArXiv]](https://arxiv.org/pdf/2407.01094) |-|[[Github]](https://github.com/MingXiangL/DEVILh)|-|
|2024| Text-to-Video | GAIA: Rethinking Action Quality Assessment for AI-Generated Videos.|[[ArXiv]](https://arxiv.org/pdf/2406.06087) |-|[[Github]](https://github.com/zijianchen98/GAIA)|-|
|2024| Text-to-Video | Subjective-Aligned Dateset and Metric for Text-to-Video Quality Assessment.|[[ArXiv]](https://arxiv.longhoe.net/pdf/2403.11956) |-|[[Github]](https://github.com/QMME/T2VQA)|-|
|2024| Text-to-Video | T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation.|[[ArXiv]](https://arxiv.org/pdf/2407.14505) |-|[[Github]](https://t2v-compbench.github.io/)|-|
|2024| Text-to-Video | TC-Bench: Benchmarking Temporal Compositionality in Text-to-Video and Image-to-Video Generation.|[[ArXiv]](https://arxiv.org/pdf/2406.08656) |-|[[Github]](https://weixi-feng.github.io/tc-bench/)|-|
|2024| Text-to-Video | VideoPhy: Evaluating Physical Commonsense for Video Generation.|[[ArXiv]](https://arxiv.org/pdf/2406.03520) |-|[[Github]](https://videophy.github.io/)|-|
|2024| Text-to-Video | MantisScore: Building Automatic Metrics to Simulate Fine-grained Human Feedback for Video Generation.|[[ArXiv]](https://arxiv.org/pdf/2406.15252) |-|[[Github]](https://tiger-ai-lab.github.io/VideoScore/)|-|
|2024| Text-to-Video | AIGC-VQA: A Holistic Perception Metric for AIGC Video Quality Assessment.|[[CVPR]](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Lu_AIGC-VQA_A_Holistic_Perception_Metric_for_AIGC_Video_Quality_Assessment_CVPRW_2024_paper.pdf) |-|-|-|
|2024| Text-to-Video | Evalcrafter: Benchmarking and evaluating large video generation models.|[[CVPR]](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_EvalCrafter_Benchmarking_and_Evaluating_Large_Video_Generation_Models_CVPR_2024_paper.pdf) |[[Homepage]](https://evalcrafter.github.io/)|[[Github]](https://github.com/EvalCrafter/EvalCrafter)|-|
|2024| Text-to-Video | T2VBench: Benchmarking Temporal Dynamics for Text-to-Video Generation.|[[CVPR]](https://openaccess.thecvf.com/content/CVPR2024W/EvGenFM/papers/Ji_T2VBench_Benchmarking_Temporal_Dynamics_for_Text-to-Video_Generation_CVPRW_2024_paper.pdf) |-|[[Github]](https://ji-pengliang.github.io/T2VBench/)|-|
|2024| Text-to-Video | Benchmarking AIGC Video Quality Assessment: A Dataset and Unified Model.|[[ArXiv]](https://arxiv.org/pdf/2407.21408) |-|-|-|

### VQA

**Vqa: Visual question answering.**<br>
*S Antol, A Agrawal, J Lu, M Mitchell, et al.*<br>
ICCV, 2015.
[[Paper](https://openaccess.thecvf.com/content_iccv_2015/papers/Antol_VQA_Visual_Question_ICCV_2015_paper.pdf)]
[[Homepage](https://visualqa.org/)]

**Making the v in vqa matter: Elevating the role of image understanding in visual question answering.**<br>
*Y Goyal, T Khot, D Summers-Stay, D Batra, D Parikh, et al.*<br>
CVPR, 2017.
[[Paper](https://openaccess.thecvf.com/content_iccv_2015/papers/Antol_VQA_Visual_Question_ICCV_2015_paper.pdf)]
[[Homepage](https://visualqa.org/)]

**Ok-vqa: A visual question answering benchmark requiring external knowledge.**<br>
*K Marino, M Rastegari, A Farhadi, R Mottaghi.*<br>
CVPR, 2019.
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Marino_OK-VQA_A_Visual_Question_Answering_Benchmark_Requiring_External_Knowledge_CVPR_2019_paper.pdf)]

**[TextVQA] Towards VQA Models That Can Read.**<br>
*A Singh, V Natarajan, M Shah, et al.*<br>
CVPR, 2019.
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Singh_Towards_VQA_Models_That_Can_Read_CVPR_2019_paper.pdf)]
[[Homepage](https://textvqa.org/)]

**[DocVQA] Docvqa: A dataset for vqa on document images.**<br>
*M Mathew, D Karatzas, CV Jawahar.*<br>
WACV, 2021.
[[Paper](https://openaccess.thecvf.com/content/WACV2021/papers/Mathew_DocVQA_A_Dataset_for_VQA_on_Document_Images_WACV_2021_paper.pdf)]
[[Homepage](https://textvqa.org/)]

**[ChartQA] ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasonin.**<br>
*A Masry, DX Long, JQ Tan, S Joty, E Hoque.*<br>
arxiv:2203.10244, 2022.
[[Paper](https://arxiv.org/pdf/2203.10244)]

**[ScienceQA] Learn to explain: Multimodal reasoning via thought chains for science question answering.**<br>
*P Lu, S Mishra, T Xia, L Qiu, KW Chang, SC Zhu, O Tafjord, P Clark, A Kalyan.*<br>
Advances in Neural Information Processing Systems, 2022.
[[Neurips](https://proceedings.neurips.cc/paper_files/paper/2022/file/11332b6b6cf4485b84afadb1352d3a9a-Paper-Conference.pdf)]
[[Github](https://scienceqa.github.io/)]

**KNVQA: A Benchmark for evaluation knowledge-based VQA.**<br>
*S Cheng, S Zhang, J Wu, M Lan.*<br>
arXiv:2311.12639, 2023.

**Maqa: A multimodal qa benchmark for negation.**<br>
*JY Li, A Jansen, Q Huang, J Lee, R Ganti, D Kuzmin.*<br>
arXiv:2301.03238, 2023.
[[ArXiv](https://arxiv.org/pdf/2301.03238)]

**Multimodal multi-hop question answering through a conversation between tools and efficiently finetuned large language models.**<br>
*H Rajabzadeh, S Wang, HJ Kwon, B Liu.*<br>
arXiv:2309.08922, 2023.
[[ArXiv](https://arxiv.org/pdf/2309.08922)]

**Scigraphqa: A large-scale synthetic multi-turn question-answering dataset for scientific graphs.**<br>
*S Li, N Tajbakhsh.*<br>
arXiv:2308.03349, 2023.
[[ArXiv](https://arxiv.org/pdf/2308.03349)]

**Slidevqa: A dataset for document visual question answering on multiple images.**<br>
*R Tanaka, K Nishida, K Nishida, T Hasegawa, I Saito, K Saito.*<br>
AAAI, 2023.
[[ArXiv](https://arxiv.org/pdf/2301.04883)]
[[Github](https://github.com/nttmdlab-nlp/SlideVQA)]

**CMMU: A Benchmark for Chinese Multi-modal Multi-type Question Understanding and Reasoning.**<br>
*Z He, X Wu, P Zhou, R Xuan, G Liu, X Yang, Q Zhu, H Huang.*<br>
arXiv:2401.14011, 2024.
[[ArXiv](https://arxiv.org/pdf/2401.14011)]
[[Github](https://github.com/FlagOpen/CMMU)]

**TableVQA-Bench: A Visual Question Answering Benchmark on Multiple Table Domains.**<br>
*Y Kim, M Yim, KY Song.*<br>
arXiv:2404.19205, 2024.
[[ArXiv](https://arxiv.org/pdf/2404.19205)]
[[Github](https://github.com/naver-ai/tablevqabench)]

**MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering.**<br>
*J Tang, Q Liu, Y Ye, J Lu, S Wei, C Lin, W Li, MFFB Mahmood, H Feng, Z Zhao, Y Wang, et al.*<br>
arXiv:2405.11985, 2024.
[[ArXiv](https://arxiv.org/pdf/2405.11985)]
[[Github](https://bytedance.github.io/MTVQA/)]

### Reasoning

**InfiMM-Eval: Complex Open-Ended Reasoning Evaluation For Multi-Modal Large Language Models.**<br>
*X Han, Q You, Y Liu, W Chen, H Zheng, K Mrini, et al.*<br>
arXiv:2311.11567, 2023.
[[ArXiv](https://arxiv.org/abs/2311.11567)]

**Measuring and improving chain-of-thought reasoning in vision-language models.**<br>
*Y Chen, K Sikka, M Cogswell, H Ji, A Divakaran.*<br>
arXiv:2309.04461, 2023.
[[ArXiv](https://arxiv.org/pdf/2309.04461)]
[[Github](https://github.com/Yangyi-Chen/CoTConsistency)]

**Complex Video Reasoning and Robustness Evaluation Suite for Video-LMMs.**<br>
*MU Khattak, MF Naeem, J Hassan, M Naseer, et al.*<br>
arXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2405.03690)]
[[Github](https://mbzuai-oryx.github.io/CVRR-Evaluation-Suite/)]

**ConTextual: Evaluating Context-Sensitive Text-Rich Visual Reasoning in Large Multimodal Models.**<br>
*R Wadhawan, H Bansal, KW Chang, N Peng.*<br>
arXiv:2401.13311, 2024.
[[ArXiv](https://arxiv.org/pdf/2401.13311)]
[[Github](https://con-textual.github.io/)]

**Exploring the reasoning abilities of multimodal large language models (mllms): A comprehensive survey on emerging trends in multimodal reasoning.**<br>
*Y Wang, W Chen, X Han, X Lin, H Zhao, Y Liu, B Zhai, J Yuan, Q You, H Yang.*<br>
arXiv:2401.06805, 2024.
[[ArXiv](https://arxiv.org/pdf/2401.06805)]

**Mementos: A comprehensive benchmark for multimodal large language model reasoning over image sequences.**<br>
*X Wang, Y Zhou, X Liu, H Lu, Y Xu, F He, J Yoon, T Lu, G Bertasius, M Bansal, H Yao, et al.*<br>
arXiv:2401.10529, 2024.
[[ArXiv](https://arxiv.org/pdf/2401.10529)]
[[Github](https://github.com/umd-huang-lab/Mementos/)]

**NPHardEval4V: A Dynamic Reasoning Benchmark of Multimodal Large Language Models.**<br>
*L Fan, W Hua, X Li, K Zhu, M Jin, L Li, H Ling, J Chi, J Wang, X Ma, Y Zhang.*<br>
arXiv:2403.01777, 2024.
[[ArXiv](https://arxiv.org/pdf/2403.01777)]
[[Github](https://github.com/lizhouf/NPHardEval4V)]

**Mathvista: Evaluating math reasoning in visual contexts with gpt-4v, bard, and other large multimodal models.**<br>
*P Lu, H Bansal, T **a, J Liu, C Li, H Hajishirzi, et al.*<br>
ICLR, 2024.
[[Homepage](https://mathvista.github.io/)]
[[Github](https://github.com/lupantech/MathVista)]

**MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark.**<br>
*X Yue, T Zheng, Y Ni, Y Wang, K Zhang, S Tong, Y Sun, M Yin, B Yu, G Zhang, H Sun, Y Su, et al.*<br>
arxiv:2409.02813, 2024.
[[ArXiv](https://arxiv.org/pdf/2409.02813)]
[[Github](https://mmmu-benchmark.github.io/#leaderboard)]

**[MATH-V] Measuring multimodal mathematical reasoning with math-vision dataset.**<br>
*K Wang, J Pan, W Shi, Z Lu, M Zhan, H Li.*<br>
arXiv:2402.14804, 2024.
[[ArXiv](https://arxiv.org/pdf/2402.14804)]
[[Github](https://mathvision-cuhk.github.io/)]

**[Mathverse] Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems?.**<br>
*R Zhang, D Jiang, Y Zhang, H Lin, Z Guo, P Qiu, A Zhou, P Lu, KW Chang, P Gao, H Li.*<br>
arXiv:2403.14624, 2024.
[[ArXiv](https://arxiv.org/pdf/2402.14804)]
[[Github](https://mathverse-cuhk.github.io/)]

### Multilingual

**CVQA: Culturally-diverse Multilingual Visual Question Answering Benchmark.**<br>
*D Romero, C Lyu, HA Wibowo, T Lynn, I Hamed, AN Kishore, A Mandal, A Dragonetti, et al.*<br>
arXiv:2406.05967, 2024.
[[Paper](https://arxiv.org/pdf/2406.05967f)]
[[DataSets](https://huggingface.co/datasets/afaji/cvqa)]

**[MMMB] Parrot: Multilingual Visual Instruction Tuning.**<br>
*HL Sun, DW Zhou, Y Li, S Lu, C Yi, QG Chen, Z Xu, W Luo, K Zhang, DC Zhan, HJ Ye.*<br>
arXiv:2406.02539, 2024.
[[Paper](https://arxiv.org/pdf/2406.02539)]
[[Github](https://github.com/AIDC-AI/Parrot)]

### Instruction-Following

**Visual instruction tuning.**<br>
*H Liu, C Li, Q Wu, YJ Lee.*<br>
NeurIPS, 2024.
[[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.pdf)]
[[Homepage](https://llava-vl.github.io/)]

**MIA-Bench: Towards Better Instruction Following Evaluation of Multimodal LLMs.**<br>
*Y Qian, H Ye, JP Fauconnier, P Grasch, Y Yang, et al.*<br>
arXiv, 2024.
[[Paper](https://arxiv.org/pdf/2407.01509)]
[[Homepage](https://llava-vl.github.io/)]

### High-Level-Vision

#### OCR

**[OCRBench] On the hidden mystery of ocr in large multimodal models.**<br>
*Y Liu, Z Li, H Li, W Yu, M Huang, D Peng, M Liu, M Chen, C Li, L Jin, X Bai.*<br>
arXiv:2305.07895, 2023.
[[ArXiv](https://arxiv.org/html/2305.07895v5)]
[[Github](https://github.com/Yuliang-Liu/MultimodalOCR)]

#### Aesthetics

**[Aesbench] Aesbench: An expert benchmark for multimodal large language models on image aesthetics perception.**<br>
*Y Huang, Q Yuan, X Sheng, Z Yang, H Wu, P Chen, Y Yang, L Li, W Lin.*<br>
arXiv:2401.08276, 2024.
[[ArXiv](https://arxiv.org/pdf/2401.08276)]
[[Github](https://github.com/yipoh/AesBench)]

**[A-Bench] A-Bench: Are LMMs Masters at Evaluating AI-generated Images?**<br>
*Z Zhang, H Wu, C Li, Y Zhou, W Sun, X Min, Z Chen, X Liu, W Lin, G Zhai.*<br>
arXiv:2406.03070, 2024.
[[ArXiv](https://arxiv.org/pdf/2401.08276)]
[[Github](https://github.com/Q-Future/A-Bench)]

### Low-Level-Vision

**Q-bench: A benchmark for general-purpose foundation models on low-level vision.**<br>
*H Wu, Z Zhang, E Zhang, C Chen, L Liao, A Wang, C Li, W Sun, Q Yan, G Zhai, W Lin.*<br>
arXiv:2309.14181, 2023.
[[ArXiv](https://arxiv.org/pdf/2309.14181)]

**A Benchmark for Multi-modal Foundation Models on Low-level Vision: from Single Images to Pairs.**<br>
*Z Zhang, H Wu, E Zhang, G Zhai, W Lin.*<br>
arXiv:2402.07116, 2024.
[[ArXiv](https://arxiv.org/pdf/2402.07116)]
[[Github](https://github.com/Q-Future/Q-Bench)]

### Reliable

|Date|Task|Title|Paper|HomePage|Github|DataSets|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|2023| Hallucination | An llm-free multi-dimensional benchmark for mllms hallucination evaluation.|[[ArXiv]](https://arxiv.org/pdf/2311.07397) |-|[[Github]](https://github.com/junyangwang0410/AMBER)|-|
|2024| Hallucination | [POPE] Evaluating object hallucination in large vision-language models.|[[ArXiv]](https://arxiv.org/pdf/2305.10355) |-|-|-|
|2024| Hallucination | LMMs-Eval: Reality Check on the Evaluation of Large Multimodal Models.|[[ArXiv]](https://arxiv.org/pdf/2407.12772) |-|-|-|
|2024| Hallucination | [Hallusionbench] Hallusionbench: You see what you think? or you think what you see? an image-context reasoning benchmark challenging for gpt-4v (ision), llava-1.5, and other multi-modality models.|[[CVPR]](https://arxiv.org/pdf/2310.14566) |-|[[Github]](https://github.com/tianyi-lab/HallusionBench)|-|
|2024| Hallucination | Evaluating the Quality of Hallucination Benchmarks for Large Vision-Language Models.|[[ArXiv]](https://arxiv.org/pdf/2402.14804) |-|[[Github]](https://mathvision-cuhk.github.io/)|-|

### Robust

**Fool your (vision and) language model with embarrassingly simple permutations.**<br>
*Y Zong, T Yu, et al.*<br>
arXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2310.01651)]
[[Github](https://github.com/ys-zong/FoolyourVLLMs)]

**Seeing Clearly, Answering Incorrectly: A Multimodal Robustness Benchmark for Evaluating MLLMs on Leading Questions.**<br>
*Y Liu, Z Liang, Y Wang, M He, J Li, B Zhao.*<br>
arXiv:2406.10638, 2024.
[[ArXiv](https://arxiv.org/pdf/2406.10638)]
[[Github](https://github.com/BAAI-DCAI/Multimodal-Robustness-Benchmark)]

**Benchmarking Robustness of Adaptation Methods on Pre-trained Vision-Language Models.**<br>
*S Chen, J Gu, Z Han, Y Ma, P Torr, V Tresp.*<br>
Advances in Neural Information Processing Systems, 2024.
[[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/file/a2a544e43acb8b954dc5846ff0d77ad5-Paper-Datasets_and_Benchmarks.pdf)]
[[Github](https://adarobustness.github.io/)]

## Security

**How many unicorns are in this image? a safety evaluation benchmark for vision llms.**<br>
*H Tu, C Cui, Z Wang, Y Zhou, B Zhao, J Han, W Zhou, H Yao, C Xie.*<br>
arXiv:2311.16101, 2023.
[[ArXiv](https://arxiv.org/pdf/2311.16101)]
[[Github](https://github.com/UCSC-VLAA/vllm-safety-benchmark)]

**T2VSafetyBench: Evaluating the Safety of Text-to-Video Generative Models.**<br>
*Y Miao, Y Zhu, Y Dong, L Yu, J Zhu, XS Gao.*<br>
arxiv:2407.05965, 2024.
[[ArXiv](https://arxiv.org/pdf/2407.05965)]

## Application

### Search

**MMSearch: Benchmarking the Potential of Large Models as Multi-modal Search Engines.**<br>
*D Jiang, R Zhang, Z Guo, Y Wu, J Lei, P Qiu, P Lu, Z Chen, G Song, P Gao, Y Liu, C Li, H Li.*<br>
arXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2409.12959)]
[[Github](https://mmsearch.github.io/)]

### Agent

**VisualAgentBench: Towards Large Multimodal Models as Visual Foundation Agents.**<br>
*X Liu, T Zhang, et al.*<br>
arXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2408.06327)]
[[Github](https://github.com/THUDM/VisualAgentBench)]

## Industry

### Medical

**Gmai-mmbench: A comprehensive multimodal evaluation benchmark towards general medical ai.**<br>
*P Chen, J Ye, G Wang, Y Li, Z Deng, W Li, T Li, H Duan, Z Huang, Y Su, B Wang, S Zhang, et al.*<br>
ArXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2408.03361)]
[[Huggingface](https://huggingface.co/papers/2408.03361)]

**Omnimedvqa: A new large-scale comprehensive evaluation benchmark for medical lvlm.**<br>
*Y Hu, T Li, Q Lu, W Shao, J He, Y Qiao, P Luo.*<br>
CVPR, 2024.
[[CVPR](https://openaccess.thecvf.com/content/CVPR2024/papers/Hu_OmniMedVQA_A_New_Large-Scale_Comprehensive_Evaluation_Benchmark_for_Medical_LVLM_CVPR_2024_paper.pdf)]
[[Github](https://github.com/OpenGVLab/Multi-Modality-Arena)]

## Human-Machine-Interaction

**WildVision: Evaluating Vision-Language Models in the Wild with Human Preferences.**<br>
*Y Lu, D Jiang, W Chen, WY Wang, Y Choi, BY Lin.*<br>
arXiv:2406.11069, 2024.
[[ArXiv](https://arxiv.org/pdf/2406.11069)]
[[DataSets](https://huggingface.co/spaces/WildVision/vision-arena)]

**AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models.**<br>
*Y Wu, W Yu, Y Cheng, Y Wang, X Zhang, J Xu, M Ding, Y Dong.*<br>
arXiv:2406.09295, 2024.
[[ArXiv](https://arxiv.org/pdf/2406.09295)]
[[Github](https://alignmmbench.github.io/)]

**Mmtom-qa: Multimodal theory of mind question answering.**<br>
*C Jin, Y Wu, J Cao, J Xiang, YL Kuo, Z Hu, T Ullman, A Torralba, JB Tenenbaum, T Shu.*<br>
arXiv:2401.08743, 2024.
[[ArXiv](https://arxiv.org/pdf/2401.08743)]
[[Github](https://chuanyangjin.com/mmtom-qa)]

## Testing-Methods

**Task Me Anything.**<br>
*J Zhang, W Huang, Z Ma, O Michel, D He, et al.*<br>
ArXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2406.11775)]
[[HomePage](https://github.com/JieyuZ2/TaskMeAnything)]

**A lightweight generalizable evaluation and enhancement framework for generative models and generated samples.**<br>
*G Zhao, V Magoulianitis, S You, CCJ Kuo.*<br>
WACV, 2024.
[[ArXiv](https://openaccess.thecvf.com/content/WACV2024W/VAQ/papers/Zhao_A_Lightweight_Generalizable_Evaluation_and_Enhancement_Framework_for_Generative_Models_WACVW_2024_paper.pdf)]

### Quality Evaluation

**MLLM-Bench: Evaluating Multimodal LLMs with Per-sample Criteria.**<br>
*W Ge, S Chen, et al.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/abs/2311.13951v2)]
[[HomePage](https://mllm-bench.llmzoo.com/)]

**Mllm-as-a-judge: Assessing multimodal llm-as-a-judge with vision-language benchmark.**<br>
*D Chen, R Chen, S Zhang, Y Liu, Y Wang, H Zhou, Q Zhang, P Zhou, Y Wan, L Sun.*<br>
arXiv:2402.04788, 2024.
[[ArXiv](https://arxiv.org/pdf/2402.04788)]
[[HomePage](https://mllm-judge.github.io/)]

**MJ-BENCH Is Your Multimodal Reward Model Really a Good Judge for Text-to-Image Generation**<br>
*Z Chen, Y Du, Z Wen, Y Zhou, C Cui, Z Weng, et al.*<br>
ArXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2407.04842)]
[[HomePage](https://huggingface.co/MJ-Bench)]

## Testing-Tools

**GenAI Arena: An Open Evaluation Platform for Generative Models.**<br>
*D Jiang, M Ku, T Li, Y Ni, S Sun, R Fan, W Chen.*<br>
arxiv:2406.04485, 2024.
[[ArXiv](https://arxiv.org/pdf/2406.04485)]
[[HomePage](https://huggingface.co/spaces/TIGER-Lab/GenAI-Arena)]

**VLMEvalKit**<br>
*Shanghai AI LAB*<br>
[[Github](https://github.com/open-compass/VLMEvalKit)]

**lmms-eval**<br>
*LMMs-Lab*<br>
[[HomePage](https://lmms-lab.github.io/)]
[[Github](https://github.com/EvolvingLMMs-Lab/lmms-eval)]

**Multi-Modality-Arena**<br>
*OpenGVLab*<br>
[[Github](https://github.com/OpenGVLab)]

## Challenges

**[MMStar] Are We on the Right Way for Evaluating Large Vision-Language Models?**<br>
*L Chen, J Li, X Dong, P Zhang, Y Zang, Z Chen, H Duan, J Wang, Y Qiao, D Lin, F Zhao, et al.*<br>
ArXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2403.20330)]
[[Github](https://mmstar-benchmark.github.io/)]

**What Are We Measuring When We Evaluate Large Vision-Language Models? An Analysis of Latent Factors and Biases.**<br>
*AMH Tiong, J Zhao, B Li, J Li, SCH Hoi, et al.*<br>
ArXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2404.02415)]
[[Github](https://github.com/jq-zh/olive-dataset)]
