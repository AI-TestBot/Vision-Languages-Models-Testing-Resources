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
  * 📖[Instruction-Following](#Instruction-Following)
  * 📖[High-Level-Vision](#High-Level-Vision)
    * 📖[OCR](#OCR)
  * 📖[Low-Level-Vision](#Low-Level-Vision)
  * 📖[Reliable](#Reliable)
    * 📖[Hallucination](#Hallucination)
  * 📖[Robust](#Robust)
* 📖[Security](#Security)
* 📖[Testing-Methods](#Testing-Methods)
* 📖[Testing-Tools](#Testing-Tools)
* 📖[Challenges](#Challenges)

## 📖Review  

**From GPT-4 to Gemini and Beyond: Assessing the Landscape of MLLMs on Generalizability, Trustworthiness and Causality through Four Modalities.**<br>
*C Lu, C Qian, G Zheng, H Fan, H Gao, J Zhang, J Shao, J Deng, J Fu, K Huang, K Li, L Li, et al.*<br>
ArXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2401.15071.pdf)]
[[Github](https://openlamm.github.io/Leaderboards)]

## 📖General 

### Comprehensive

**Holistic evaluation of text-to-image models.**<br>
*T Lee, M Yasunaga, C Meng, Y Mai, JS Park, A Gupta, Y Zhang, D Narayanan, H Teufel, et al.*<br>
Advances in Neural Information Processing Systems, 2024.
[[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/dd83eada2c3c74db3c7fe1c087513756-Paper-Datasets_and_Benchmarks.pdf)]
[[Github](https://github.com/stanford-crfm/helm)

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

**Seed-bench: Benchmarking multimodal llms with generative comprehension.**<br>
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

**Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi.**<br>
*X Yue, Y Ni, K Zhang, T Zheng, R Liu, G Zhang, S Stevens, D Jiang, W Ren, Y Sun, C Wei, et al.*<br>
CVPR, 2024.
[[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Yue_MMMU_A_Massive_Multi-discipline_Multimodal_Understanding_and_Reasoning_Benchmark_for_CVPR_2024_paper.pdf)]
[[Github](https://mmmu-benchmark.github.io/)]

**Vbench: Comprehensive benchmark suite for video generative models.**<br>
*Z Huang, Y He, J Yu, F Zhang, C Si, Y Jiang, Y Zhang, T Wu, Q Jin, N Chanpaisit, Y Wang, et al.*<br>
CVPR, 2024.
[[ArXiv](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.pdf)]
[[Github](https://vchitect.github.io/VBench-project/)]

**Beyond task performance: Evaluating and reducing the flaws of large multimodal models with in-context learning.**<br>
*M Shukor, A Rame, C Dancette, M Cord.*<br>
ICLR, 2024.
[[ArXiv](https://arxiv.org/pdf/2310.00647)]
[[Github](https://github.com/mshukor/EvALign-ICL)]

### Understanding

|Date|Task|Title|Paper|HomePage|Github|DataSets|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|2023| Content | MM-BigBench: Evaluating Multimodal Models on Multimodal Content Comprehension Tasks.|[[ArXiv]](https://arxiv.org/pdf/2310.09036) |-|[[Github]](https://github.com/declare-lab/MM-InstructEval)|-|
|2024| Dialog | MMDU: A Multi-Turn Multi-Image Dialog Understanding Benchmark and Instruction-Tuning Dataset for LVLM.|[[ArXiv]](https://arxiv.org/pdf/2406.11833) |-|[[Github]](https://github.com/Liuziyu77/MMDU/)|-|
|2023| Image | Journeydb: A benchmark for generative image understanding.|[[NeurIPS]](https://proceedings.neurips.cc/paper_files/paper/2023/file/9bc59aff4685e39e1a8175d5303248a1-Paper-Datasets_and_Benchmarks.pdf) |-|[[Github]](https://journeydb.github.io/)|-|
|2024| Video | MVBench: A Comprehensive Multi-modal Video Understanding Benchmark.|[[CVPR]](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_MVBench_A_Comprehensive_Multi-modal_Video_Understanding_Benchmark_CVPR_2024_paper.pdf) |-|[[Github]](https://github.com/OpenGVLab/Ask-Anything)|-|
|2024| Video | MLVU: A Comprehensive Benchmark for Multi-Task Long Video Understanding.|[[ArXiv]](https://arxiv.org/pdf/2406.04264) |-|[[Github]](https://github.com/JUNJIE99/MLVU)|-|

### Generation

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
|2023| Text-to-Video | Fetv: A benchmark for fine-grained evaluation of open-domain text-to-video generation.|[[NeurIPS]](https://proceedings.neurips.cc/paper_files/paper/2023/file/c481049f7410f38e788f67c171c64ad5-Paper-Datasets_and_Benchmarks.pdf) |-|[[Github]](https://github.com/llyx97/FETV)|-|
|2024| Text-to-Video | Subjective-Aligned Dateset and Metric for Text-to-Video Quality Assessment.|[[ArXiv]](https://arxiv.longhoe.net/pdf/2403.11956) |-|[[Github]](https://github.com/QMME/T2VQA)|-|
|2024| Text-to-Video | T2VBench: Benchmarking Temporal Dynamics for Text-to-Video Generation.|[[CVPR]](https://openaccess.thecvf.com/content/CVPR2024W/EvGenFM/papers/Ji_T2VBench_Benchmarking_Temporal_Dynamics_for_Text-to-Video_Generation_CVPRW_2024_paper.pdf) |-|[[Github]](https://ji-pengliang.github.io/T2VBench/)|-|

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

**CMMU: A Benchmark for Chinese Multi-modal Multi-type Question Understanding and Reasoning.**<br>
*Z He, X Wu, P Zhou, R Xuan, G Liu, X Yang, Q Zhu, H Huang.*<br>
arXiv:2401.14011, 2024.
[[ArXiv](https://arxiv.org/pdf/2401.14011)]
[[Github](https://github.com/FlagOpen/CMMU)]

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

### Instruction-Following

**Visual instruction tuning.**<br>
*H Liu, C Li, Q Wu, YJ Lee.*<br>
NeurIPS, 2024.
[[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.pdf)]
[[Homepage](https://llava-vl.github.io/)]

### High-Level-Vision

#### OCR

**On the hidden mystery of ocr in large multimodal models.**<br>
*Y Liu, Z Li, H Li, W Yu, M Huang, D Peng, M Liu, M Chen, C Li, L Jin, X Bai.*<br>
arXiv:2305.07895, 2023.
[[ArXiv](https://arxiv.org/html/2305.07895v5)]
[[Github](https://github.com/Yuliang-Liu/MultimodalOCR)]

### Low-Level-Vision

**A Benchmark for Multi-modal Foundation Models on Low-level Vision: from Single Images to Pairs.**<br>
*Z Zhang, H Wu, E Zhang, G Zhai, W Lin.*<br>
arXiv:2402.07116, 2024.
[[ArXiv](https://arxiv.org/pdf/2402.07116)]
[[Github](https://github.com/Q-Future/Q-Bench)]

### Reliable

|Date|Task|Title|Paper|HomePage|Github|DataSets|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|2023| Hallucination | An llm-free multi-dimensional benchmark for mllms hallucination evaluation.|[[ArXiv]](https://arxiv.org/pdf/2311.07397) |-|[[Github]](https://github.com/junyangwang0410/AMBER)|-|
|2024| Hallucination | Evaluating object hallucination in large vision-language models.|[[ArXiv]](https://arxiv.org/pdf/2305.10355) |-|-|-|

### Robust

**Seeing Clearly, Answering Incorrectly: A Multimodal Robustness Benchmark for Evaluating MLLMs on Leading Questions.**<br>
*Y Liu, Z Liang, Y Wang, M He, J Li, B Zhao.*<br>
arXiv:2406.10638, 2024.
[[ArXiv](https://arxiv.org/pdf/2406.10638)]
[[Github](https://github.com/BAAI-DCAI/Multimodal-Robustness-Benchmark)]

## Security

**How many unicorns are in this image? a safety evaluation benchmark for vision llms.**<br>
*H Tu, C Cui, Z Wang, Y Zhou, B Zhao, J Han, W Zhou, H Yao, C Xie.*<br>
arXiv:2311.16101, 2023.
[[ArXiv](https://arxiv.org/pdf/2311.16101)]
[[Github](https://github.com/UCSC-VLAA/vllm-safety-benchmark)]

## Testing-Methods

### Evaluation

**MLLM-Bench: Evaluating Multimodal LLMs with Per-sample Criteria.**<br>
*W Ge, S Chen, et al.*<br>
ArXiv, 2023.
[[ArXiv](https://arxiv.org/abs/2311.13951v2)]
[[HomePage](https://mllm-bench.llmzoo.com/)]

## Testing-Tools

**lmms-eval**<br>
*Openai*<br>
[[HomePage](https://lmms-lab.github.io/)]
[[Github](https://github.com/EvolvingLMMs-Lab/lmms-eval)]

## Challenges

**Are We on the Right Way for Evaluating Large Vision-Language Models?**<br>
*L Chen, J Li, X Dong, P Zhang, Y Zang, Z Chen, H Duan, J Wang, Y Qiao, D Lin, F Zhao, et al.*<br>
ArXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2403.20330)]
[[Github](https://mmstar-benchmark.github.io/)]

**What Are We Measuring When We Evaluate Large Vision-Language Models? An Analysis of Latent Factors and Biases.**<br>
*AMH Tiong, J Zhao, B Li, J Li, SCH Hoi, et al.*<br>
ArXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2404.02415)]
[[Github](https://github.com/jq-zh/olive-dataset)]
