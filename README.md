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

**Towards an Exhaustive Evaluation of Vision-Language Foundation Models.**<br>
*E Salin, S Ayache, B Favre.*<br>
ICCV, 2023.
[[Paper](https://openaccess.thecvf.com/content/ICCV2023W/MMFM/papers/Salin_Towards_an_Exhaustive_Evaluation_of_Vision-Language_Foundation_Models_ICCVW_2023_paper.pdf)]

**Lvlm-ehub: A comprehensive evaluation benchmark for large vision-language models.**<br>
*P Xu, W Shao, K Zhang, P Gao, S Liu, M Lei, F Meng, S Huang, Y Qiao, P Luo.*<br>
arXiv:2306.09265, 2023.
[[ArXiv](https://arxiv.org/pdf/2306.09265)]

**Mmbench: Is your multi-modal model an all-around player?**<br>
*Y Liu, H Duan, Y Zhang, B Li, S Zhang, W Zhao, Y Yuan, J Wang, C He, Z Liu, K Chen, D Lin.*<br>
arXiv:2307.06281, 2023.
[[ArXiv](https://arxiv.org/pdf/2307.06281.pdf)]
[[Github](https://mmbench.opencompass.org.cn/home)]

**Seed-bench: Benchmarking multimodal llms with generative comprehension.**<br>
*B Li, R Wang, G Wang, Y Ge, Y Ge, Y Shan.*<br>
arXiv:2307.16125, 2023.
[[ArXiv](https://arxiv.org/pdf/2307.16125)]
[[Github](https://github.com/AILab-CVC/SEED-Bench)]

**Mm-vet: Evaluating large multimodal models for integrated capabilities.**<br>
*W Yu, Z Yang, L Li, J Wang, K Lin, Z Liu, X Wang, L Wang.*<br>
arXiv:2308.02490, 2023.
[[ArXiv](https://arxiv.org/pdf/2308.02490)]
[[Github](https://github.com/yuweihao/MM-Vet)]

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

**Beyond task performance: Evaluating and reducing the flaws of large multimodal models with in-context learning.**<br>
*M Shukor, A Rame, C Dancette, M Cord.*<br>
arXiv:2310.00647, 2023.
[[ArXiv](https://arxiv.org/pdf/2310.00647)]
[[Github](https://github.com/mshukor/EvALign-ICL)]

**Vbench: Comprehensive benchmark suite for video generative models.**<br>
*Z Huang, Y He, J Yu, F Zhang, C Si, Y Jiang, Y Zhang, T Wu, Q Jin, N Chanpaisit, Y Wang, et al.*<br>
CVPR, 2024.
[[ArXiv](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.pdf)]
[[Github](https://vchitect.github.io/VBench-project/)]

### Understanding

|Date|Task|Title|Paper|HomePage|Github|DataSets|Organization|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|2023| Content | MM-BigBench: Evaluating Multimodal Models on Multimodal Content Comprehension Tasks.|[[ArXiv]](https://arxiv.org/pdf/2310.09036) |-|[[github]](https://github.com/declare-lab/MM-InstructEval)|-|Northeastern University|
|2024| Dialog | MMDU: A Multi-Turn Multi-Image Dialog Understanding Benchmark and Instruction-Tuning Dataset for LVLM.|[[ArXiv]](https://arxiv.org/pdf/2406.11833) |-|[[github]](https://github.com/Liuziyu77/MMDU/)|-|WHU|
|2024| Image | Journeydb: A benchmark for generative image understanding.|[[NeurIPS]](https://proceedings.neurips.cc/paper_files/paper/2023/file/9bc59aff4685e39e1a8175d5303248a1-Paper-Datasets_and_Benchmarks.pdf) |-|[[github]](https://journeydb.github.io/)|-|Chinese University of Hong Kong|
|2024| Video |MVBench: A Comprehensive Multi-modal Video Understanding Benchmark.|[[CVPR]](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_MVBench_A_Comprehensive_Multi-modal_Video_Understanding_Benchmark_CVPR_2024_paper.pdf) |-|[[github]](https://github.com/OpenGVLab/Ask-Anything)|-|Chinese Academy of Sciences|
|2024| Video |MLVU: A Comprehensive Benchmark for Multi-Task Long Video Understanding.|[[arXiv]](https://arxiv.org/pdf/2406.04264) |-|[[Github]](https://github.com/JUNJIE99/MLVU)|-|Beijing Academy of Artificial Intelligence. |

### Generation

#### Text-to-Image

**Evaluating Text to Image Synthesis: Survey and Taxonomy of Image Quality Metrics.**<br>
*S Hartwig, D Engel, L Sick, H Kniesel, T Payer, T Ropinski.*<br>
ArXiv, 2024.
[[ArXiv](https://arxiv.longhoe.net/pdf/2403.11821)]

**Llmscore: Unveiling the power of large language models in text-to-image synthesis evaluation.**<br>
*Y Lu, X Yang, X Li, XE Wang, WY Wang.*<br>
Advances in Neural Information Processing Systems, 2024.
[[Paper](https://so2.cljtscd.com/scholar?hl=zh-CN&as_sdt=0%2C5&q=LLMScore%3A+Unveiling+the+Power+of+Large+Language+Models+in+Text-to-Image+Synthesis+Evaluation&btnG=)]
[[Github](https://github.com/YujieLu10/LLMScore)]

**Imagereward: Learning and evaluating human preferences for text-to-image generation.**<br>
*J Xu, X Liu, Y Wu, Y Tong, Q Li, M Ding, J Tang, Y Dong.*<br>
Advances in Neural Information Processing Systems, 2024.
[[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/33646ef0ed554145eab65f6250fab0c9-Paper-Conference.pdf)]

**UNIAA: A Unified Multi-modal Image Aesthetic Assessment Baseline and Benchmark.**<br>
*Z Zhou, Q Wang, B Lin, Y Su, R Chen, X Tao, A Zheng, L Yuan, P Wan, D Zhang.*<br>
ArXiv, 2024.
[[ArXiv](https://arxiv.org/pdf/2404.09619)]

#### Text-to-Video

**Fetv: A benchmark for fine-grained evaluation of open-domain text-to-video generation.**<br>
*Y Liu, L Li, S Ren, R Gao, S Li, S Chen, X Sun, L Hou.*<br>
Advances in Neural Information Processing Systems, 2024.
[[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/c481049f7410f38e788f67c171c64ad5-Paper-Datasets_and_Benchmarks.pdf)]
[[Github](https://github.com/llyx97/FETV)]

**Subjective-Aligned Dateset and Metric for Text-to-Video Quality Assessment.**<br>
*T Kou, X Liu, Z Zhang, C Li, H Wu, X Min, G Zhai, N Liu.*<br>
arxiv:2403.11956, 2024.
[[ArXiv](https://arxiv.longhoe.net/pdf/2403.11956)]
[[Github](https://github.com/QMME/T2VQA)]

**T2VBench: Benchmarking Temporal Dynamics for Text-to-Video Generation.**<br>
*P Ji, C Xiao, H Tai, M Huo, et al.*<br>
CVPR, 2024.
[[ArXiv](https://openaccess.thecvf.com/content/CVPR2024W/EvGenFM/papers/Ji_T2VBench_Benchmarking_Temporal_Dynamics_for_Text-to-Video_Generation_CVPRW_2024_paper.pdf)]
[[Github](https://ji-pengliang.github.io/T2VBench/)]

### VQA

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

### Visual Instruction Following

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

#### Hallucination

**An llm-free multi-dimensional benchmark for mllms hallucination evaluation.**<br>
*J Wang, Y Wang, G Xu, J Zhang, Y Gu, H Jia, M Yan, J Zhang, J Sang.*<br>
arXiv:2311.07397, 2023.
[[ArXiv](https://arxiv.org/html/2311.07397v2)]
[[Github](https://github.com/junyangwang0410/AMBER)]

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
