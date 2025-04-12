# RACA
code for "Region-locating Agent-guided Cross-modal Alignment for Chest X-ray Report Generation"
## Abstract:
Automated chest radiology report generation aims to leverage artificial intelligence models to produce diagnostic reports from medical images. A critical challenge in this multimodal task is achieving effective cross-modal semantic alignment between visual features and textual descriptions, particularly at the fine-grained level of anatomical regions. This study proposes a Region-locating Agent-guided Cross-modal Alignment (RACA) model that addresses limitations of existing approaches. The RACA framework introduces three key innovations: (1) a reinforcement learning-based region-locating agent (RLA) that adaptively identifies clinically relevant anatomical regions without requiring explicit region-level annotations; (2) a fine-grained cross-modal alignment mechanism that leverages large language models to semantically annotate report sentences and aligns them with localized image regions through contrastive learning; and (3) a bidirectional cross-attention (BCA) module that effectively integrates global contextual information with localized regional features. Extensive experiments on the IU X-ray and MIMIC-CXR datasets demonstrate that RACA performs competitively with current approaches, showing notable improvements in several key metrics. 
## Requirements:
## Data:
Please download the IU and MIMIC datasets, and place them in the `./data/` folder.<br>
👉[IU-Xray](https://iuhealth.org/find-medical-services/x-rays)<br>
👉[MIMIC-CXR ](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)<br>
# Training
run `./train.py`
# Testing
run `./test.py`
## Acknowledges:
We thank and [R2Gen](https://github.com/cuhksz-nlp/R2Gen) for their open source works.



