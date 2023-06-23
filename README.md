# Rethinking the Evaluation for Conversational Recommendation in the Era of Large Language Models

This repo provides the source code & data of our paper: Rethinking the Evaluation for Conversational Recommendation in the Era of Large Language Models.

## 😀 Overview

**Highlights**:
- 1️⃣ We are the first to examine ChatGPT in conversational recommendation systematically, the ability of which is **underestimated** in traditional evaluation approach.
- 💡 We propose a new interactive approach that employs LLM-based user simulators for evaluating CRSs.
- 🔝 The recall@50 metric can be boosted from 0.218 to 0.739 on the redial dataset with our new interactive evaluation approach, even surpassing the currently leading CRS baseline.

we propose an **i**nteractive **Eval**uation approach based on **LLM**s named **iEvaLM** that harnesses LLM-based user simulators. We take the ground-truth items from the example as the user preference through the interaction, and use them to set up the persona of the simulated users by LLMs through instructions. To further make a comprehensive evaluation, we consider two types of interaction: *attribute-based question answering* and *free-form chit-chat*.

<p align="center">
  <img src="./asset/eval.png" width="75%" height="75% title="Overview of iEvaLM-CRS" alt="">
</p>


## 🚀 Quick Start

### Requirements

- python == 3.9.16
- pytorch == 1.13.1
- transformers == 4.28.1
- pyg == 2.3.0
- accelerate == 0.18.0

### Download Models

You can download our fine-tuned models from the [link](https://drive.google.com/drive/folders/1h2AcRn3cd9qXToM9hFAXFV5evXJM-wyD?usp=sharing), which include recommendation and conversation models of **KBRD**, **BARCOR** and **UniCRS**. Please put the downloaded model into src/utils/model directory.

### Interact with the user simulator

- dataset: [redial, opendialkg]
- mode: [ask, chat]
- model: [kbrd, barcor, unicrs, chatgpt]

```bash
cd script
bash {dataset}/cache_item.sh 
bash {dataset}/{mode}_{model}.sh 
```

You can customize your iEvaLM-CRS by specifying these configs:
- `--api_key`: your API key
- `--turn_num`: number of conversation turns. We employ five-round interaction in iEvaLM-CRS.

After the execution, you will find detailed interaction information under "save_{turn_num}/{mode}/{model}/{dataset}/".

### Evaluate

```bash
cd script
bash {dataset}/Rec_eval.sh
```

You can customize your iEvaLM-CRS by specifying these configs:
- `--turn_num`: number of conversation turns.
- `--mode`: [ask, chat]

After the execution, you will find evaluation results under "save_{turn_num}/result/{mode}/{model}/{dataset}.json".
