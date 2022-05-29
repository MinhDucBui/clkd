<div align="center">

# Cross-Lingual Knowledge Distillation

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## 📌&nbsp;&nbsp;Introduction
This project enables to distill multilingual transformers into language-specific students. Features contain:

- Adjust any distillation loss 
- Any number of students and languages per student
- Change the teacher and student architecture 
- Choose between monolingual, bilingual, or multilingual distillation setup
- Compnent Sharing across students
- Initialization of students from teacher layers

## 🚀&nbsp;&nbsp;Quickstart

Configure your environment first.

```bash
# clone project
git clone https://github.com/MinhDucBui/clkd.git
cd clkd

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install requirements
pip install -r requirements.txt

# Please make sure to have the right Pytorch Version: Install pytorch according to instructions
# https://pytorch.org/get-started/
```

Execute the main script. The default setting uses the same strategy as MonoShot.
```bash
# Choose GPU Device
export CUDA_VISIBLE_DEVICES=0

# execute main script
python run.py
```

### ⚡&nbsp;&nbsp;Your Superpowers

<details>
<summary><b>Change Distillation Loss</b></summary>

> Hydra allows you to easily overwrite any parameter defined in your config. See students/individuals/loss for all loss functions.

```bash
python run.py students/individual/loss=monoalignment
```

To contruct your own distillation loss, we provide [bass losses](https://github.com/MinhDucBui/clkd/tree/main/configs/students/individual/loss/base_loss), that can be used to construct the final loss. Furthermore, we provide all distillation losses used in this thesis [here](https://github.com/MinhDucBui/clkd/tree/main/configs/students/individual/loss).
  
Example of constucting the distillation loss from the MLM loss and logit distillation with CE loss with equal weighting.
  

```
_target_: src.loss.loss.GeneralLoss
defaults:
  - base_loss@base_loss.mlm: mlm.yaml
  - base_loss@base_loss.softtargets_ce: softtargets_ce.yaml

base_loss:
  softtargets_ce:
    temperature: 4.0

loss_weighting:
  mlm: 0.5
  softtargets_ce: 0.5
```
  
  
</details>

<details>
<summary><b>Change Student Number and Languages</b></summary>

> We constructed some default configs for different scenarios:

```bash
# monolingual setting with english-turkish language pair
python train.py experiment=monolingual

# monolingual setting with english-basque language pair
python train.py experiment=monolingual_eu

# monolingual setting with english-turkish language pair
python train.py experiment=monolingual_sw

# monolingual setting with english-turkish language pair
python train.py experiment=monolingual_ur

# bilingual setting with english-turkish language pair
python train.py experiment=monolingual_bilingual
```

To construct a custom setting, please see the documentation [here](https://github.com/MinhDucBui/clkd/blob/main/configs/experiment/monolingual.yaml).

</details>

<details>
<summary><b>Embedding Sharing across Students</b></summary>

```bash
# Share language embeddings only in each student, not across students.
python run.py students.embed_sharing="in_each_model" 
```
To construct a custom setting, please see the documentation [here](https://github.com/MinhDucBui/clkd/blob/main/configs/experiment/monolingual.yaml).

  
</details>

<details>
<summary><b>Layer Sharing across Students</b></summary>

Please see the documentation [here](https://github.com/MinhDucBui/clkd/blob/main/configs/students/default.yaml#L4-L10).

  
</details>

<details>
<summary><b>Change Student Architecture</b></summary>

```bash
# Use the same architecture as the teacher
python run.py students/individual/model=from_teacher
```
More architectures can be found [here](https://github.com/MinhDucBui/clkd/tree/main/configs/students/individual/model).   

</details>


<details>
<summary><b>Student Initialization</b></summary>
  
> Default uses weights from the teacher.  
```bash
# Randomly Initialize Embedding Weights
python run.py students.individual.model.weights_from_teacher.embeddings=False
  
# Randomly Initialize Layer Weights
python run.py students.individual.model.weights_from_teacher.transformer_blocks=False
```

</details>

<br>

## ℹ️&nbsp;&nbsp;Project Structure
The directory structure of new project looks like this:
```

├── configs                 <- Hydra configuration files
│   ├── callbacks               <- Callbacks configs
│   ├── collate_fn              <- Collate functions configs
│   ├── datamodule              <- Datamodule configs
│   ├── distillation_setup      <- Distillation configs
│   ├── evaluation              <- Evaluation configs
│   ├── experiment              <- Experiment configs
│   ├── hydra                   <- Hydra related configs
│   ├── logger                  <- Logger configs
│   ├── students                <- Student configs
│   ├── teacher                 <- Teacher configs
│   ├── trainer                 <- Trainer configs
│   │
│   └── config.yaml             <- Main project configuration file
│
├── data                    <- Project data
│
├── logs                    <- Logs generated by Hydra and PyTorch Lightning loggers
│
├── src
│   ├── callbacks               <- Lightning callbacks
│   ├── datamodules             <- Lightning datamodules
│   ├── distillation            <- Distillation Setup Files
│   ├── evaluation              <- Evaluation Files
│   ├── los                     <- Loss Files
│   ├── models                  <- Lightning models
│   ├── utils                   <- Utility scripts
│   │
│   └── train.py                <- Training pipeline
│
├── run.py                  <- Run pipeline with chosen experiment configuration
│
├── .env.example            <- Template of the file for storing private environment variables
├── .gitignore              <- List of files/folders ignored by git
├── .pre-commit-config.yaml <- Configuration of automatic code formatting
├── setup.cfg               <- Configurations of linters and pytest
├── Dockerfile              <- File for building docker container
├── requirements.txt        <- File for installing python dependencies
├── LICENSE
└── README.md
```
<br>
