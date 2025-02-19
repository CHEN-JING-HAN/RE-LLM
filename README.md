# RE-LLM: Refining Empathetic Speech-LLM Reponses by Integrating Emotion Nuance Modeling

## Overview

![framework](Framework.jpg)
The architecture of our proposed RE-LLM comprises a speech-LLM and an emotion nuance module. A preprocessing generation and expected behavioral alignment constrained on nuance emotion training strategy are depicted as well.

## Abstract
As generative AI advances, enhancing empathy in human-AI interactions is crucial. While prior work focuses on emotional reflection, emotional exploration—key to deeper engagement—remains overlooked. Existing LLMs rely on text or ASR-transcript embeddings, which capture limited emotion nuances. To address this, we propose RE-LLM, a speech-LLM integrating dimensional emotion embeddings and auxiliary learning. Experiments show significant gains in computational empathy, with RE-LLM relatively improving the “Emotional Reaction” score by 14.79% and 6.76% compared to text-only and speech-LLM baselines on ESD, and slightly on IEMOCAP. Notably, it relatively enhances the “Exploration” score by 35.42% and 3.91% on IEMOCAP and 139.28% and 9.83% on ESD. Additionally, it boosts unweighted accuracy by 5.4% on IEMOCAP and 2.3% on ESD in speech emotion recognition. These results highlight the enriched emotional understanding and improved empathetic response generation of RE-LLM.

## Getting Started
### 0. Environment Setup
Ensure that the following dependencies are installed:
```bash
pip install -r requirements.txt
```
### 1. Preprocessing Data

Execute data preprocessing:

```bash
python preprocess.py --input data/raw --output data/processed
```

### 2. Training the Model

Train the model:

```bash
python train.py --config config.yaml
```

### 3. Inference

Use the model for inference:

```bash
python inference.py --model model.pth --input sample.jpg
```

### 4. Evaluate result

Evaluate model performance:

```bash
python evaluate.py --predictions results.json
```
