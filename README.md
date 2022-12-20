---
pipeline_tag: sentence-similarity
language: en
license: apache-2.0
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- transformers
---

# hku-nlp/instructor-large
This is a general embedding model: It maps **any** piece of text (e.g., a title, a sentence, a document, etc.) to a fixed-length vector in test time **without further training**. With instructions, the embeddings are **domain-specific** (e.g., specialized for science, finance, etc.) and **task-aware** (e.g., customized for classification, information retrieval, etc.)

The model is easy to use with `sentence-transformer` library.

## Installation
```bash
git clone https://github.com/HKUNLP/instructor-embedding
cd sentence-transformers
pip install -e .
```

## Compute your customized embeddings
Then you can use the model like this to calculate domain-specific and task-aware embeddings:
```python
from sentence_transformers import SentenceTransformer
sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments"
instruction = "Represent the Science title; Input:"
model = SentenceTransformer('hku-nlp/instructor-large')
embeddings = model.encode([[instruction,sentence,0]])
print(embeddings)
```

## Calculate Sentence similarities
You can further use the model to compute similarities between two groups of sentences, with **customized embeddings**.
```python
from sklearn.metrics.pairwise import cosine_similarity
sentences_a = [['Represent the Science sentence; Input: ','Parton energy loss in QCD matter',0], 
               ['Represent the Financial statement; Input: ','The Federal Reserve on Wednesday raised its benchmark interest rate.',0]
sentences_b = [['Represent the Science sentence; Input: ','The Chiral Phase Transition in Dissipative Dynamics', 0],
               ['Represent the Financial statement; Input: ','The funds rose less than 0.5 per cent on Friday',0]
embeddings_a = model.encode(sentences_a)
embeddings_b = model.encode(sentences_b)
similarities = cosine_similarity(embeddings_a,embeddings_b)
print(similarities)
```