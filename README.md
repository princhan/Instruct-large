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

# hkunlp/instructor-large
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
model = SentenceTransformer('hkunlp/instructor-large')
embeddings = model.encode([[instruction,sentence,0]])
print(embeddings)
```

## Calculate Sentence similarities
You can further use the model to compute similarities between two groups of sentences, with **customized embeddings**.
```python
from sklearn.metrics.pairwise import cosine_similarity
sentences_a = [['Represent the Science sentence; Input: ','Parton energy loss in QCD matter',0], 
               ['Represent the Financial statement; Input: ','The Federal Reserve on Wednesday raised its benchmark interest rate.',0]]
sentences_b = [['Represent the Science sentence; Input: ','The Chiral Phase Transition in Dissipative Dynamics', 0],
               ['Represent the Financial statement; Input: ','The funds rose less than 0.5 per cent on Friday',0]]
embeddings_a = model.encode(sentences_a)
embeddings_b = model.encode(sentences_b)
similarities = cosine_similarity(embeddings_a,embeddings_b)
print(similarities)
```

## Information Retrieval
You can also use **customized embeddings** for information retrieval.
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
query  = [['Represent the Wikipedia question for retrieving supporting documents; Input: ','where is the food stored in a yam plant',0]]
corpus = [['Represent the Wikipedia document for retrieval; Input: ','Capitalism has been dominant in the Western world since the end of feudalism, but most feel[who?] that the term "mixed economies" more precisely describes most contemporary economies, due to their containing both private-owned and state-owned enterprises. In capitalism, prices determine the demand-supply scale. For example, higher demand for certain goods and services lead to higher prices and lower demand for certain goods lead to lower prices.', 0],
          ['Represent the Wikipedia document for retrieval; Input: ',"The disparate impact theory is especially controversial under the Fair Housing Act because the Act regulates many activities relating to housing, insurance, and mortgage loansâ€”and some scholars have argued that the theory's use under the Fair Housing Act, combined with extensions of the Community Reinvestment Act, contributed to rise of sub-prime lending and the crash of the U.S. housing market and ensuing global economic recession",0],
          ['Represent the Wikipedia document for retrieval; Input: ','Disparate impact in United States labor law refers to practices in employment, housing, and other areas that adversely affect one group of people of a protected characteristic more than another, even though rules applied by employers or landlords are formally neutral. Although the protected classes vary by statute, most federal civil rights laws protect based on race, color, religion, national origin, and sex as protected traits, and some laws include disability status and other traits as well.',0]]
query_embeddings = model.encode(query)
corpus_embeddings = model.encode(corpus)
similarities = cosine_similarity(query_embeddings,corpus_embeddings)
retrieved_doc_id = np.argmax(similarities)
print(retrieved_doc_id)
```

## Clustering
Use **customized embeddings** for clustering texts in groups.
```python
import sklearn
sentences = [['Represent the Medicine sentence for clustering; Input: ','Dynamical Scalar Degree of Freedom in Horava-Lifshitz Gravity', 0],
             ['Represent the Medicine sentence for clustering; Input: ','Comparison of Atmospheric Neutrino Flux Calculations at Low Energies', 0],
             ['Represent the Medicine sentence for clustering; Input: ','Fermion Bags in the Massive Gross-Neveu Model', 0],
             ['Represent the Medicine sentence for clustering; Input: ',"QCD corrections to Associated t-tbar-H production at the Tevatron",0],
             ['Represent the Medicine sentence for clustering; Input: ','A New Analysis of the R Measurements: Resonance Parameters of the Higher,  Vector States of Charmonium',0]]
embeddings = model.encode(sentences)
clustering_model = sklearn.cluster.MiniBatchKMeans(n_clusters=2)
clustering_model.fit(embeddings)
cluster_assignment = clustering_model.labels_
print(cluster_assignment)
```