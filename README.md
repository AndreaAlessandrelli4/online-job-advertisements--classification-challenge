# Web Intelligence - Classification Challenge

This repository contains the codebase for the team **"theitalianjob"** participating in the ["THE CLASSIFICATION OF OCCUPATIONS FOR ONLINE JOB ADVERTISEMENTS CHALLENGE – The second round of the European Statistics Awards for Web Intelligence"](https://statistics-awards.eu/competitions/12#participate-submit_results) organized by Eurostat.

The core workflow is based on a modular pipeline integrating retrieval systems, language models, and classification logic.

# Dataset Overview

The dataset used consists of online job advertisements with various text fields, which required processing and enrichment to be compatible with the ESCO classification framework. External data used includes:

* **ESCO Dataset**: Occupations, titles, descriptions, required skills, and category codes.
* **Job Ads**: Original text-based ads in multiple languages and formats.

# Methodology Overview

The task was to classify job advertisements into ISCO categories. Our approach combines hybrid search, retrieval-augmented generation (RAG), and large language models (LLMs). Below are the major steps:

### 1. **Build the ESCO Dataset**

* Merged ESCO tables to create a structured dataset with:

  * Occupation Titles & Alternatives
  * Descriptions & Skills
  * Category Titles & Codes
* **Execution time**: < 1 minute

### 2. **Translate, Clean, and Extract Features**

* Used **gemma-2-9b-it-SimPO** LLM to:

  * Translate ads into English
  * Extract key fields: title, description, skills, experience, industry
* **Execution time**: \~24 hours (on NVIDIA H100 80GB)

### 3. **Build Weaviate Index**

* Created a hybrid semantic-keyword search index using:

  * `all-mpnet-base-v2` for vector embedding
* **Execution time**: < 5 minutes

### 4. **RAG + Reranking**

* Queried Weaviate to retrieve top 10 relevant occupations (alpha = 0, 0.5, 1)
* Reranked with `bge-reranker-large`
* Matched job titles with ESCO database for extra candidates
* Constructed prompts for the LLM with grouped and formatted occupations
* LLM assigned the final **ISCO-4-digit code**
* **Execution time**: \~10 hours

# How to Run the Pipeline

To perform the classification using our pipeline, follow these steps:

1. **Create a Weaviate sandbox**

   * Sign up and launch a sandbox at [weaviate.io](https://weaviate.io/)

2. **Index the ISCO occupation data**

   * Update `indexing_en.py` with your sandbox URL and API key
   * Run it to index all occupation descriptions

3. **Extract features from job ads**

   * Run `feature.py` and `query_feature.py` using the model `princeton-nlp/gemma-2-9b-it-SimPO`
   * This step translates and extracts structured features like title, skills, industry, and experience

4. **Retrieve and rerank matching occupations**

   * Run `embedding.py` to search the Weaviate index and group results under their corresponding 4-digit ISCO categories

5. **Classify job ads into ISCO categories**

   * Either:

     * Run `Gemma_classification.py` locally with `gemma-2-9b-it-SimPO`, or
     * Run `Chat_classification.py` using OpenAI's `gpt-4o-mini` API

# [Architecture Overview](images/)

The pipeline is modular and includes:

* Data pre-processing
* Retrieval (hybrid search via Weaviate)
* Feature extraction using LLMs
* RAG-based classification and reranking
  This modular design ensures **scalability**, **configurability**, and **interpretability**.

# Hardware Specifications

* **Machine 1**: 112 physical cores, NVIDIA H100 80GB GPU, 2TB RAM
* **Machine 2**: Apple GPU 8GB, 8-core CPU, 8GB RAM

# Libraries Used

* `numpy`
* `pandas`
* `torch`
* `transformers`
* `sentence_transformers`
* `FlagEmbedding`
* `weaviate`
* `OpenAI` (optional)

# Repository Structure

* [`src/`](src/) - Contains all the source code.
* [`assets/`](assets/) Contains the project's documents regarding the reproducibility award (i.e., reproducibility_approach_description).
* [`images/`](images/) - Diagrams and architecture visuals
* [`requirements.txt`](requirements.txt) Contains the project's requirements.

# Originality & Future Directions

* Leverages **open-source LLMs** (e.g. gemma-2-9b-it-SimPO)
* Demonstrates that **RAG reduces hallucinations** and enhances classification precision
* Shows that **English-only retrieval** on translated ads outperforms multilingual retrieval
* Suggests potential for human-in-the-loop systems for scalable classification


# Authors

* [Andrea Alessandrelli](mailto:a.alessandrelli@studenti.unipi.it)
* [Pasquale Maritato](mailto:pasquale.maritato@outlook.com)
