# BioGraphletQA Dataset Documentation

This document provides a detailed description of the data files associated with the paper **"BioGraphletQA: Knowledge-Anchored Generation of Complex QA Datasets"**.

This repository contains the full dataset, intermediate outputs from our generation pipeline, and the rephrased subsets used for downstream task evaluation.

-----

## 1\. Generation Templates

This file contains the raw graphlets extracted from the OREGANO knowledge graph, which serve as the factual basis for the entire generation process.

  * **File:** `generation_templates/templates.jsonl`
  * **Total Entries:** 269,574
  * **Description:** Each line is a JSON object representing a single graphlet (a small, connected subgraph).

### Schema: `templates.jsonl`

| Key | Type | Description |
| :--- | :--- | :--- |
| `id` | Integer | A unique identifier for the graphlet. |
| `nodes` | List[String] | A list of node IDs from the OREGANO KG (e.g., "COMPOUND:78"). |
| `edges` | List[List[Int]]| An edge list representing the connections between nodes, using indices from the `nodes` list. |
| `graphlet_id` | Integer | An ID corresponding to one of the 29 unique graphlet shapes. |
| `edge_labels` | List[String] | The relation type for each edge. |
| `node_names`| List[String] | The human-readable names corresponding to the node IDs. |
| `graphlet_text`| String | A formatted string listing the nodes and their names for easy readability. |
| `num_nodes`| Integer | The total number of nodes in the graphlet. |

#### Example:

```json
{
  "id": 0,
  "nodes": ["EFFECT:46", "COMPOUND:78", "COMPOUND:669", "COMPOUND:8321"],
  "edges": [[2, 0], [2, 1], [3, 0], [3, 1], [1, 0], [1, 2], [1, 3]],
  "graphlet_id": 7,
  "edge_labels": ["has_effect", "increase_efficacy", "has_effect", ...],
  "node_names": ["neuromuscular blockade", "Botulinum_toxin", "Tobramycin", ...],
  "graphlet_text": [" - 0: neuromuscular blockade \n", " - 1: Botulinum_toxin \n", ...],
  "num_nodes": 4
}
```

-----

## 2\. Raw Generated QA Pairs

This file contains the direct output from the initial LLM generation stage, after basic cleaning (JSON parsing validation and Z-score outlier removal).

  * **File:** `_llm_outs/DATASET/cleaned_dataset.jsonl`
  * **Total Entries:** 264,373
  * **Description:** Each line is a JSON object containing the original graphlet information plus the full text output from the LLM, which includes its reasoning, self-reflection, and the final generated question-answer pair.

### Schema: `cleaned_dataset.jsonl`

This file includes all fields from `templates.jsonl` plus the following:

| Key | Type | Description |
| :--- | :--- | :--- |
| `text` | String | The complete, raw text output from the LLM, including its analysis and reflection steps. |
| `question` | String | The final generated question. |
| `answer` | String | The final generated answer. |
| `json_parsed` | Boolean | A flag indicating that the LLM's output was successfully parsed as JSON. |

#### Example:

````json
{
  "id": 0,
  "nodes": ["EFFECT:46", ...],
  "edges": [[2, 0], ...],
  ...
  "text": "**Analysis of Graphlet**\n\n* **Question Node:** ...\n\n**Final QA Pair (Refined based on Reflection)**\n\n```json\n{\n  \"question\": \"In the context of neuromuscular blockade...\",\n  \"answer\": \"Botulinum toxin exposure could lead to...\"\n}\n```",
  "question": "In the context of neuromuscular blockade during surgery, identify a potential underlying condition that could lead to unexpectedly prolonged paralysis...",
  "answer": "Botulinum toxin exposure could lead to prolonged paralysis in patients undergoing neuromuscular blockade...",
  "json_parsed": true
}
````

-----

## 3\. Filtered Datasets

These files are the output of the LLM-as-a-judge filtering stage. The dataset is split into pairs that were accepted (deemed high-quality) and those that were rejected.

  * **File (Accepted):** `_llm_outs/FILTER_DATASET/filtered_dataset.jsonl`
      * **Total Entries:** 119,856
  * **File (Rejected):** `_llm_outs/FILTER_DATASET/rejected_dataset.jsonl`
      * **Total Entries:** 127,416
  * **Note:** An additional 17,101 pairs that were not parseable as JSON are excluded from these files.

### Schema: Filtered Files

These files include all fields from `cleaned_dataset.jsonl` plus the following, which contains the reasoning from the LLM filter:

| Key | Type | Description |
| :--- | :--- | :--- |
| `filtering_text` | String | The full text output from the LLM filter, including its own reasoning, its generated answer for comparison, and the final boolean judgments (`valid_question` and `original_answer_valid`). |

#### Example:

````json
{
  "id": 2,
  "nodes": ["PATHWAY:335", ...],
  ...
  "question": "In the context of cellular protein regulation, how might the removal of Orc1 from chromatin influence...",
  "answer": "The removal of Orc1 from chromatin may facilitate the proteasomal degradation of DVL...",
  ...
  "filtering_text": "Here is the evaluation in the requested JSON format:\n\n```\n{\n  \"question_reasoning\": \"Analysis of the question reveals...\",\n  \"valid_question\": true,\n  \"my_answer\": \"The removal of Orc1 from chromatin could disrupt...\",\n  \"answer_reasoning\": \"My answer is based on the understanding that Orc1...\",\n  \"original_answer_valid\": true\n}\n```\n\n**Breakdown of the Evaluation:**..."
}
````

-----

## 4\. Supporting Document Retrieval

This section describes the two files related to enriching QA pairs with documents from PubMed.

### 4.1. BM25 Retrieved Documents (Input for Annotation)

  * **File:** `_retrieval/shuffled_data_qa.jsonl`
  * **Description:** This file contains the top 10 PubMed abstracts retrieved by the BM25 algorithm for each QA pair. This serves as the **input** for the LLM annotation step, using the 2025 PubMed index.

#### Schema: `shuffled_data_qa.jsonl`

| Key | Type | Description |
| :--- | :--- | :--- |
| `id` | Integer | The unique ID of the QA pair. |
| `query_text` | String | The concatenated question and answer used as a query for retrieval. |
| `bm25` | List[Dict] | A list of the top 10 retrieved documents. Each dictionary contains the PubMed `id` and the full `text` of the title and abstract. |

#### Example:

```json
{
  "id": 2,
  "query_text": "In the context of cellular protein regulation, how might the removal of Orc1 from chromatin influence protein degradation pathways...",
  "bm25": [
    {"id": "27427963", "text": "The Integrated Role of Wnt/\u03b2-Catenin, N-Glycosylation, and E-Cadherin-Mediated Adhesion..."},
    {"id": "37633647", "text": "Ubiquitination and deubiquitination: Implications on cancer therapy..."},
    ...
  ]
}
```

### 4.2. LLM-Annotated Documents (Final Output)

  * **File:** `_retrieval/retrieval_annotated_data.jsonl`
  * **Description:** This file is the **output** of the annotation process. It contains the QA pairs, the original retrieved documents, and the LLM's relevance judgment and extracted snippets for each document.

#### Schema: `retrieval_annotated_data.jsonl`

This file includes all fields from the input file plus `processed_documents`.

| Key | Type | Description |
| :--- | :--- | :--- |
| `processed_documents` | List[Dict] | The LLM's annotation. Each dictionary contains the PubMed `id`, a boolean `relevant` flag, and a list of `snippets` (strings) extracted by the LLM if relevant. |

#### Example:

```json
{
  "id": 2,
  "query_text": "In the context of cellular protein regulation, how might the removal of Orc1 from chromatin influence protein degradation pathways...",
  "bm25": [
    {"id": "27427963", "text": "The Integrated Role of Wnt/\u03b2-Catenin..."},
    ...
  ],
  "processed_documents": [
    {"id": "27427963", "relevant": false, "snippets": []},
    {"id": "33257473", "relevant": true, "snippets": ["In addition, it was recently revealed that Wnt signaling can also act independently of \u03b2-catenin..."]},
    ...
  ]
}
```

-----

## 5\. Rephrased Datasets for Downstream Tasks

These files contain a subset of 20,000 QA pairs that have been automatically rephrased to match the format of specific downstream benchmarks.

  * **File (MedQA):** `_rephrase/20_000_qwen3_MedQA_rephrase.json`
  * **File (PubMedQA):** `_rephrase/20_000_qwen3_PubMedQA_rephrase.json`
  * **Note:** These files are saved with a `.jsonl` extension for compatibility with BioLinkBERT but are formatted as single, large JSON objects.

### Schema: Rephrased Files

Each file is a single JSON object where keys are the `id` of the original QA pair.

| Key | Type | Description |
| :--- | :--- | :--- |
| `id` | Integer | The unique ID of the original QA pair. |
| `qa` | String | The original question and answer, concatenated. |
| `medQA_similar` | List[String] | A list of five example questions from the target dataset, used to guide the LLM's rephrasing. |
| `snippets` | List[String] | The supporting text snippets from the retrieval stage. |
| `medQA_rephrase` | Dict | A dictionary containing the newly generated multiple-choice question (`question`), five options (`opa` to `ope`), the `reasoning` for the correct answer, and the correct option letter (`cop`). |

#### Example (MedQA):

```json
{
  "2": {
    "id": 2,
    "qa": "In the context of cellular protein regulation, how might the removal of Orc1...",
    "medQA_similar": ["A young immigrant girl presents with low-grade fever..."],
    "snippets": ["In addition, it was recently revealed that Wnt signaling..."],
    "medQA_rephrase": {
      "question": "The removal of Orc1 from chromatin is associated with the proteasomal degradation of DVL... Which of the following 26S proteasome subunits is most directly involved in this process?",
      "opa": "a) ATPase subunit Rpt1...",
      "opb": "b) Ubiquitin ligase SCF complex...",
      "opc": "c) Non-ATPase subunit 6 and SEM1, which mediate DVL degradation",
      "opd": "d) Autophagy-related protein ATG7...",
      "ope": "e) LRP6 kinase...",
      "reasoning": "The correct answer is (c) because the 26S proteasome's non-ATPase subunits 6 and SEM1 are specifically implicated...",
      "cop": "c"
    }
  }
}
```