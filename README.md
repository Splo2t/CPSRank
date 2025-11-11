# CPSRank:Unsupervised Keyphrase Extraction via Contextual Perturbation (CIKM '25). paper: [ACM Digital Library — DOI:10.1145/3746252.3760945](https://dl.acm.org/doi/10.1145/3746252.3760945)
- The importance of a phrase within a document becomes most evident through its absence rather than its presence. Inspired by this observation, we redefine keyphrases as those whose removal most disrupts the document's meaning. Traditional unsupervised methods typically rely on document-level signals, such as term frequency or phrase-to document similarity, which overlook the contextual contribution of a phrase. This paper proposes CPSRank, an unsupervised keyphrase extraction method that evaluates the semantic importance of candidate phrases via a contextual perturbation score (CPS). The CPS quantifies the critical role of each phrase by combining contextual perturbation and content loss. CPSRank outperforms existing baselines in terms of F1 scores while providing deeper insights into the semantic value of keyphrases. 

## Requirements (Thanks to [SAMRank](https://github.com/kangnlp/SAMRank))

- [stanford-corenlp-full-2018-02-27](https://drive.google.com/file/d/1K4Ll54ypTf_tF83Mkkar2QKOcZ4Uskl5/view?usp=sharing)  (please download the .zip file and extract it)

Run 'stanford-corenlp-full-2018-02-27' on your computer's terminal using the following command:

```
(1) cd stanford-corenlp-full-2018-02-27/

(2) java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
    -preload tokenize,ssplit,pos -status_port 9000 -port 9000 -timeout 15000 &
```

- transformers  
- pytorch  
- nltk  
- pandas  
- tqdm  

---

## Installation

```bash
# (권장) 새 가상환경
python -m venv .venv && source .venv/bin/activate   # Windows는 .venv\Scripts\activate

pip install -U pip
pip install torch transformers pandas numpy tqdm nltk
```

Please make sure to run the Stanford CoreNLP server in a **separate terminal** (see the Requirements above).

---

## Data format

Each dataset should be stored in data/*.jsonl as line-delimited JSON objects (one JSON object per line).

```json
{"text": "document text ...", "keyphrases": ["term1", "term2", "term3"]}
```

---

## Running

```bash
python cpsrank.py \
  --dataset Inspec \
  --plm BERT \
  --alpha 0.5 \
  --beta 0.6 \
  --model_name bert-base-uncased
```

### Arguments
- `--dataset` : `Inspec` or `SemEval2017`
- `--plm` : `BERT`
- `--model_name` : BERT masked‑LM checkpoint (e.g., `bert-base-uncased`)
- `--alpha` (float, 0–1): weight balancing masked‑token distance vs mean distance of non‑masked tokens
- `--beta` (float, 0–1): weight for combining embedding‑based score with LM‑CPS score

### Outputs
- `experiment_results/<DATASET>/BERT_short.csv` : table containing @5/@10/@15 metrics
- `experiment_results/<DATASET>/log_BERT.txt` : run log (hyperparameters / top-3 layers, etc.)

---

## Notes

- If a GPU is available, the script will automatically use `cuda`; otherwise it will run on `cpu`.
- If `UGIR_stopwords.txt` is missing, a warning is printed and an empty list is used.
- The script uses the hidden states of layer 8 (0-based). If needed, change `target_layer` in the code.

--- 

## Relation to SAMRank

This implementation is based on [CPSRank](https://dl.acm.org/doi/10.1145/3746252.3760945) and also draws on [SAMRank](https://github.com/kangnlp/SAMRank) as background research.

---

## CPSRank Citation

```
@inproceedings{10.1145/3746252.3760945,
author = {Yu, Hyunwook and Kim, Minju and Kim, Euijin and Kim, Mucheol},
title = {CPSRank: Unsupervised Keyphrase Extraction via Contextual Perturbation},
year = {2025},
isbn = {9798400720406},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3746252.3760945},
doi = {10.1145/3746252.3760945},
abstract = {The importance of a phrase within a document becomes most evident through its absence rather than its presence. Inspired by this observation, we redefine keyphrases as those whose removal most disrupts the document's meaning. Traditional unsupervised methods typically rely on document-level signals, such as term frequency or phrase-to document similarity, which overlook the contextual contribution of a phrase. This paper proposes CPSRank, an unsupervised keyphrase extraction method that evaluates the semantic importance of candidate phrases via a contextual perturbation score (CPS). The CPS quantifies the critical role of each phrase by combining contextual perturbation and content loss. CPSRank outperforms existing baselines in terms of F1 scores while providing deeper insights into the semantic value of keyphrases. We release our code at https://github.com/Splo2t/CPSRank.},
booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
pages = {5454–5458},
numpages = {5},
keywords = {contextual perturbation, information extraction, keyphrase extraction, semantic importance, unsupervised learning},
location = {Seoul, Republic of Korea},
series = {CIKM '25}
}
```
