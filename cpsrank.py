#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keyphrase ranking (short documents only) using BERT hidden states + LM CPS.
- Removed rank_long_documents and all long-doc utilities
- Removed GPT-2 code path (BERT-only; uses [MASK])
- Deduplicated helpers and cleaned imports
- Safer stopword loading; clearer CLI
- Keeps evaluation + CSV/log saving identical to original behavior

Prereqs
-------
1) Stanford CoreNLP server running (tokenize, ssplit, pos):
   java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
        -preload tokenize,ssplit,pos -status_port 9000 -port 9000 -timeout 15000 &

2) Python deps (examples):
   pip install torch transformers pandas numpy tqdm nltk
   pip install swisscom_ai_research_keyphrase

Data format
-----------
Each line in data/*.jsonl is a JSON object with fields:
  - "text": str
  - "keyphrases": List[str]
"""

import argparse
import datetime
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from nltk.stem import PorterStemmer
from transformers import AutoModelForMaskedLM, AutoTokenizer

from swisscom_ai.research_keyphrase.preprocessing.postagging import PosTaggingCoreNLP
from swisscom_ai.research_keyphrase.model.input_representation import InputTextObj
from swisscom_ai.research_keyphrase.model.extractor import extract_candidates

# ---------------------------
# Globals & lightweight utils
# ---------------------------

HOST = "localhost"
PORT = 9000
stemmer = PorterStemmer()


def load_stopwords(path: str):
    """Load stopwords from file; return [] if file missing."""
    stop = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    stop.append(line)
    except FileNotFoundError:
        print(f"[warn] Stopword file not found: {path}. Proceeding with empty list.")
    return stop


STOPWORDS = load_stopwords("UGIR_stopwords.txt")


def read_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def get_candidates(core_nlp: PosTaggingCoreNLP, text: str):
    """POS tag + extract candidate phrases via swisscom_ai toolkit."""
    tagged = core_nlp.pos_tag_raw_text(text)
    text_obj = InputTextObj(tagged, "en")
    return extract_candidates(text_obj)


def get_phrase_indices(text_tokens, phrase: str, subword_prefix: str):
    """
    Map a (whitespace) phrase to (start,end) token spans over subword tokens.
    - Works by matching the de-prefixed subwords with the phrase (spaces removed).

    Returns: List[[start_idx, end_idx)) -- inclusive-exclusive spans
    """
    tokens = [t.replace(subword_prefix, "") for t in text_tokens]
    compact_phrase = phrase.replace(" ", "")

    matched_indices = []
    cur_match = []
    target = compact_phrase

    for i, tok in enumerate(tokens):
        sub_len = min(len(tok), len(target))
        if tok[:sub_len].lower() == target[:sub_len]:
            cur_match.append(i)
            target = target[sub_len:]
            if not target:
                matched_indices.append([cur_match[0], cur_match[-1] + 1])
                target = compact_phrase
                cur_match = []
        else:
            # reset and try again with this token as a fresh start
            cur_match = []
            target = compact_phrase
            if tok[:sub_len].lower() == target[:sub_len]:
                cur_match.append(i)
                target = target[sub_len:]
                if not target:
                    matched_indices.append([cur_match[0], cur_match[-1] + 1])
                    target = compact_phrase
                    cur_match = []

    return matched_indices


def remove_repeated_sub_word(candidates_pos_dict: dict):
    """
    If a multi-word phrase contains a single-word candidate fully inside its span,
    drop those *embedded* single-word positions to reduce duplication noise.
    """
    for phrase in list(candidates_pos_dict.keys()):
        parts = phrase.split()
        if len(parts) > 1:
            for word in parts:
                if word in candidates_pos_dict:
                    single_positions = candidates_pos_dict[word]
                    phrase_positions = candidates_pos_dict[phrase]
                    filtered = [
                        pos
                        for pos in single_positions
                        if not any(pos[0] >= pp[0] and pos[1] <= pp[1] for pp in phrase_positions)
                    ]
                    candidates_pos_dict[word] = filtered
    return candidates_pos_dict


def cal_score(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Pairwise cosine *distance* (1 - cosine similarity) between rows of a and b.
    Shapes:
      a: [n, d]
      b: [n, d]
    Returns:
      [n] distances scaled by 100
    """
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    a_norm = a / (a.norm(dim=1, keepdim=True) + 1e-8)
    b_norm = b / (b.norm(dim=1, keepdim=True) + 1e-8)
    sim = (a_norm * b_norm).sum(dim=1)
    return (1.0 - sim) * 100.0

def _parse_layer_spec(spec: str, num_hidden_states: int):
    """Parse a layer spec like '8', 'all', '1,4,8', or '1-4,8,10-12'.
    Note: hidden_states length == num_hidden_layers + 1 (index 0 = embeddings).
    Returns: sorted list of valid layer indices (0-based)."""
    spec = str(spec).strip().lower()
    if spec in ("all", "*"):
        return list(range(num_hidden_states))
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                a = int(a.strip()); b = int(b.strip())
                if a <= b:
                    for k in range(a, b + 1):
                        if 0 <= k < num_hidden_states:
                            out.add(k)
            except ValueError:
                continue
        else:
            try:
                k = int(part)
                if 0 <= k < num_hidden_states:
                    out.add(k)
            except ValueError:
                continue
    res = sorted(out)
    if not res:
        # fallback: layer 8 if available, else last layer
        res = [8] if num_hidden_states > 8 else [num_hidden_states - 1]
    return res



# ---------------------------
# Evaluation helpers (unchanged)
# ---------------------------

def get_score_full(candidates, references, maxDepth=15):
    precision, recall = [], []
    reference_set = set(references)
    referencelen = len(reference_set)
    true_positive = 0
    for i in range(maxDepth):
        if len(candidates) > i:
            kp_pred = candidates[i]
            if kp_pred in reference_set:
                true_positive += 1
            precision.append(true_positive / float(i + 1))
            recall.append(true_positive / float(referencelen if referencelen else 1))
        else:
            denom = float(len(candidates)) if len(candidates) else 1.0
            precision.append(true_positive / denom)
            recall.append(true_positive / float(referencelen if referencelen else 1))
    return precision, recall


def evaluate(predictions, references):
    results = {}
    precision_scores = {5: [], 10: [], 15: []}
    recall_scores = {5: [], 10: [], 15: []}
    f1_scores = {5: [], 10: [], 15: []}

    for pred, ref in zip(predictions, references):
        p_arr, r_arr = get_score_full(pred, ref)
        for k in [5, 10, 15]:
            p, r = p_arr[k - 1], r_arr[k - 1]
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
            precision_scores[k].append(p)
            recall_scores[k].append(r)
            f1_scores[k].append(f1)

    print("########################\nMetrics")
    for k in [5, 10, 15]:
        p = np.mean(precision_scores[k]) if precision_scores[k] else 0.0
        r = np.mean(recall_scores[k]) if recall_scores[k] else 0.0
        f1 = np.mean(f1_scores[k]) if f1_scores[k] else 0.0
        print(f"@{k}\nF1:{f1}\nP:{p}\nR:{r}")
        results[f"precision@{k}"] = p
        results[f"recall@{k}"] = r
        results[f"f1@{k}"] = f1
    print("#########################")
    return results


def evaluate_all_heads(layer_predicted_top15, dataset, args):
    experiment_results = []
    for layer, preds_per_doc in layer_predicted_top15.items():
        gold_keyphrase_list = []
        predicted_keyphrase_list = []
        for i in range(len(dataset)):
            pred = [phrase.lower() for phrase in preds_per_doc[i]]
            predicted_keyphrase_list.append(pred)
            gold = [key.lower() for key in dataset[i]["keyphrases"]]
            gold_keyphrase_list.append(gold)

        total_score = evaluate(predicted_keyphrase_list, gold_keyphrase_list)
        total_score["layer"] = layer + 1
        experiment_results.append(total_score)

    df = pd.DataFrame(experiment_results)
    out_dir = f"experiment_results/{args.dataset}/"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(f"{out_dir}{args.plm}_{args.mode}.csv", index=False)

    top3_f1_5 = df.nlargest(3, "f1@5").reset_index(drop=True)
    top3_f1_10 = df.nlargest(3, "f1@10").reset_index(drop=True)
    top3_f1_15 = df.nlargest(3, "f1@15").reset_index(drop=True)
    return top3_f1_5, top3_f1_10, top3_f1_15


# ---------------------------
# Scoring (LM CPS for masked spans)
# ---------------------------

def compute_lm_cps(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    sentence: str,
    spans: list,            # List[(start_idx,end_idx)] over tokenized input (inclusive-exclusive)
    device: str = "cpu",
    length_penalty: bool = True,
):
    """
    LM CPS = max masked-token CE - original average CE, optionally length-normalized.
    """
    enc = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]          # [1, seq_len]
    attention_mask = enc["attention_mask"]

    with torch.no_grad():
        out_orig = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    L_orig = out_orig.loss.item()

    masked_ids = input_ids.clone()
    labels = torch.full_like(masked_ids, -100)
    total_len = 0
    for s, e in spans:
        labels[0, s:e] = input_ids[0, s:e]
        masked_ids[0, s:e] = tokenizer.mask_token_id
        total_len += (e - s)

    with torch.no_grad():
        out_mask = model(input_ids=masked_ids, attention_mask=attention_mask, labels=labels)

    logits = out_mask.logits
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    flattened_logits = logits.view(-1, logits.size(-1))
    flattened_labels = labels.view(-1)
    losses = loss_fct(flattened_logits, flattened_labels)
    valid_losses = losses[flattened_labels != -100]

    # max loss among masked tokens (as in original script)
    L_mask = valid_losses.max().item() if valid_losses.numel() > 0 else 0.0
    cps = L_mask - L_orig
    if length_penalty and total_len > 0:
        cps /= np.log1p(total_len)
    return cps


# ---------------------------
# Core: Rank short documents
# ---------------------------

def rank_short_documents(args, dataset, model, tokenizer, pos_tagger):
    """
    For each document:
      1) Extract candidate phrases with CoreNLP POS
      2) Build a batch: original + one copy per candidate where candidate span is [MASK]ed
      3) Score each candidate with (alpha * max_mask_dist + (1-alpha) * avg_normal_dist)
      4) Combine with LM CPS via beta
      5) Select top-15 stemmed phrases
    """
    assert args.plm.upper() == "BERT", "This cleaned script supports BERT only."

    subword_prefix = "##"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    model.to(device)
    model.eval()

    layer_predicted_top15 = defaultdict(list)

    for data in tqdm(dataset):
        with torch.no_grad():
            org_sentence = data["text"]

            # 1) candidates
            candidates = get_candidates(pos_tagger, org_sentence)
            candidates = [p for p in candidates if p and p.split(" ")[0] not in STOPWORDS]

            # Tokenize original once to map phrase -> spans
            tok_single = tokenizer(org_sentence, return_tensors="pt")
            text_tokens = tokenizer.convert_ids_to_tokens(tok_single["input_ids"].squeeze(0))

            # 2) map candidate -> list of spans (inclusive-exclusive)
            cand2spans = {}
            for phrase in candidates:
                spans = get_phrase_indices(text_tokens, phrase, subword_prefix)
                if spans:
                    cand2spans[phrase] = spans

            # prune embedded single word spans inside multi-word phrases
            cand2spans = remove_repeated_sub_word(cand2spans)

            if not cand2spans:
                # if no mappable candidates, keep empty prediction entry
                layer_predicted_top15[8].append([])
                continue

            # 3) Build masked batch: [orig] + one sample per candidate
            sentences = [org_sentence] * (len(cand2spans) + 1)
            inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

            # Replace candidate spans with [MASK] in each copy
            idx2phrase = {}
            mask_id = tokenizer.mask_token_id
            now_idx = 1
            for phrase, spans in cand2spans.items():
                idx2phrase[now_idx] = phrase
                for s, e in spans:
                    inputs["input_ids"][now_idx][s:e] = mask_id
                now_idx += 1

            # 4) Forward
            outputs = model(**{k: v.to(device) for k, v in inputs.items()})
            hidden_states = outputs.hidden_states  # tuple(layer) of [batch, seq, h]

            # Select target layers (0-based over hidden_states)
            num_hs = len(hidden_states)
            target_layers = _parse_layer_spec(getattr(args, "layers", "8"), num_hs)

            for target_layer in target_layers:
                scores = {}

                for sent_idx in range(1, len(sentences)):  # skip the original at index 0
                    phrase = idx2phrase[sent_idx]
                    spans = cand2spans[phrase]

                    attn_mask = inputs["attention_mask"][sent_idx]
                    valid_indices = [i for i, m in enumerate(attn_mask.tolist()) if m == 1]

                    mask_token_indices = []
                    for s, e in spans:
                        mask_token_indices.extend(list(range(s, e)))

                    mask_indices = [i for i in valid_indices if i in mask_token_indices]
                    normal_indices = [i for i in valid_indices if i not in mask_token_indices]
                    if not mask_indices:
                        continue

                    lm_cps = compute_lm_cps(
                        model, tokenizer, org_sentence, spans, device=device, length_penalty=True
                    )

                    ref_emb = hidden_states[target_layer][0]         # [seq, h]
                    sent_emb = hidden_states[target_layer][sent_idx] # [seq, h]

                    normal_dists = cal_score(ref_emb[normal_indices], sent_emb[normal_indices]).tolist() if normal_indices else []
                    mask_dists = cal_score(ref_emb[mask_indices], sent_emb[mask_indices]).tolist()
                    if not mask_dists:
                        continue

                    alpha = float(args.alpha)
                    beta = float(args.beta)
                    max_mask = max(mask_dists)
                    avg_normal = (sum(normal_dists) / len(normal_dists)) if normal_dists else 0.0
                    combined = alpha * max_mask + (1 - alpha) * avg_normal
                    final_score = beta * combined + (1 - beta) * lm_cps
                    scores[phrase] = final_score

                # Rank & stem for this layer
                ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                stemmed_ranked = []
                for phrase, val in ordered:
                    stemmed = " ".join(stemmer.stem(w) for w in phrase.split())
                    if stemmed not in stemmed_ranked:
                        stemmed_ranked.append(stemmed)
                    if len(stemmed_ranked) >= 15:
                        break

                layer_predicted_top15[target_layer].append(stemmed_ranked)


            torch.cuda.empty_cache()

    # Evaluate & save
    top3_f1_5, top3_f1_10, top3_f1_15 = evaluate_all_heads(layer_predicted_top15, dataset, args)
    print("top@5_f1  Top3 layers:")
    print(top3_f1_5[["f1@5", "f1@10", "f1@15", "layer"]].to_string(index=False))
    print("top@10_f1 Top3 layers:")
    print(top3_f1_10[["f1@5", "f1@10", "f1@15", "layer"]].to_string(index=False))
    print("top@15_f1 Top3 layers:")
    print(top3_f1_15[["f1@5", "f1@10", "f1@15", "layer"]].to_string(index=False))

    # Append run log
    save_log_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"experiment_results/{args.dataset}/"
    os.makedirs(out_dir, exist_ok=True)
    log_filename = os.path.join(out_dir, f"log_{args.plm}.txt")
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write("------------------------------\n")
        f.write(f"Alpha: {args.alpha}, Beta: {args.beta}, time: {save_log_time}\n")
        f.write("Top@5_f1  Top3 layers:\n")
        f.write(top3_f1_5[["f1@5", "f1@10", "f1@15", "layer"]].to_string(index=False) + "\n\n")
        f.write("Top@10_f1 Top3 layers:\n")
        f.write(top3_f1_10[["f1@5", "f1@10", "f1@15", "layer"]].to_string(index=False) + "\n\n")
        f.write("Top@15_f1 Top3 layers:\n")
        f.write(top3_f1_15[["f1@5", "f1@10", "f1@15", "layer"]].to_string(index=False) + "\n")


# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Rank keyphrases for short documents (BERT-only).")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="One of: Inspec, SemEval2017 (short-document datasets only).",
    )
    parser.add_argument(
        "--plm",
        type=str,
        default="BERT",
        help="Only 'BERT' is supported in this short-doc-only script.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="short",
        help="Kept for compatibility with CSV naming; not functionally used.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for combining max-mask and avg-normal distances (0~1).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.6,
        help="Weight for combining embedding score and LM CPS (0~1).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="BERT masked-LM checkpoint (must support [MASK]).",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layer spec over hidden_states: '8', 'all', '1,4,8', or '1-4,8,10-12' (0-based).",
    )
    args = parser.parse_args()

    # Only short-document datasets allowed here
    dataset_map = {
        "Inspec": ("data/Inspec.jsonl", "short"),
        "SemEval2017": ("data/SemEval2017.jsonl", "short"),
    }
    key = args.dataset
    if key not in dataset_map:
        raise SystemExit(f"Invalid dataset '{key}'. Use one of: {', '.join(dataset_map.keys())}.")

    data_path, doc_type = dataset_map[key]
    if doc_type != "short":
        raise SystemExit(f"Dataset '{key}' is not a short-document dataset. This script only supports short docs.")

    dataset = read_jsonl(data_path)

    # Initialize CoreNLP POS tagger (requires running server)
    pos_tagger = PosTaggingCoreNLP(HOST, PORT)

    # BERT (masked LM) with hidden states
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name, output_hidden_states=True)

    # Run
    rank_short_documents(args, dataset, model, tokenizer, pos_tagger)


if __name__ == "__main__":
    main()
