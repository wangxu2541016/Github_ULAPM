import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.plain_multitask import PlainMultiTask
from src.models.ulapm import build_ulapm_from_artifacts
from src.eval_robot_protocol import load_dotenv_if_needed, load_llm_cache, predict_b3_llm
from src.paper_planner import planner_project_prediction


_REAL_AUTO_FROM_PRETRAINED = AutoModel.from_pretrained


def _offline_auto_from_pretrained(*args, **kwargs):
    kwargs.setdefault("local_files_only", True)
    return _REAL_AUTO_FROM_PRETRAINED(*args, **kwargs)


AutoModel.from_pretrained = _offline_auto_from_pretrained


EMO7 = ["Joy", "Sadness", "Anger", "Fear", "Surprise", "Disgust", "Neutral"]
BEHAVIORS = ["HUG", "VERBAL_COMFORT", "LISTEN", "CONGRATULATE", "NEUTRAL"]
DISTANCE_ZONES = ["intimate", "close", "personal", "far"]

EMO_TO_ID = {name: idx for idx, name in enumerate(EMO7)}
BEHAVIOR_TO_ID = {name: idx for idx, name in enumerate(BEHAVIORS)}
ZONE_TO_ID = {name: idx for idx, name in enumerate(DISTANCE_ZONES)}


B3_EMOTION_ALIASES = {
    "joy": "Joy",
    "happy": "Joy",
    "sadness": "Sadness",
    "sad": "Sadness",
    "anger": "Anger",
    "angry": "Anger",
    "fear": "Fear",
    "afraid": "Fear",
    "surprise": "Surprise",
    "surprised": "Surprise",
    "disgust": "Disgust",
    "disgusted": "Disgust",
    "neutral": "Neutral",
}


def acc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def macro_f1(y_true, y_pred, labels=None):
    try:
        from sklearn.metrics import f1_score

        kwargs = {"average": "macro"}
        if labels is not None:
            kwargs["labels"] = labels
        return float(f1_score(y_true, y_pred, **kwargs))
    except Exception:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if labels is None:
            labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        scores = []
        for c in labels:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            precision = tp / (tp + fp + 1e-12)
            recall = tp / (tp + fn + 1e-12)
            scores.append(2 * precision * recall / (precision + recall + 1e-12))
        return float(np.mean(scores)) if scores else float("nan")


def load_state_dict_robust(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        return sd["model"]
    return sd


def zone_from_distance(distance_m):
    d = float(distance_m)
    if d < 0.8:
        return "intimate"
    if d < 1.2:
        return "close"
    if d < 1.8:
        return "personal"
    return "far"


def normalize_b3_emotion(raw):
    if raw is None:
        return "Neutral"
    key = str(raw).strip().lower().replace("-", " ").replace("_", " ")
    key = " ".join(key.split())
    return B3_EMOTION_ALIASES.get(key, "Neutral")


def emotion_to_behavior_distance(c_pred):
    behavior = np.zeros_like(c_pred, dtype=np.int64)
    distance = np.zeros((len(c_pred),), dtype=np.float32)

    for i, c in enumerate(c_pred):
        emo = EMO7[int(c)]
        if emo == "Joy":
            behavior[i] = BEHAVIOR_TO_ID["CONGRATULATE"]
            distance[i] = 1.20
        elif emo == "Sadness":
            behavior[i] = BEHAVIOR_TO_ID["VERBAL_COMFORT"]
            distance[i] = 1.00
        elif emo == "Fear":
            behavior[i] = BEHAVIOR_TO_ID["LISTEN"]
            distance[i] = 1.40
        elif emo in {"Anger", "Disgust"}:
            behavior[i] = BEHAVIOR_TO_ID["LISTEN"]
            distance[i] = 1.60
        elif emo == "Surprise":
            behavior[i] = BEHAVIOR_TO_ID["NEUTRAL"]
            distance[i] = 1.50
        else:
            behavior[i] = BEHAVIOR_TO_ID["NEUTRAL"]
            distance[i] = 1.80

    return behavior, distance


def read_relabel_rows(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    valid = []
    for row in rows:
        if not (row.get("final_emotion", "") or "").strip():
            continue
        if not (row.get("final_behavior", "") or "").strip():
            continue
        if not (row.get("final_distance_zone", "") or "").strip():
            continue
        valid.append(row)
    return valid


def build_gold(rows):
    gold = {
        "sample_id": [],
        "source_index": [],
        "text": [],
        "emotion": [],
        "behavior": [],
        "distance_zone": [],
        "original_emotion": [],
        "original_behavior": [],
        "original_distance_zone": [],
    }
    for row in rows:
        gold["sample_id"].append((row.get("sample_id", "") or "").strip())
        gold["source_index"].append(int(row["source_index"]))
        gold["text"].append(row["text"])
        gold["emotion"].append(EMO_TO_ID[(row["final_emotion"] or "").strip()])
        gold["behavior"].append(BEHAVIOR_TO_ID[(row["final_behavior"] or "").strip()])
        gold["distance_zone"].append(ZONE_TO_ID[(row["final_distance_zone"] or "").strip()])
        gold["original_emotion"].append(EMO_TO_ID[(row["original_emotion"] or "").strip()])
        gold["original_behavior"].append(BEHAVIOR_TO_ID[(row["original_behavior"] or "").strip()])
        gold["original_distance_zone"].append(ZONE_TO_ID[(row["original_distance_zone"] or "").strip()])
    return gold


def batch_texts(tokenizer, texts, max_len):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )


def predict_stage1(
    model,
    tokenizer,
    texts,
    device,
    batch_size=32,
    max_len=128,
    distance_mode="used",
    seed=42,
    planner_projection=False,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    emo_pred = []
    beh_pred = []
    dist_pred = []
    latencies = []

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            enc = batch_texts(tokenizer, chunk, max_len=max_len)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)

            t0 = time.perf_counter()
            out = model(input_ids, attn)
            latencies.append((time.perf_counter() - t0) / max(1, len(chunk)))

            emo_pred.extend(torch.argmax(out["c_logits"], dim=-1).cpu().numpy().tolist())

            b_logits = out["b_logits"].cpu().numpy()
            sis_con = out["sis"].cpu().numpy() if "sis" in out else None
            if distance_mode == "used":
                dist = out["d"].cpu().numpy().reshape(-1)
            elif distance_mode == "raw":
                dist = out["d_raw"].cpu().numpy().reshape(-1)
            elif distance_mode == "posthoc":
                dist = torch.clamp(out["d_raw"], 0.3, 2.5).cpu().numpy().reshape(-1)
            else:
                raise ValueError(f"unsupported distance_mode: {distance_mode}")

            if planner_projection and sis_con is not None:
                for logits_row, sis_row, dist_val in zip(b_logits, sis_con, dist.tolist()):
                    planner = planner_project_prediction(
                        behavior_logits=logits_row,
                        distance_m=dist_val,
                        sis_vec=sis_row,
                        behaviors=BEHAVIORS,
                    )
                    beh_pred.append(int(planner["behavior_index"]))
                    dist_pred.append(float(planner["distance_m"]))
            else:
                beh_pred.extend(np.argmax(b_logits, axis=-1).tolist())
                dist_pred.extend(dist.tolist())

    return {
        "emotion": np.asarray(emo_pred, dtype=np.int64),
        "behavior": np.asarray(beh_pred, dtype=np.int64),
        "distance_m": np.asarray(dist_pred, dtype=np.float32),
        "latency_ms": 1000.0 * float(np.mean(latencies)),
    }


def predict_b2(model, tokenizer, texts, device, batch_size=32, max_len=128):
    emo_pred = []
    beh_pred = []
    dist_pred = []
    latencies = []

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            enc = batch_texts(tokenizer, chunk, max_len=max_len)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)

            t0 = time.perf_counter()
            out = model(input_ids, attn)
            latencies.append((time.perf_counter() - t0) / max(1, len(chunk)))

            emo_pred.extend(torch.argmax(out["c_logits"], dim=-1).cpu().numpy().tolist())
            beh_pred.extend(torch.argmax(out["b_logits"], dim=-1).cpu().numpy().tolist())
            dist_pred.extend(out["d"].cpu().numpy().reshape(-1).tolist())

    return {
        "emotion": np.asarray(emo_pred, dtype=np.int64),
        "behavior": np.asarray(beh_pred, dtype=np.int64),
        "distance_m": np.asarray(dist_pred, dtype=np.float32),
        "latency_ms": 1000.0 * float(np.mean(latencies)),
    }


def predict_b1(model, tokenizer, texts, device, batch_size=32, max_len=128, seed=42):
    stage1_pred = predict_stage1(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        batch_size=batch_size,
        max_len=max_len,
        distance_mode="used",
        seed=seed,
    )
    beh_pred, dist_pred = emotion_to_behavior_distance(stage1_pred["emotion"])
    return {
        "emotion": stage1_pred["emotion"],
        "behavior": beh_pred,
        "distance_m": np.asarray(dist_pred, dtype=np.float32),
        "latency_ms": stage1_pred["latency_ms"],
    }


def predict_b3(texts, args):
    load_dotenv_if_needed(args.env_file)
    cache = load_llm_cache(args.llm_cache)

    emo_pred = []
    beh_pred = []
    dist_pred = []
    latencies = []

    for idx, text in enumerate(texts):
        item = {
            "id": f"human420-{idx:04d}",
            "text": text,
        }
        pred = predict_b3_llm(item, args, cache)
        emo_name = normalize_b3_emotion(pred.get("llm_emotion"))
        beh_name = pred["behavior"] if pred.get("behavior") in BEHAVIOR_TO_ID else "NEUTRAL"
        dist_val = float(pred.get("distance_m", 1.5))

        emo_pred.append(EMO_TO_ID[emo_name])
        beh_pred.append(BEHAVIOR_TO_ID[beh_name])
        dist_pred.append(dist_val)
        latencies.append(float(pred.get("latency_ms", float("nan"))))

    return {
        "emotion": np.asarray(emo_pred, dtype=np.int64),
        "behavior": np.asarray(beh_pred, dtype=np.int64),
        "distance_m": np.asarray(dist_pred, dtype=np.float32),
        "latency_ms": float(np.nanmean(latencies)) if latencies else float("nan"),
    }


def add_zone_predictions(pred):
    zones = [ZONE_TO_ID[zone_from_distance(v)] for v in pred["distance_m"]]
    pred["distance_zone"] = np.asarray(zones, dtype=np.int64)
    pred["distance_out_of_range_rate"] = float(
        np.mean((pred["distance_m"] < 0.3) | (pred["distance_m"] > 2.5))
    )
    return pred


def score_prediction(pred, gold):
    emo_true = np.asarray(gold["emotion"], dtype=np.int64)
    beh_true = np.asarray(gold["behavior"], dtype=np.int64)
    zone_true = np.asarray(gold["distance_zone"], dtype=np.int64)

    emo_pred = np.asarray(pred["emotion"], dtype=np.int64)
    beh_pred = np.asarray(pred["behavior"], dtype=np.int64)
    zone_pred = np.asarray(pred["distance_zone"], dtype=np.int64)

    joint_exact = (emo_true == emo_pred) & (beh_true == beh_pred) & (zone_true == zone_pred)

    return {
        "emotion_acc": acc(emo_true, emo_pred),
        "emotion_macro_f1": macro_f1(emo_true, emo_pred),
        "behavior_acc": acc(beh_true, beh_pred),
        "behavior_macro_f1": macro_f1(beh_true, beh_pred),
        "distance_zone_acc": acc(zone_true, zone_pred),
        "distance_zone_macro_f1": macro_f1(zone_true, zone_pred),
        "joint_exact": float(joint_exact.mean()),
        "distance_out_of_range_rate": float(pred.get("distance_out_of_range_rate", 0.0)),
        "latency_ms": float(pred.get("latency_ms", float("nan"))),
    }


def build_long_rows(method_name, pred, gold):
    rows = []
    for i, sample_id in enumerate(gold["sample_id"]):
        rows.append(
            {
                "method": method_name,
                "sample_id": sample_id,
                "source_index": gold["source_index"][i],
                "gold_emotion": EMO7[gold["emotion"][i]],
                "pred_emotion": EMO7[int(pred["emotion"][i])],
                "gold_behavior": BEHAVIORS[gold["behavior"][i]],
                "pred_behavior": BEHAVIORS[int(pred["behavior"][i])],
                "gold_distance_zone": DISTANCE_ZONES[gold["distance_zone"][i]],
                "pred_distance_zone": DISTANCE_ZONES[int(pred["distance_zone"][i])],
                "pred_distance_m": f"{float(pred['distance_m'][i]):.4f}",
            }
        )
    return rows


def make_markdown(results, reference):
    lines = [
        "# Human-Adjudicated 420-Subset Evaluation",
        "",
        "## Compact Table",
        "",
        "| Method | Emotion Acc | Behavior Acc | Distance-Zone Acc | 3-way Exact |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        m = item["metrics"]
        lines.append(
            f"| {item['method']} | {100*m['emotion_acc']:.2f}% | {100*m['behavior_acc']:.2f}% | "
            f"{100*m['distance_zone_acc']:.2f}% | {100*m['joint_exact']:.2f}% |"
        )

    if reference is not None:
        r = reference["metrics"]
        lines.extend(
            [
                "",
                "## Pseudo-Label Reference",
                "",
                "This row is not a model result. It shows agreement between the original pseudo-labels and the adjudicated human labels on the same 420-sample subset.",
                "",
                "| Reference | Emotion Acc | Behavior Acc | Distance-Zone Acc | 3-way Exact |",
                "| --- | ---: | ---: | ---: | ---: |",
                f"| Original pseudo labels | {100*r['emotion_acc']:.2f}% | {100*r['behavior_acc']:.2f}% | "
                f"{100*r['distance_zone_acc']:.2f}% | {100*r['joint_exact']:.2f}% |",
            ]
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The gold labels are adjudicated human labels for emotion, behavior, and distance zone.",
            "- Distance-zone bins follow the relabel guideline: intimate < 0.8 m, close < 1.2 m, personal < 1.8 m, far <= 2.5 m.",
            "- This file is intended as a draft artifact for the paper or supplement; the JSON/CSV outputs contain additional Macro-F1 and latency fields.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relabel_csv", default="data/relabel/human_relabel_subset_420_master_paper_final.csv")
    ap.add_argument("--full_ckpt", default="runs/full_sis_er_v3/stage1_best.pt")
    ap.add_argument("--b2_ckpt", default="runs/b2_plain/b2_plain_best.pt")
    ap.add_argument("--nosis_ckpt", default="runs/nosis_clean/stage1_nosis_best.pt")
    ap.add_argument("--nohard_ckpt", default="runs/nohard_clean/stage1_nohard.pt")
    ap.add_argument("--tokenizer", default="distilbert-base-uncased")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include_pseudo_reference", action="store_true")
    ap.add_argument("--include_b3", action="store_true")
    ap.add_argument("--env_file", default=".env")
    ap.add_argument("--llm_model", default="gpt-4o-mini")
    ap.add_argument("--b3_prompt_style", default="zero_shot", choices=["zero_shot", "few_shot_v1"])
    ap.add_argument("--llm_api_base", default="https://api.openai.com/v1")
    ap.add_argument("--llm_temperature", type=float, default=0.0)
    ap.add_argument("--llm_max_tokens", type=int, default=200)
    ap.add_argument("--llm_timeout", type=int, default=60)
    ap.add_argument("--llm_force_json_object", action="store_true")
    ap.add_argument("--llm_cache", default="eval_dumps/b3_human420_cache.json")
    ap.add_argument("--llm_cache_only", action="store_true")
    ap.add_argument(
        "--planner_projection",
        action="store_true",
        help="Apply the paper-style SIS-conditioned planner projection for stage1-family methods.",
    )
    ap.add_argument("--out_json", default="eval_dumps/human_adjudicated_eval_420.json")
    ap.add_argument("--out_csv", default="eval_dumps/human_adjudicated_eval_420.csv")
    ap.add_argument("--out_predictions_csv", default="eval_dumps/human_adjudicated_eval_420_predictions.csv")
    ap.add_argument("--out_md", default="eval_dumps/human_adjudicated_eval_420.md")
    args = ap.parse_args()

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    rows = read_relabel_rows(Path(args.relabel_csv))
    gold = build_gold(rows)
    texts = gold["text"]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    full_model = build_ulapm_from_artifacts(ckpt_path=args.full_ckpt, local_files_only=True)
    full_model.load_state_dict(load_state_dict_robust(args.full_ckpt), strict=True)
    if hasattr(full_model, "deterministic_latent"):
        full_model.deterministic_latent = True
    full_model.to(device)

    b2_model = PlainMultiTask(encoder_name=args.tokenizer, local_files_only=True)
    b2_model.load_state_dict(torch.load(args.b2_ckpt, map_location="cpu"), strict=True)
    b2_model.to(device)

    nosis_model = build_ulapm_from_artifacts(ckpt_path=args.nosis_ckpt, local_files_only=True)
    nosis_model.load_state_dict(load_state_dict_robust(args.nosis_ckpt), strict=True)
    if hasattr(nosis_model, "deterministic_latent"):
        nosis_model.deterministic_latent = True
    nosis_model.to(device)

    nohard_model = build_ulapm_from_artifacts(ckpt_path=args.nohard_ckpt, local_files_only=True)
    nohard_model.load_state_dict(load_state_dict_robust(args.nohard_ckpt), strict=True)
    if hasattr(nohard_model, "deterministic_latent"):
        nohard_model.deterministic_latent = True
    nohard_model.to(device)

    method_specs = [
        ("Emotion->Rule (B1)", lambda: predict_b1(full_model, tokenizer, texts, device, args.batch, args.max_len, args.seed)),
        ("Plain Multi-task (B2)", lambda: predict_b2(b2_model, tokenizer, texts, device, args.batch, args.max_len)),
        (
            "ULAPM-SIS",
            lambda: predict_stage1(
                full_model,
                tokenizer,
                texts,
                device,
                args.batch,
                args.max_len,
                "used",
                args.seed,
                planner_projection=args.planner_projection,
            ),
        ),
        (
            "NoSIS",
            lambda: predict_stage1(
                nosis_model,
                tokenizer,
                texts,
                device,
                args.batch,
                args.max_len,
                "used",
                args.seed,
                planner_projection=args.planner_projection,
            ),
        ),
        (
            "NoHard",
            lambda: predict_stage1(
                nohard_model,
                tokenizer,
                texts,
                device,
                args.batch,
                args.max_len,
                "raw",
                args.seed,
                planner_projection=args.planner_projection,
            ),
        ),
        (
            "NoHard+Posthoc",
            lambda: predict_stage1(
                nohard_model,
                tokenizer,
                texts,
                device,
                args.batch,
                args.max_len,
                "posthoc",
                args.seed,
                planner_projection=args.planner_projection,
            ),
        ),
    ]
    if args.include_b3:
        method_specs.insert(
            2,
            ("Direct LLM Prompting (B3)", lambda: predict_b3(texts, SimpleNamespace(**vars(args)))),
        )

    results = []
    long_rows = []
    for method_name, fn in method_specs:
        print(f"[RUN] {method_name}")
        pred = add_zone_predictions(fn())
        metrics = score_prediction(pred, gold)
        results.append({"method": method_name, "metrics": metrics})
        long_rows.extend(build_long_rows(method_name, pred, gold))

    reference = None
    if args.include_pseudo_reference:
        pseudo_pred = {
            "emotion": np.asarray(gold["original_emotion"], dtype=np.int64),
            "behavior": np.asarray(gold["original_behavior"], dtype=np.int64),
            "distance_zone": np.asarray(gold["original_distance_zone"], dtype=np.int64),
            "distance_m": np.zeros(len(gold["sample_id"]), dtype=np.float32),
            "distance_out_of_range_rate": 0.0,
            "latency_ms": 0.0,
        }
        reference = {
            "method": "Original pseudo labels",
            "metrics": score_prediction(pseudo_pred, gold),
        }

    payload = {
        "n_samples": len(gold["sample_id"]),
        "meta": {
            "relabel_csv": args.relabel_csv,
            "full_ckpt": args.full_ckpt,
            "b2_ckpt": args.b2_ckpt,
            "nosis_ckpt": args.nosis_ckpt,
            "nohard_ckpt": args.nohard_ckpt,
            "tokenizer": args.tokenizer,
            "max_len": args.max_len,
            "batch": args.batch,
            "seed": args.seed,
            "include_pseudo_reference": args.include_pseudo_reference,
            "include_b3": args.include_b3,
            "planner_projection": args.planner_projection,
        },
        "metrics": results,
        "pseudo_reference": reference,
        "label_space": {
            "emotion": EMO7,
            "behavior": BEHAVIORS,
            "distance_zone": DISTANCE_ZONES,
        },
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv = Path(args.out_csv)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "method",
            "emotion_acc",
            "emotion_macro_f1",
            "behavior_acc",
            "behavior_macro_f1",
            "distance_zone_acc",
            "distance_zone_macro_f1",
            "joint_exact",
            "distance_out_of_range_rate",
            "latency_ms",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            row = {"method": item["method"]}
            row.update({k: f"{v:.6f}" for k, v in item["metrics"].items()})
            writer.writerow(row)
        if reference is not None:
            row = {"method": reference["method"]}
            row.update({k: f"{v:.6f}" for k, v in reference["metrics"].items()})
            writer.writerow(row)

    out_predictions = Path(args.out_predictions_csv)
    with out_predictions.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "sample_id",
                "source_index",
                "gold_emotion",
                "pred_emotion",
                "gold_behavior",
                "pred_behavior",
                "gold_distance_zone",
                "pred_distance_zone",
                "pred_distance_m",
            ],
        )
        writer.writeheader()
        writer.writerows(long_rows)

    Path(args.out_md).write_text(make_markdown(results, reference), encoding="utf-8")

    print("\n==================== HUMAN-ADJUDICATED 420 EVAL ====================")
    for item in results:
        m = item["metrics"]
        print(
            f"{item['method']:<24s} "
            f"emo_acc={100*m['emotion_acc']:.2f}% "
            f"beh_acc={100*m['behavior_acc']:.2f}% "
            f"zone_acc={100*m['distance_zone_acc']:.2f}% "
            f"joint={100*m['joint_exact']:.2f}%"
        )
    if reference is not None:
        m = reference["metrics"]
        print(
            f"{reference['method']:<24s} "
            f"emo_acc={100*m['emotion_acc']:.2f}% "
            f"beh_acc={100*m['behavior_acc']:.2f}% "
            f"zone_acc={100*m['distance_zone_acc']:.2f}% "
            f"joint={100*m['joint_exact']:.2f}%"
        )
    print(f"[SAVE] {out_json}")
    print(f"[SAVE] {out_csv}")
    print(f"[SAVE] {out_predictions}")
    print(f"[SAVE] {args.out_md}")


if __name__ == "__main__":
    main()
