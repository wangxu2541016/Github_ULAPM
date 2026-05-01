import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.plain_multitask import PlainMultiTask
from src.models.ulapm import build_ulapm_from_artifacts
from src.paper_planner import planner_project_prediction


BEHAVIORS = ["HUG", "VERBAL_COMFORT", "LISTEN", "CONGRATULATE", "NEUTRAL"]
GLOBAL_D_MIN = 0.3
GLOBAL_D_MAX = 2.5
B3_PROMPT_VERSION = "b3_protocol_v1"


def load_dotenv_if_needed(path, override=True):
    if not path or not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and (override or key not in os.environ):
                os.environ[key] = value


def load_llm_cache(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_llm_cache(path, cache):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def extract_json_object(text):
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"cannot parse json from: {text[:200]}")


def normalize_behavior(text):
    if text is None:
        return None
    raw = str(text).strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "HUG": "HUG",
        "VERBAL_COMFORT": "VERBAL_COMFORT",
        "COMFORT": "VERBAL_COMFORT",
        "LISTEN": "LISTEN",
        "LISTENING": "LISTEN",
        "CONGRATULATE": "CONGRATULATE",
        "CONGRATS": "CONGRATULATE",
        "NEUTRAL": "NEUTRAL",
    }
    return aliases.get(raw)


def get_b3_prompt_version(prompt_style):
    if prompt_style == "few_shot_v1":
        return "b3_protocol_v2_fewshot_structured"
    return B3_PROMPT_VERSION


def build_b3_messages(text, prompt_style="zero_shot"):
    system = (
        "You are a strict HRI planning function. "
        "Given a user utterance, output exactly one JSON object with keys: "
        "emotion, behavior, distance_m, utterance. "
        "behavior must be one of: HUG, VERBAL_COMFORT, LISTEN, CONGRATULATE, NEUTRAL. "
        "distance_m must be a single real number in meters. "
        "utterance should be a short, natural robot reply in one sentence. "
        "Do not include any extra text outside the JSON object."
    )
    user = (
        "Plan a robot social response for the following user utterance.\n\n"
        f"User utterance: {text}\n\n"
        "Return JSON only."
    )
    messages = [{"role": "system", "content": system}]
    if prompt_style == "few_shot_v1":
        examples = [
            (
                "I still can't believe my grandmother is gone. The house feels so empty without her.",
                {
                    "emotion": "sadness",
                    "behavior": "VERBAL_COMFORT",
                    "distance_m": 1.0,
                    "utterance": "I'm sorry you're carrying that loss, and I'm here with you."
                },
            ),
            (
                "Please give me some space. I don't want anyone right next to me at the moment.",
                {
                    "emotion": "anger",
                    "behavior": "LISTEN",
                    "distance_m": 1.8,
                    "utterance": "I understand, and I'll give you space while staying available."
                },
            ),
            (
                "I just got the email saying I was accepted. This is the best news I've had all year.",
                {
                    "emotion": "joy",
                    "behavior": "CONGRATULATE",
                    "distance_m": 1.2,
                    "utterance": "Congratulations, that's wonderful news and you worked hard for it."
                },
            ),
            (
                "Can you just stay with me for a bit? I think I just need some company tonight.",
                {
                    "emotion": "neutral",
                    "behavior": "LISTEN",
                    "distance_m": 1.5,
                    "utterance": "Of course, I can stay with you and listen."
                },
            ),
        ]
        for ex_text, ex_json in examples:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Plan a robot social response for the following user utterance.\n\n"
                        f"User utterance: {ex_text}\n\n"
                        "Return JSON only."
                    ),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(ex_json, ensure_ascii=False),
                }
            )
    messages.append({"role": "user", "content": user})
    return messages


def call_openai_chat(messages, model, api_key, api_base, temperature=0.0, max_tokens=200, timeout=60, force_json_object=False):
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if force_json_object:
        payload["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            pass
        raise RuntimeError(
            f"OpenAI API HTTP {e.code} {e.reason}. Response body: {err_body}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"OpenAI API connection failed: {e}") from e
    data = json.loads(body)
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"unexpected OpenAI response: {data}") from e


def predict_b3_llm(item, args, cache):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Provide it via environment or --env_file.")

    cache_key_src = json.dumps(
        {
            "prompt_version": get_b3_prompt_version(args.b3_prompt_style),
            "model": args.llm_model,
            "prompt_style": args.b3_prompt_style,
            "force_json_object": bool(args.llm_force_json_object),
            "text": item["text"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    cache_key = hashlib.sha1(cache_key_src.encode("utf-8")).hexdigest()

    if cache_key in cache:
        return dict(cache[cache_key])

    if args.llm_cache_only:
        raise RuntimeError(f"cache miss for item {item['id']} while --llm_cache_only is enabled")

    messages = build_b3_messages(item["text"], prompt_style=args.b3_prompt_style)
    t0 = time.perf_counter()
    raw_text = call_openai_chat(
        messages=messages,
        model=args.llm_model,
        api_key=api_key,
        api_base=args.llm_api_base,
        temperature=args.llm_temperature,
        max_tokens=args.llm_max_tokens,
        timeout=args.llm_timeout,
        force_json_object=args.llm_force_json_object,
    )
    latency_ms = 1000.0 * (time.perf_counter() - t0)

    parse_error = None
    emotion = None
    utterance = None
    try:
        obj = extract_json_object(raw_text)
        emotion = obj.get("emotion")
        utterance = obj.get("utterance")
        behavior = normalize_behavior(obj.get("behavior"))
        distance_m = float(obj.get("distance_m"))
        if behavior is None:
            raise ValueError(f"invalid behavior: {obj.get('behavior')}")
    except Exception as e:
        parse_error = str(e)
        behavior = "NEUTRAL"
        distance_m = 1.5

    pred = {
        "behavior": behavior,
        "distance_m": distance_m,
        "latency_ms": latency_ms,
        "llm_model": args.llm_model,
        "llm_emotion": emotion,
        "llm_utterance": utterance,
        "llm_raw_output": raw_text,
        "llm_parse_error": parse_error,
    }
    cache[cache_key] = pred
    save_llm_cache(args.llm_cache, cache)
    return dict(pred)


def load_protocol(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def posthoc_clip_distance(d_val, d_min=GLOBAL_D_MIN, d_max=GLOBAL_D_MAX):
    return float(np.clip(d_val, d_min, d_max))


def predict_full(model, tokenizer, text, device, output_mode="constrained", planner_projection=False):
    enc = tokenizer([text], padding=True, truncation=True, max_length=128, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids, attn)
    latency_ms = 1000.0 * (time.perf_counter() - t0)
    b_logits = out["b_logits"][0].cpu().numpy()
    b_idx = int(np.argmax(b_logits))
    d_raw = float(out["d_raw"][0].cpu()) if "d_raw" in out else float(out["d"][0].cpu())
    d_con = float(out["d"][0].cpu()) if "d" in out else d_raw
    sis_con = out["sis"][0].cpu().numpy() if "sis" in out else None
    if output_mode == "raw":
        d_val = d_raw
    elif output_mode == "posthoc":
        d_val = posthoc_clip_distance(d_raw)
    else:
        d_val = d_con
    behavior = BEHAVIORS[b_idx]
    planner_state = None
    feasible_behaviors = None
    planner_distance_range_m = None
    if planner_projection and sis_con is not None:
        planner = planner_project_prediction(
            behavior_logits=b_logits,
            distance_m=d_val,
            sis_vec=sis_con,
            behaviors=BEHAVIORS,
        )
        behavior = planner["behavior"]
        d_val = planner["distance_m"]
        planner_state = planner["planner_state"]
        feasible_behaviors = planner["feasible_behaviors"]
        planner_distance_range_m = planner["distance_range_m"]
    return {
        "behavior": behavior,
        "distance_m": d_val,
        "distance_raw_m": d_raw,
        "distance_constrained_m": d_con,
        "latency_ms": latency_ms,
        "planner_state": planner_state,
        "planner_feasible_behaviors": feasible_behaviors,
        "planner_distance_range_m": planner_distance_range_m,
    }


def predict_b2(model, tokenizer, text, device):
    enc = tokenizer([text], padding=True, truncation=True, max_length=128, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids, attn)
    latency_ms = 1000.0 * (time.perf_counter() - t0)
    b_idx = int(torch.argmax(out["b_logits"], dim=-1)[0].cpu())
    d_val = float(out["d"][0].cpu())
    return {
        "behavior": BEHAVIORS[b_idx],
        "distance_m": d_val,
        "latency_ms": latency_ms,
    }


def predict_b1(model, tokenizer, text, device):
    enc = tokenizer([text], padding=True, truncation=True, max_length=128, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids, attn)
    latency_ms = 1000.0 * (time.perf_counter() - t0)
    emo_idx = int(torch.argmax(out["c_logits"], dim=-1)[0].cpu())
    emo = ["Joy", "Sadness", "Anger", "Fear", "Surprise", "Disgust", "Neutral"][emo_idx]
    if emo == "Joy":
        behavior, dist = "CONGRATULATE", 1.20
    elif emo == "Sadness":
        behavior, dist = "VERBAL_COMFORT", 1.00
    elif emo == "Fear":
        behavior, dist = "LISTEN", 1.40
    elif emo in {"Anger", "Disgust"}:
        behavior, dist = "LISTEN", 1.60
    elif emo == "Surprise":
        behavior, dist = "NEUTRAL", 1.50
    else:
        behavior, dist = "NEUTRAL", 1.80
    return {
        "behavior": behavior,
        "distance_m": dist,
        "latency_ms": latency_ms,
    }


def score_item(item, pred):
    behavior_ok = int(pred["behavior"] in item["allowed_behaviors"])
    d_min, d_max = item["distance_range_m"]
    distance_violation = int((pred["distance_m"] < d_min) or (pred["distance_m"] > d_max))
    physical_distance_bound_violation = int(
        (pred["distance_m"] < GLOBAL_D_MIN) or (pred["distance_m"] > GLOBAL_D_MAX)
    )
    return {
        "behavior_appropriateness": behavior_ok,
        "distance_violation": distance_violation,
        "physical_distance_bound_violation": physical_distance_bound_violation,
    }


def aggregate(rows):
    by_scene = {}
    for scene in sorted({r["scene_class"] for r in rows}):
        sub = [r for r in rows if r["scene_class"] == scene]
        by_scene[scene] = {
            "n": len(sub),
            "behavior_appropriateness_rate": float(np.mean([r["behavior_appropriateness"] for r in sub])),
            "distance_violation_rate": float(np.mean([r["distance_violation"] for r in sub])),
            "physical_distance_bound_violation_rate": float(np.mean([r["physical_distance_bound_violation"] for r in sub])),
            "avg_latency_ms": float(np.mean([r["latency_ms"] for r in sub])),
        }
    overall = {
        "n": len(rows),
        "behavior_appropriateness_rate": float(np.mean([r["behavior_appropriateness"] for r in rows])),
        "distance_violation_rate": float(np.mean([r["distance_violation"] for r in rows])),
        "physical_distance_bound_violation_rate": float(np.mean([r["physical_distance_bound_violation"] for r in rows])),
        "avg_latency_ms": float(np.mean([r["latency_ms"] for r in rows])),
    }
    return overall, by_scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="data/protocols/robot_protocol_120_paper_final.json")
    ap.add_argument("--method", required=True, choices=["b1", "b2", "b3", "full"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tokenizer", default="distilbert-base-uncased")
    ap.add_argument("--local_files_only", action="store_true",
                    help="Load tokenizer/model only from local Hugging Face cache.")
    ap.add_argument(
        "--full_output_mode",
        default="constrained",
        choices=["constrained", "raw", "posthoc"],
        help="Only used when --method full. constrained=use model d, raw=use d_raw, posthoc=clip d_raw to [0.3,2.5].",
    )
    ap.add_argument(
        "--planner_projection",
        action="store_true",
        help="For method=full, apply the paper-style SIS-conditioned planner projection to behavior and distance.",
    )
    ap.add_argument("--env_file", default=".env")
    ap.add_argument("--llm_model", default="gpt-4o-mini")
    ap.add_argument("--b3_prompt_style", default="zero_shot", choices=["zero_shot", "few_shot_v1"])
    ap.add_argument("--llm_api_base", default="https://api.openai.com/v1")
    ap.add_argument("--llm_temperature", type=float, default=0.0)
    ap.add_argument("--llm_max_tokens", type=int, default=200)
    ap.add_argument("--llm_timeout", type=int, default=60)
    ap.add_argument("--llm_force_json_object", action="store_true")
    ap.add_argument("--llm_cache", default="eval_dumps/b3_llm_cache.json")
    ap.add_argument("--llm_cache_only", action="store_true")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    if args.method in {"b1", "b2", "full"} and not args.ckpt:
        ap.error("--ckpt is required for methods b1, b2, and full")

    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    protocol = load_protocol(args.protocol)
    device = torch.device("cpu")
    tokenizer = None
    model = None
    cache = {}

    if args.method == "full":
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=args.local_files_only)
        model = build_ulapm_from_artifacts(ckpt_path=args.ckpt, local_files_only=args.local_files_only)
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=True)
        if hasattr(model, "deterministic_latent"):
            model.deterministic_latent = True
        predictor = lambda m, tok, text, dev: predict_full(
            m,
            tok,
            text,
            dev,
            output_mode=args.full_output_mode,
            planner_projection=args.planner_projection,
        )
    elif args.method == "b2":
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=args.local_files_only)
        model = PlainMultiTask(encoder_name=args.tokenizer, local_files_only=args.local_files_only)
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=True)
        predictor = predict_b2
    elif args.method == "b3":
        load_dotenv_if_needed(args.env_file)
        cache = load_llm_cache(args.llm_cache)
        predictor = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=args.local_files_only)
        model = build_ulapm_from_artifacts(ckpt_path=args.ckpt, local_files_only=args.local_files_only)
        sd = torch.load(args.ckpt, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        model.load_state_dict(sd, strict=True)
        if hasattr(model, "deterministic_latent"):
            model.deterministic_latent = True
        predictor = predict_b1

    if model is not None:
        model.to(device)
        model.eval()

    rows = []
    for item in protocol["items"]:
        if args.method == "b3":
            pred = predict_b3_llm(item, args, cache)
        else:
            pred = predictor(model, tokenizer, item["text"], device)
        score = score_item(item, pred)
        row = {
            "id": item["id"],
            "scene_class": item["scene_class"],
            "subtype": item["subtype"],
            "text": item["text"],
            "allowed_behaviors": item["allowed_behaviors"],
            "distance_range_m": item["distance_range_m"],
            "pred_behavior": pred["behavior"],
            "pred_distance_m": pred["distance_m"],
            "latency_ms": pred["latency_ms"],
            **score,
        }
        for k in ["llm_model", "llm_emotion", "llm_utterance", "llm_raw_output", "llm_parse_error"]:
            if k in pred:
                row[k] = pred[k]
        for k in ["planner_state", "planner_feasible_behaviors", "planner_distance_range_m"]:
            if k in pred and pred[k] is not None:
                row[k] = pred[k]
        rows.append(row)

    overall, by_scene = aggregate(rows)

    print("\n==================== ROBOT PROTOCOL EVAL ====================")
    print(f"method: {args.method}")
    if args.method == "full":
        print(f"full_output_mode: {args.full_output_mode}")
        print(f"planner_projection: {args.planner_projection}")
    print(f"N: {overall['n']}")
    print(f"behavior appropriateness rate = {100*overall['behavior_appropriateness_rate']:.2f}%")
    print(f"distance violation rate      = {100*overall['distance_violation_rate']:.2f}%")
    print(f"physical d bound violation  = {100*overall['physical_distance_bound_violation_rate']:.2f}%")
    print(f"avg latency                 = {overall['avg_latency_ms']:.3f} ms/sample")
    print("\nPer scene:")
    for scene, stats in by_scene.items():
        print(
            f"  {scene:<24s} "
            f"n={stats['n']:3d} "
            f"beh_ok={100*stats['behavior_appropriateness_rate']:.2f}% "
            f"d_viol={100*stats['distance_violation_rate']:.2f}% "
            f"phys_d_viol={100*stats['physical_distance_bound_violation_rate']:.2f}% "
            f"lat={stats['avg_latency_ms']:.3f}ms"
        )
    print("============================================================")

    if args.out_json is not None:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        payload = {
            "method": args.method,
            "meta": {
                "protocol": args.protocol,
                "ckpt": args.ckpt,
                "tokenizer": args.tokenizer,
                "local_files_only": args.local_files_only,
                "full_output_mode": args.full_output_mode,
                "planner_projection": args.planner_projection,
            },
            "overall": overall,
            "by_scene": by_scene,
            "items": rows,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] wrote {args.out_json}")


if __name__ == "__main__":
    main()
