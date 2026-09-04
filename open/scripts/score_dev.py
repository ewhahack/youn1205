#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dev 세트(200건)로 baseline 추론을 돌리고 24개 항목의 F1 · Macro F1을 계산한다.

무엇을 하는가
  1) data/dev.jsonl.gz(입력) · data/dev_labels.csv(정답)를 로드한다.
     (실제 배포본에서는 두 파일이 open/ 바로 아래에 있다 — DEV_INPUT/DEV_LABELS 참고)
  2) baseline/script.py 의 run() — 노트북과 동일한 프롬프트 구성 · 파싱 · 후처리 로직 — 으로
     dev 200건을 추론해 output/dev_submission.csv 를 만든다.
  3) v1~v24 각 항목의 위반(1) 클래스 Precision/Recall/F1과 전체 Macro F1을 계산해 출력하고,
     data/dev_score_report.md 에 F1 오름차순 표로 저장한다.

실행 환경 안내
  이 저장소에는 torch/vllm이 없으므로 기본값은 --runner mock(모델 없이 전항목 0 예측)이다.
  이 경우 예측이 전부 0이라 모든 항목 F1이 0.0 으로 나오는 게 정상이다 — "모델 성능"이 아니라
  "채점 파이프라인이 정상 동작하는지"를 확인하는 용도다. GPU·vLLM·모델 가중치가 갖춰진 환경에서는

      python scripts/score_dev.py --runner vllm --model-dir /path/to/gemma

  로 실제 모델 점수를 낼 수 있다. RAG(법령 조문 검색) 프롬프트를 쓰려면 --rag를 추가한다
  (rank_bm25 필요: pip install rank_bm25).

  GPU가 없는 로컬 프로토타이핑 단계에서 "그래도 실제 예측값"이 보고 싶으면 --runner api로
  Claude API(Anthropic 공식 SDK, pip install anthropic)를 쓸 수 있다. 이건 순수 로컬 실험용이며
  baseline/script.py(최종 제출 코드)는 전혀 건드리지 않는다 — score_dev.py 안에만 APIRunner로
  존재한다. ANTHROPIC_API_KEY 환경변수(또는 `ant auth login` 프로필)가 필요하다.

      python scripts/score_dev.py --runner api --limit 5      # 우선 5건만 확인
      python scripts/score_dev.py --runner api --api-model claude-haiku-4-5

사용 예
  python scripts/score_dev.py                       # mock, 200건 전체
  python scripts/score_dev.py --limit 20             # mock, 앞 20건만 (빠른 점검)
  python scripts/score_dev.py --runner vllm --rag --model-dir google/gemma-4-26B-A4B-it
  python scripts/score_dev.py --runner api --limit 5 --api-concurrency 4
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import random
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # open/
BASELINE_DIR = ROOT / "baseline"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
DEV_INPUT = ROOT / "dev.jsonl.gz"
DEV_LABELS = ROOT / "dev_labels.csv"
REPORT_PATH = DATA_DIR / "dev_score_report.md"

sys.path.insert(0, str(BASELINE_DIR))
import script as bl  # noqa: E402  (baseline/script.py)

ITEMS = bl.ITEMS


# ===== RAG 프롬프트 (RAG 노트북 R1/R2 셀과 동일 로직, --rag 지정 시에만 사용) =====
ART = re.compile(r"^제\s?\d+조(?:의\s?\d+)?\s*\(", re.M)


def _load_articles(data_dir: str):
    out = []
    for p in sorted(glob.glob(os.path.join(data_dir, "법령패키지", "법령", "*.txt"))):
        law = unicodedata.normalize("NFC", os.path.splitext(os.path.basename(p))[0])
        t = unicodedata.normalize("NFC", io.open(p, encoding="utf-8").read())
        cuts = [m.start() for m in ART.finditer(t)]
        if not cuts:
            out.append((law, "", t))
            continue
        for i, s in enumerate(cuts):
            e = cuts[i + 1] if i + 1 < len(cuts) else len(t)
            body = t[s:e].strip()
            out.append((law, body.split("\n", 1)[0][:60], body))
    return out


def _tokenize(s: str):
    toks = re.findall(r"[가-힣]{2,}|[A-Za-z]{2,}|\d{2,}", s)
    grams = []
    for w in toks:
        grams.append(w)
        if len(w) > 2:
            grams += [w[i:i + 2] for i in range(len(w) - 1)]
    return grams


class LawIndex:
    def __init__(self, data_dir: str):
        from rank_bm25 import BM25Okapi  # 지연 import — --rag 미사용 시 의존성 불필요
        self.arts = _load_articles(data_dir)
        self.bm25 = BM25Okapi([_tokenize(a[2]) for a in self.arts])
        bl.log(f"[RAG] 법령 조문 {len(self.arts)}개 색인")

    def search(self, query: str, topk: int = 8, max_chars: int = 8000) -> str:
        sc = self.bm25.get_scores(_tokenize(query))
        order = sorted(range(len(sc)), key=lambda i: -sc[i])[:topk]
        chunks, used = [], 0
        for i in order:
            law, head, body = self.arts[i]
            block = f"[{law}] {body}"
            if used + len(block) > max_chars:
                block = block[: max(0, max_chars - used)]
            if not block:
                break
            chunks.append(block)
            used += len(block)
        return "\n\n".join(chunks)


def enable_rag(data_dir: str) -> "LawIndex":
    """script.build_user_prompt 를 검색 결과 주입 버전으로 교체한다."""
    idx = LawIndex(data_dir)
    TOPK, RAG_CHARS, QUERY_CHARS = 8, 8000, 3000
    orig_build_user_prompt = bl.build_user_prompt

    def build_user_prompt(rec, max_chars):
        q = "\n".join(d["text"] for d in rec["docs"] if d["type"] == "공고문")[:QUERY_CHARS]
        law = idx.search(q, TOPK, RAG_CHARS)
        return (f"[검색된 법령 조문]\n{law}\n\n" if law else "") + orig_build_user_prompt(rec, max_chars)

    bl.build_user_prompt = build_user_prompt  # build_messages는 모듈 전역을 참조하므로 자동 반영됨
    return idx


# ===== Claude API 러너 (로컬 프로토타이핑 전용 — baseline/script.py는 건드리지 않는다) =====
# MockRunner/VLLMRunner와 동일 인터페이스: __init__(schema, **kw) · .load_seconds ·
# .count_tokens(messages) -> int · .chat(batch: List[List[dict]]) -> List[str]
# (baseline/script.py 의 bl.run()이 정확히 이 3가지만 호출한다 — grep -n "runner\." baseline/script.py 로 확인)

API_PRICE_PER_MTOK = {  # $/1M 토큰, docs.claude.com/pricing 확인 (2026-09-04 기준)
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
}


def _strip_unsupported_schema_keywords(node: Any) -> Any:
    """구조화 출력(JSON Schema) 미지원 키워드(minLength·maxLength·minimum 등) 제거.

    SDK가 알아서 걸러준다고 문서에 나오지만(원문: "Python/TypeScript SDK가 자동으로 처리"),
    raw dict 스키마를 직접 넘기는 경로라 우리가 한 번 더 방어적으로 정리한다.
    """
    UNSUPPORTED = {"minLength", "maxLength", "minimum", "maximum", "multipleOf",
                   "minItems", "maxItems", "pattern", "format"}
    if isinstance(node, dict):
        return {k: _strip_unsupported_schema_keywords(v) for k, v in node.items() if k not in UNSUPPORTED}
    if isinstance(node, list):
        return [_strip_unsupported_schema_keywords(v) for v in node]
    return node


def _flatten_nullable_string_unions(node: Any) -> Any:
    """'type': ['string', 'null'] 같은 유니온을 'type': 'string' 으로 평탄화한다.

    data/정답스키마_디코딩.json은 24개 항목 중 19개(부재탐지 5개 제외)의 근거문구가
    ["string","null"] 유니온이다. Anthropic 구조화 출력은 스키마당 유니온/nullable 타입
    파라미터를 16개까지만 허용하는데(실측: 400 "too many parameters with union types
    (19 parameters ... limit: 16)"), 19개라 그대로는 거부된다. null 대신 빈 문자열("")로
    "근거 없음"을 표현하도록 바꿔서 유니온을 없앤다 — 부재탐지 항목(type:"null" 단일 타입,
    유니온 아님)은 건드리지 않는다. postprocess()가 어차피 근거문구 없으면 ""로 정리하므로
    다운스트림 로직에는 영향이 없다.
    """
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "type" and isinstance(v, list):
                non_null = [t for t in v if t != "null"]
                out[k] = non_null[0] if len(non_null) == 1 else (non_null or "null")
            else:
                out[k] = _flatten_nullable_string_unions(v)
        return out
    if isinstance(node, list):
        return [_flatten_nullable_string_unions(v) for v in node]
    return node


def estimate_run_cost(n_records: int, avg_chars_per_record: int, model: str,
                       max_tokens: int) -> Dict[str, float]:
    """--limit(또는 200건) 기준 대략적인 예상 비용을 계산한다. 정확할 필요 없음 — 자리수 확인용."""
    CHARS_PER_TOKEN = 2.0  # 한국어·영어 혼용 텍스트 대략치(정확한 토크나이저 아님)
    input_tokens_per_rec = avg_chars_per_record / CHARS_PER_TOKEN
    price = API_PRICE_PER_MTOK.get(model, API_PRICE_PER_MTOK["claude-haiku-4-5"])
    input_cost = n_records * input_tokens_per_rec / 1_000_000 * price["input"]
    output_cost = n_records * max_tokens / 1_000_000 * price["output"]  # 상한 기준 최댓값(보수적)
    return {
        "n_records": n_records, "input_tokens_per_rec": round(input_tokens_per_rec),
        "input_cost_usd": input_cost, "output_cost_upper_usd": output_cost,
        "total_upper_usd": input_cost + output_cost,
    }


class APIRunner:
    """Claude API(공식 anthropic SDK) 구조화 출력으로 v1~v24 판정을 받는 로컬 프로토타이핑용 러너.

    baseline/script.py 의 MockRunner/VLLMRunner와 동일한 인터페이스만 구현하면 bl.run()의
    나머지 로직(프롬프트 구성 · fit_to_budget · 파싱 · 후처리 · CSV 저장 · 자가검증)을
    전혀 건드리지 않고 그대로 재사용할 수 있다.
    """
    def __init__(self, schema: Dict[str, Any], model: str = "claude-haiku-4-5",
                 max_tokens: int = 1536, max_retries: int = 6, base_delay: float = 1.0,
                 max_delay: float = 60.0, concurrency: int = 4, request_delay: float = 0.2, **_):
        import anthropic
        t0 = time.time()
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()  # ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN(+ANTHROPIC_BASE_URL) 환경변수에서 자동 로드
        self.schema = _flatten_nullable_string_unions(_strip_unsupported_schema_keywords(schema))
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.concurrency = max(1, concurrency)
        self.request_delay = request_delay
        self.load_seconds = time.time() - t0
        bl.log(f"[API] Claude API 러너 준비 완료 · model={model} · 동시성={self.concurrency} "
               f"· 요청간 딜레이={request_delay}s")

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """실제 API 호출 없이 대략치만 계산(예산 확인용) — MockRunner와 동일한 방식의 근사치."""
        return max(1, sum(len(m["content"]) for m in messages) // 2)

    def _call_once(self, messages: List[Dict[str, str]]) -> str:
        anthropic = self._anthropic
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_msgs = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]

        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.request_delay)  # 요청 사이 짧은 딜레이 — rate limit 예방
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system_msg,
                    messages=user_msgs,
                    output_config={"format": {"type": "json_schema", "schema": self.schema}},
                )
                return next((b.text for b in resp.content if b.type == "text"), "")
            except anthropic.RateLimitError as e:
                last_exc = e
                retry_after = None
                try:
                    retry_after = int(e.response.headers.get("retry-after", "0")) or None
                except Exception:
                    pass
                delay = retry_after or min(self.base_delay * (2 ** attempt) + random.uniform(0, 1), self.max_delay)
                bl.log(f"  [API] rate limit — {delay:.1f}s 후 재시도 ({attempt + 1}/{self.max_retries})")
                time.sleep(delay)
            except anthropic.APIConnectionError as e:
                last_exc = e
                delay = min(self.base_delay * (2 ** attempt) + random.uniform(0, 1), self.max_delay)
                bl.log(f"  [API] 연결 에러 — {delay:.1f}s 후 재시도 ({attempt + 1}/{self.max_retries}): {e}")
                time.sleep(delay)
            except anthropic.APIStatusError as e:
                last_exc = e
                if e.status_code >= 500:
                    delay = min(self.base_delay * (2 ** attempt) + random.uniform(0, 1), self.max_delay)
                    bl.log(f"  [API] 서버 에러 {e.status_code} — {delay:.1f}s 후 재시도 ({attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                else:
                    bl.log(f"  [API] 클라이언트 에러 {e.status_code}: {e.message} — 재시도 안 함")
                    raise
        raise last_exc if last_exc else RuntimeError("APIRunner: 알 수 없는 실패")

    def chat(self, batch: List[List[Dict[str, str]]]) -> List[str]:
        """배치를 --api-concurrency 만큼 동시에 호출한다. 개별 건 실패는 빈 문자열로 반환하고
        (run_chunk가 건 단위로 다시 재시도하므로) 배치 전체를 죽이지 않는다."""
        results: List[str] = [""] * len(batch)
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            futs = {ex.submit(self._call_once, msgs): i for i, msgs in enumerate(batch)}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    bl.log(f"  [API] {i}번째 건 최종 실패 → 빈 출력: {type(e).__name__}: {e}")
                    results[i] = ""
        return results


# ===== 정답 로드 =====
def load_dev_labels(path: Path) -> Dict[str, Dict[str, int]]:
    with io.open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["id"]: {v: int(r[v]) for v in ITEMS} for r in rows}


def load_item_names(data_dir: Path) -> Dict[str, str]:
    tbl = json.load(io.open(data_dir / "항목표.json", encoding="utf-8"))["항목"]
    return {v: tbl[v]["항목명"] for v in ITEMS}


# ===== 지표 =====
def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def score(preds: Dict[str, Dict[str, int]], labels: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, Any]]:
    ids = sorted(set(preds) & set(labels))
    missing = set(labels) - set(preds)
    if missing:
        bl.log(f"[주의] 예측에 없는 id {len(missing)}건 — 정답 있음/예측 없음 (F1 계산에서 제외)")
    out = {}
    for v in ITEMS:
        tp = fp = fn = pos = 0
        for i in ids:
            y, p = labels[i][v], preds[i].get(v, 0)
            pos += y
            if y == 1 and p == 1:
                tp += 1
            elif y == 0 and p == 1:
                fp += 1
            elif y == 1 and p == 0:
                fn += 1
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        out[v] = {"위반건수": pos, "tp": tp, "fp": fp, "fn": fn,
                   "precision": precision, "recall": recall, "f1": f1}
    return out


def write_report(path: Path, per_item: Dict[str, Dict[str, Any]], names: Dict[str, str],
                  macro_f1: float, n_dev: int, runner_desc: str) -> None:
    rows = sorted(per_item.items(), key=lambda kv: kv[1]["f1"])  # F1 낮은 순
    lines = [
        "# Dev 세트 채점 리포트",
        "",
        f"- dev 건수: {n_dev}",
        f"- 추론 러너: {runner_desc}",
        f"- **Macro F1 (24개 항목 평균): {macro_f1:.4f}**",
        "",
        "F1이 낮은 항목부터 정렬했다 — 위에서부터 확인할 것.",
        "",
        "| 순위 | 항목ID | 항목명 | dev 내 위반 건수 | Precision | Recall | F1 |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for rank, (v, m) in enumerate(rows, 1):
        lines.append(
            f"| {rank} | {v} | {names[v]} | {m['위반건수']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="dev 세트로 baseline 추론 후 항목별 F1 · Macro F1 계산")
    ap.add_argument("--runner", choices=["mock", "vllm", "api"], default="mock",
                     help="mock=모델 없이 전항목 0 예측(기본) · vllm=실제 모델 추론(GPU) · "
                          "api=Claude API(Anthropic) 구조화 출력 — 로컬 프로토타이핑 전용")
    ap.add_argument("--rag", action="store_true", help="법령 조문 검색 결과를 프롬프트에 주입(RAG 노트북과 동일 로직)")
    ap.add_argument("--model-dir", default=bl.MODEL_DIR)
    ap.add_argument("--quantization", default=bl.QUANT)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--max-tokens", type=int, default=bl.MAX_TOKENS)
    ap.add_argument("--limit", type=int, default=None, help="앞 N건만(빠른 점검용)")
    ap.add_argument("--out-csv", default=str(OUTPUT_DIR / "dev_submission.csv"))
    ap.add_argument("--report", default=str(REPORT_PATH))
    # --runner api 전용 옵션
    ap.add_argument("--api-model", default="claude-haiku-4-5",
                     help="Claude API 모델 ID (기본 claude-haiku-4-5)")
    ap.add_argument("--api-concurrency", type=int, default=4, help="동시 요청 수 상한")
    ap.add_argument("--api-request-delay", type=float, default=0.2, help="요청 사이 딜레이(초)")
    ap.add_argument("--api-max-retries", type=int, default=6, help="rate limit/서버 에러 재시도 횟수")
    a = ap.parse_args()

    if a.rag:
        enable_rag(str(DATA_DIR))

    quant = None if str(a.quantization).lower() in ("none", "") else a.quantization
    runner_cls = {"mock": bl.MockRunner, "vllm": bl.VLLMRunner, "api": APIRunner}[a.runner]
    if a.runner == "mock":
        runner_kw = {}
    elif a.runner == "vllm":
        runner_kw = dict(model_dir=a.model_dir, quant=quant, max_tokens=a.max_tokens,
                          seed=bl.SEED, gpu_mem=a.gpu_mem, tp=a.tp)
    else:  # api
        runner_kw = dict(model=a.api_model, max_tokens=a.max_tokens, max_retries=a.api_max_retries,
                          concurrency=a.api_concurrency, request_delay=a.api_request_delay)

    # ===== 5) 실행 전 예상 비용 로그 (api 러너일 때만) =====
    if a.runner == "api":
        n_run = a.limit or sum(1 for _ in bl.iter_records(str(DEV_INPUT)))
        sample = list(bl.iter_records(str(DEV_INPUT), limit=min(n_run, 20)))
        avg_chars = sum(len(bl.build_user_prompt(r, a.max_chars)) for r in sample) // len(sample)
        est = estimate_run_cost(n_run, avg_chars, a.api_model, a.max_tokens)
        bl.log(
            "[비용 예상] (정확하지 않음 — 자릿수 확인용) "
            f"{est['n_records']}건 × 입력 약 {est['input_tokens_per_rec']:,}tok/건 "
            f"(model={a.api_model}, ${API_PRICE_PER_MTOK.get(a.api_model, API_PRICE_PER_MTOK['claude-haiku-4-5'])['input']}/1Mtok 입력, "
            f"${API_PRICE_PER_MTOK.get(a.api_model, API_PRICE_PER_MTOK['claude-haiku-4-5'])['output']}/1Mtok 출력) "
            f"≈ 입력 ${est['input_cost_usd']:.4f} + 출력 상한 ${est['output_cost_upper_usd']:.4f} "
            f"= 총 상한 약 ${est['total_upper_usd']:.4f} (실제 출력은 max_tokens보다 훨씬 짧게 끝나는 게 보통이라 이보다 낮게 나올 가능성 높음)"
        )

    report = bl.run(
        input_path=str(DEV_INPUT), out_path=a.out_csv, runner_cls=runner_cls,
        limit=a.limit, chunk=a.chunk, max_chars=a.max_chars, data_dir=str(DATA_DIR), **runner_kw,
    )
    print("[baseline.run 리포트]", json.dumps(report, ensure_ascii=False))

    labels_all = load_dev_labels(DEV_LABELS)
    if a.limit:
        labels_all = {k: v for k, v in labels_all.items() if k in
                      {rec["id"] for rec in bl.iter_records(str(DEV_INPUT), limit=a.limit)}}
    names = load_item_names(DATA_DIR)

    with io.open(a.out_csv, encoding="utf-8", newline="") as f:
        pred_rows = list(csv.DictReader(f))
    preds = {r["id"]: {v: int(r[v]) for v in ITEMS} for r in pred_rows}

    per_item = score(preds, labels_all)
    macro_f1 = sum(m["f1"] for m in per_item.values()) / len(per_item)

    runner_desc = (f"api:{a.api_model}" if a.runner == "api" else a.runner) + ("+rag" if a.rag else "")
    print(f"\n=== 항목별 위반(1) 클래스 F1 (낮은 순) ===")
    for v, m in sorted(per_item.items(), key=lambda kv: kv[1]["f1"]):
        print(f"{v:>4}  {names[v]:<28}  위반 {m['위반건수']:>3}건  "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    print(f"\nMacro F1 (24개 항목 평균) = {macro_f1:.4f}  [러너={runner_desc}]")

    write_report(Path(a.report), per_item, names, macro_f1, len(labels_all), runner_desc)
    print(f"\n리포트 저장: {a.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
