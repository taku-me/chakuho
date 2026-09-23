"""chakuho(択法)— System One 型の判定エンジン。

生成しない判定: state と、答えの形を宣言した questions を受け取り、
バックエンド(OpenAI 互換 API)へ 1 トークンだけ生成させ、宣言済みラベル上の
確率分布へ集約して返す。推定方式は 2 つ(CHAKUHO_ESTIMATOR で切替、既定 sampling):
- sampling(既定): n 個の 1 トークンサンプル(temperature 1.0)を要求し、ラベルの
  出現頻度を数えて分布にする。logprobs を要求しないため、logprobs 未対応の
  backend(SGLang+投機的デコード等)でも動く
- logprobs: 1 回の生成の top_logprobs(temperature 0)をラベルへ集約する。
  backend が対応していれば使える旧方式
設計は docs/design.md を参照。

    from chakuho import core
    core.evaluate({"screen": "..."}, {"q1": {"type": "choice", ...}})

3 つの判定プリミティブ:
- choice: 複数選択肢から 1 つを選ぶ。probabilities は選択肢ごとの確率。
  52 個までは 1 ラウンド("stages": 1)。53〜2704 個は 52 個ずつのチャンクへ
  分けて並列評価し、各チャンクの上位 k 個(k = 52 // チャンク数)を決勝へ進めて
  1 回で決める 2 段トーナメント("stages": 2)。"__none__"(該当なし)があれば
  全チャンクと決勝に含める。2704 個を超えると ValueError
- noul: yes/no の 2 択。noul は p(yes)
- score: 段階(低→高)のリストから、期待値を 0..1 に正規化して返す

選択肢・段階には A-Z, a-z の順でラベルを振る(1 ラウンドの上限 52)。coverage は
sampling なら宣言ラベルに載ったサンプルの割合、logprobs なら top_logprobs のうち
宣言ラベルへ載った確率質量。0 なら一様分布を返し "degraded": true を付ける
(判定できなかったことを、判定した体で返さない)。
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

DEFAULT_BACKEND_URL = "http://localhost:8006/v1"
BACKEND_TIMEOUT_SEC = 60.0

_UPPER = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
_LOWER = [chr(c) for c in range(ord("a"), ord("z") + 1)]
LABEL_ALPHABET: list[str] = _UPPER + _LOWER
MAX_OPTIONS = len(LABEL_ALPHABET)  # 52、1 ラウンドの上限
MAX_TOURNAMENT_OPTIONS = MAX_OPTIONS * MAX_OPTIONS  # 2704、トーナメント込みの上限

SYSTEM_PROMPT = (
    "You are a decision engine inside a program. You never explain. "
    "You answer with exactly one label from the allowed list."
)

_model_cache: dict[str, str] = {}
_model_cache_lock = threading.Lock()

IMAGES_KEY = "images"  # state(dict)のこのキーに画像 URL(data: / http)のリストを置くと、VLM backend へ画像パートとして渡す
NONE_OPTION = "__none__"  # 呼ぶ側が足す「該当なし」。トーナメントでは全チャンクと決勝に必ず含める
DEFAULT_MAX_INFLIGHT = 16  # backend への同時リクエスト上限(vLLM max-num-seqs 112 を chakuho 単独で埋めない)
DEFAULT_TOP_LOGPROBS = 20  # mlx_lm.server は上限 11。CHAKUHO_TOP_LOGPROBS で下げる

ESTIMATOR_SAMPLING = "sampling"
ESTIMATOR_LOGPROBS = "logprobs"
DEFAULT_ESTIMATOR = ESTIMATOR_SAMPLING  # SGLang+投機的デコードが logprobs を拒否するため(2026-09-23)
DEFAULT_SAMPLES = 16  # bench/sampling_n_sweep.py の実測(n=16/32/64)で選定。根拠は docs/design.md

_inflight: threading.BoundedSemaphore | None = None
_inflight_lock = threading.Lock()


_top_logprobs_warned = False
_estimator_warned = False
_samples_warned = False


def top_logprobs_limit() -> int:
    """backend に要求する top_logprobs 数。env CHAKUHO_TOP_LOGPROBS が不正(非整数・1 未満)なら既定値へ倒し、一度だけ警告する。"""
    global _top_logprobs_warned
    raw = os.environ.get("CHAKUHO_TOP_LOGPROBS")
    if raw is None:
        return DEFAULT_TOP_LOGPROBS
    try:
        value = int(raw)
        if value < 1:
            raise ValueError(raw)
        return value
    except ValueError:
        if not _top_logprobs_warned:
            print(f"chakuho: CHAKUHO_TOP_LOGPROBS={raw!r} は不正。既定値 {DEFAULT_TOP_LOGPROBS} を使う", file=sys.stderr)
            _top_logprobs_warned = True
        return DEFAULT_TOP_LOGPROBS


def estimator_mode() -> str:
    """CHAKUHO_ESTIMATOR(sampling|logprobs)。未設定・不正なら既定値(sampling)へ倒し、不正時のみ一度警告する。"""
    global _estimator_warned
    raw = os.environ.get("CHAKUHO_ESTIMATOR")
    if raw is None:
        return DEFAULT_ESTIMATOR
    value = raw.strip().lower()
    if value in (ESTIMATOR_SAMPLING, ESTIMATOR_LOGPROBS):
        return value
    if not _estimator_warned:
        print(f"chakuho: CHAKUHO_ESTIMATOR={raw!r} は不正。既定値 {DEFAULT_ESTIMATOR!r} を使う", file=sys.stderr)
        _estimator_warned = True
    return DEFAULT_ESTIMATOR


def sample_count() -> int:
    """sampling 推定の n。env CHAKUHO_SAMPLES が不正(非整数・1 未満)なら既定値へ倒し、一度だけ警告する。"""
    global _samples_warned
    raw = os.environ.get("CHAKUHO_SAMPLES")
    if raw is None:
        return DEFAULT_SAMPLES
    try:
        value = int(raw)
        if value < 1:
            raise ValueError(raw)
        return value
    except ValueError:
        if not _samples_warned:
            print(f"chakuho: CHAKUHO_SAMPLES={raw!r} は不正。既定値 {DEFAULT_SAMPLES} を使う", file=sys.stderr)
            _samples_warned = True
        return DEFAULT_SAMPLES


def configure_inflight(limit: int | None = None) -> threading.BoundedSemaphore:
    """backend 同時実行数のセマフォを(再)生成する。limit 省略時は env CHAKUHO_MAX_INFLIGHT。"""
    global _inflight
    if limit is None:
        limit = int(os.environ.get("CHAKUHO_MAX_INFLIGHT", str(DEFAULT_MAX_INFLIGHT)))
    if limit < 1:
        raise ValueError(f"CHAKUHO_MAX_INFLIGHT must be >= 1 (got {limit})")
    with _inflight_lock:
        _inflight = threading.BoundedSemaphore(limit)
        return _inflight


def _get_inflight() -> threading.BoundedSemaphore:
    with _inflight_lock:
        if _inflight is not None:
            return _inflight
    return configure_inflight()


class BackendError(RuntimeError):
    """バックエンドへの到達・応答形式の失敗。呼び出し側は HTTP 503 に写す。"""


def labels_for(n_options: int) -> list[str]:
    """A-Z, a-z の順で n_options 個のラベルを返す(1 ラウンド分、最大 52)。

    Raises:
        ValueError: n_options が 0 以下、または MAX_OPTIONS(52) を超える場合。
    """
    if n_options <= 0:
        raise ValueError("at least one option is required")
    if n_options > MAX_OPTIONS:
        raise ValueError(f"prefilter options to {MAX_OPTIONS} or fewer (got {n_options})")
    return LABEL_ALPHABET[:n_options]


def _render(value: Any) -> str:
    """state / instructions を prompt に埋め込むテキストへ変換する。"""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, dict):
        return "\n".join(f"{k}: {_render(v)}" for k, v in value.items() if k != IMAGES_KEY)
    return json.dumps(value, ensure_ascii=False)


ANSWER_CUE = "Label:"  # 答えの合図。assistant 側の書き出し(prefill)として渡す


def prefill_enabled() -> bool:
    """env CHAKUHO_PREFILL が 0/false/no/off なら user 側に合図を書く旧方式、それ以外は prefill。"""
    return os.environ.get("CHAKUHO_PREFILL", "1").strip().lower() not in ("0", "false", "no", "off")


def _append_cue(content: Any) -> Any:
    if isinstance(content, str):
        return content + "\n" + ANSWER_CUE
    parts = list(content)
    for part in reversed(parts):
        if part.get("type") == "text":
            part["text"] = part["text"] + "\n" + ANSWER_CUE
            break
    return parts


def _images_of(state: Any) -> list[str]:
    """state が dict で images キーを持てば、その URL 一覧(文字列のみ)を返す。"""
    if isinstance(state, dict):
        imgs = state.get(IMAGES_KEY)
        if isinstance(imgs, list):
            return [u for u in imgs if isinstance(u, str) and u]
    return []


def _build_prompt(instructions: str, menu: str, state_text: str, question_line: str, allowed: list[str]) -> str:
    """instructions -> options -> state -> question の順で組む(state は末尾、recency)。

    答えの合図 "Label:" は user 側には書かず、query_backend が assistant 側の書き出し(prefill)として
    渡す。user 側に書くと、ラベルが 52 個並ぶ長い一覧の後でモデルが "Label" を復唱して
    確率質量の 9 割を落とす実測があった(coverage 0.085 → prefill で 0.997、2026-09-20)。
    """
    return (
        f"INSTRUCTIONS:\n{instructions}\n\n"
        f"OPTIONS:\n{menu}\n\n"
        f"STATE:\n{state_text}\n\n"
        f"QUESTION: {question_line}\n"
        f"Allowed labels: {', '.join(allowed)}"
    )


def aggregate(
    top_logprobs: dict[str, float], labels: list[str], *, case_insensitive: bool = False
) -> tuple[dict[str, float], float]:
    """top_logprobs を宣言ラベル上へ集約し、正規化した分布と coverage(集約前の確率質量)を返す。

    case_insensitive=True の時は大小文字を無視して照合する(yes/no 等の単語ラベル向け)。
    False の時は strip のみ行い大小文字はそのまま照合する(A-Z, a-z を両方使う場合、
    大文字小文字を畳むと衝突するため)。coverage が 0 の時は一様分布を返す。
    """
    mass = {label: 0.0 for label in labels}
    lowered_lookup = {label.lower(): label for label in labels} if case_insensitive else {}
    for token, logprob in top_logprobs.items():
        key = token.strip()
        if case_insensitive:
            match = lowered_lookup.get(key.lower())
        else:
            match = key if key in mass else None
        if match is not None:
            mass[match] += math.exp(logprob)
    coverage = sum(mass.values())
    if coverage == 0:
        n = len(labels)
        return {label: 1.0 / n for label in labels}, 0.0
    return {label: value / coverage for label, value in mass.items()}, coverage


def aggregate_counts(
    samples: list[str], labels: list[str], *, case_insensitive: bool = False
) -> tuple[dict[str, float], float]:
    """n 個の 1 トークンサンプル(生成テキスト)を宣言ラベルへ集約し、正規化した分布と
    coverage(宣言ラベルに載ったサンプルの割合)を返す。aggregate() のサンプリング版。

    case_insensitive の意味は aggregate() と同じ。宣言ラベルに一致しないサンプル
    (空文字列・ラベル外のトークン・投機的デコードの副作用で空になったもの等)は
    無視される。1 件も一致しなければ一様分布を返し coverage=0.0("degraded")。
    """
    counts = {label: 0 for label in labels}
    lowered_lookup = {label.lower(): label for label in labels} if case_insensitive else {}
    matched = 0
    for sample in samples:
        key = sample.strip()
        if case_insensitive:
            match = lowered_lookup.get(key.lower())
        else:
            match = key if key in counts else None
        if match is not None:
            counts[match] += 1
            matched += 1
    total = len(samples)
    coverage = matched / total if total else 0.0
    if matched == 0:
        n = len(labels)
        return {label: 1.0 / n for label in labels}, 0.0
    return {label: value / matched for label, value in counts.items()}, coverage


def query_backend(
    prompt: str, backend_url: str, model: str, *, timeout: float = BACKEND_TIMEOUT_SEC,
    images: list[str] | None = None,
) -> tuple[dict[str, float], int]:
    """backend の /chat/completions へ 1 トークンだけ生成させ、(top_logprobs, prompt_tokens) を返す。

    images があれば user メッセージを OpenAI 互換の content parts(image_url × n + text)にする。
    画像は prompt テキストの前に置く(VLM backend の prefix cache に載る)。
    """
    if images:
        content: Any = [{"type": "image_url", "image_url": {"url": u}} for u in images] + [{"type": "text", "text": prompt}]
    else:
        content = prompt
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    body: dict[str, Any] = {"model": model, "messages": messages}
    if prefill_enabled():
        # 答えの合図を assistant 側の書き出しにする(vLLM / 自前 mlx_backend が対応)。
        # 対応しない backend(mlx_lm.server 等)は CHAKUHO_PREFILL=0 で user 側の "Label:" に戻す
        messages.append({"role": "assistant", "content": ANSWER_CUE})
        body["continue_final_message"] = True
        body["add_generation_prompt"] = False
    else:
        messages[1]["content"] = _append_cue(content)
    body |= {
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": top_logprobs_limit(),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = backend_url.rstrip("/") + "/chat/completions"
    req = urllib_request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _get_inflight():
            with urllib_request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
    except (urllib_error.URLError, TimeoutError, OSError) as exc:
        raise BackendError(f"backend request to {url} failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BackendError(f"backend {url} returned invalid JSON: {exc}") from exc
    try:
        top_entries = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        top_logprobs = {entry["token"]: entry["logprob"] for entry in top_entries}
        prompt_tokens = data["usage"]["prompt_tokens"]
    except (KeyError, IndexError, TypeError) as exc:
        raise BackendError(f"unexpected response shape from {url}: {exc}") from exc
    return top_logprobs, prompt_tokens


def _strip_sample_text(content: str | None) -> str:
    """sampling 応答の1件を整形する。prefill(continue_final_message)対応 backend は
    ANSWER_CUE を含まない差分だけを返すのが実測済みの通例だが(2026-09-23、GX10 SGLang)、
    プレフィックスを含めて返す backend への備えとして先頭一致なら剥がす。"""
    text = content or ""
    if prefill_enabled() and text.startswith(ANSWER_CUE):
        text = text[len(ANSWER_CUE):]
    return text.strip()


def query_backend_sampling(
    prompt: str, backend_url: str, model: str, *, n: int, timeout: float = BACKEND_TIMEOUT_SEC,
    images: list[str] | None = None,
) -> tuple[list[str], int]:
    """backend の /chat/completions へ n 個の 1 トークンサンプルを要求し、
    (各サンプルの生成テキスト n 個, prompt_tokens) を返す。

    logprobs は一切要求しない(SGLang+投機的デコードが return_logprob 非対応で
    HTTP 400 を返すため、2026-09-23 実測)。temperature は 1.0 固定
    (艦隊の temperature=0 禁止方針の例外は query_backend の logprobs 経路のみ)。
    """
    if images:
        content: Any = [{"type": "image_url", "image_url": {"url": u}} for u in images] + [{"type": "text", "text": prompt}]
    else:
        content = prompt
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    body: dict[str, Any] = {"model": model, "messages": messages}
    if prefill_enabled():
        messages.append({"role": "assistant", "content": ANSWER_CUE})
        body["continue_final_message"] = True
        body["add_generation_prompt"] = False
    else:
        messages[1]["content"] = _append_cue(content)
    body |= {
        "max_tokens": 1,
        "temperature": 1.0,
        "n": n,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = backend_url.rstrip("/") + "/chat/completions"
    req = urllib_request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _get_inflight():
            with urllib_request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
    except (urllib_error.URLError, TimeoutError, OSError) as exc:
        raise BackendError(f"backend request to {url} failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BackendError(f"backend {url} returned invalid JSON: {exc}") from exc
    try:
        samples = [_strip_sample_text(choice_["message"]["content"]) for choice_ in data["choices"]]
        prompt_tokens = data["usage"]["prompt_tokens"]
    except (KeyError, IndexError, TypeError) as exc:
        raise BackendError(f"unexpected response shape from {url}: {exc}") from exc
    return samples, prompt_tokens


def _query_distribution(
    prompt: str, labels: list[str], backend_url: str, model: str, timeout: float,
    images: list[str] | None, *, case_insensitive: bool,
) -> tuple[dict[str, float], float, int]:
    """backend へ問い合わせ、宣言ラベル上の分布・coverage・prompt_tokens を返す。
    CHAKUHO_ESTIMATOR(既定 sampling)で logprobs/sampling を切り替える一本化窓口。"""
    if estimator_mode() == ESTIMATOR_LOGPROBS:
        top_logprobs, prompt_tokens = query_backend(prompt, backend_url, model, timeout=timeout, images=images)
        dist, coverage = aggregate(top_logprobs, labels, case_insensitive=case_insensitive)
    else:
        samples, prompt_tokens = query_backend_sampling(
            prompt, backend_url, model, n=sample_count(), timeout=timeout, images=images
        )
        dist, coverage = aggregate_counts(samples, labels, case_insensitive=case_insensitive)
    return dist, coverage, prompt_tokens


def resolve_model(backend_url: str, *, timeout: float = BACKEND_TIMEOUT_SEC) -> str:
    """CHAKUHO_MODEL があればそれを使う。無ければ backend の /models 先頭を取得しキャッシュする。"""
    env_model = os.environ.get("CHAKUHO_MODEL")
    if env_model:
        return env_model
    with _model_cache_lock:
        cached = _model_cache.get(backend_url)
    if cached:
        return cached
    return probe_backend(backend_url, timeout=timeout)


def probe_backend(backend_url: str, *, timeout: float = BACKEND_TIMEOUT_SEC) -> str:
    """backend の /models を毎回実際に叩いてモデル ID を返す(キャッシュを使わない。/health の生死判定用)。"""
    url = backend_url.rstrip("/") + "/models"
    req = urllib_request.Request(url, method="GET")
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib_error.URLError, TimeoutError, OSError) as exc:
        raise BackendError(f"failed to resolve model from {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BackendError(f"{url} returned invalid JSON: {exc}") from exc
    try:
        entries = data["data"] if isinstance(data, dict) else data
        model_id = entries[0]["id"]
    except (KeyError, IndexError, TypeError) as exc:
        raise BackendError(f"unexpected response shape from {url}: {exc}") from exc
    with _model_cache_lock:
        _model_cache[backend_url] = model_id
    return model_id


def clear_model_cache() -> None:
    """テスト用: backend_url ごとにキャッシュしたモデル ID を消す。"""
    with _model_cache_lock:
        _model_cache.clear()


def _single_round_choice(
    state_text: str,
    instructions_text: str,
    options: list[str],
    descriptions: dict[str, str] | None,
    backend_url: str,
    model: str,
    timeout: float,
    images: list[str] | None = None,
) -> tuple[dict[str, float], float, int]:
    """52 択以下の 1 ラウンド分の choice 判定。(probabilities, coverage, prompt_tokens) を返す。"""
    labels = labels_for(len(options))
    menu = "\n".join(
        f"{label}: {option}" + (f" — {descriptions[option]}" if descriptions else "")
        for label, option in zip(labels, options, strict=True)
    )
    prompt = _build_prompt(instructions_text, menu, state_text, "choose one option.", labels)
    dist, coverage, prompt_tokens = _query_distribution(
        prompt, labels, backend_url, model, timeout, images, case_insensitive=len(labels) <= 26
    )
    probabilities = {option: dist[label] for label, option in zip(labels, options, strict=True)}
    return probabilities, coverage, prompt_tokens


def choice(
    state: Any,
    instructions: Any,
    criteria: dict[str, str] | list[str],
    *,
    backend_url: str = DEFAULT_BACKEND_URL,
    model: str | None = None,
    timeout: float = BACKEND_TIMEOUT_SEC,
) -> dict[str, Any]:
    """複数選択肢から 1 つを選ぶ。criteria は {option: description} か [option, ...]。

    52 個までは 1 ラウンドで判定する("stages": 1)。53〜2704 個は 52 個ずつの
    チャンクへ分けて並列評価し、各チャンクの argmax を決勝へ進めて 1 回で決める
    2 段トーナメントになる("stages": 2)。決勝で敗れた・予選で落ちた選択肢は
    probabilities で 0.0 になる。2704 個を超えると ValueError。

    Returns:
        {"choice": str, "probabilities": {option: p}, "coverage": p, "stages": 1|2,
         "degraded": True(coverage==0の時のみ), "_input_tokens": n}
        "_input_tokens" は evaluate() が usage.input_tokens へ集約するための内部キー。
    """
    options = list(criteria.keys()) if isinstance(criteria, dict) else list(criteria)
    descriptions = criteria if isinstance(criteria, dict) else None
    if len(options) == 0:
        raise ValueError("at least one option is required")
    if len(options) > MAX_TOURNAMENT_OPTIONS:
        raise ValueError(
            f"prefilter options to {MAX_TOURNAMENT_OPTIONS} or fewer (got {len(options)})"
        )

    resolved_model = model or resolve_model(backend_url, timeout=timeout)
    instructions_text = _render(instructions)
    state_text = _render(state)

    if len(options) <= MAX_OPTIONS:
        probabilities, coverage, prompt_tokens = _single_round_choice(
            state_text, instructions_text, options, descriptions, backend_url, resolved_model, timeout,
            images=_images_of(state),
        )
        best = max(probabilities, key=probabilities.get)
        result: dict[str, Any] = {
            "choice": best,
            "probabilities": probabilities,
            "coverage": coverage,
            "stages": 1,
            "_input_tokens": prompt_tokens,
        }
        if coverage == 0:
            result["degraded"] = True
        return result

    # トーナメント: 52 個以下の束へ分け、各束の上位 k 個を決勝へ進める。
    # k = max(1, 52 // 束数)。argmax だけ通すと強い候補が同じ束で潰し合うため(レビュー指摘)。
    # NONE_OPTION("該当なし")があれば、全束と決勝に必ず含める。
    has_none = NONE_OPTION in options
    contenders = [o for o in options if o != NONE_OPTION]
    per_chunk = MAX_OPTIONS - (1 if has_none else 0)
    chunks = [contenders[i : i + per_chunk] for i in range(0, len(contenders), per_chunk)]
    if has_none:
        chunks = [chunk + [NONE_OPTION] for chunk in chunks]
    top_k = max(1, (MAX_OPTIONS - (1 if has_none else 0)) // len(chunks))

    def _run_chunk(chunk: list[str]) -> tuple[list[str], int]:
        probs, _coverage, tokens = _single_round_choice(
            state_text, instructions_text, chunk, descriptions, backend_url, resolved_model, timeout,
            images=_images_of(state),
        )
        ranked = sorted((o for o in chunk if o != NONE_OPTION), key=lambda o: probs[o], reverse=True)
        return ranked[:top_k], tokens

    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        chunk_results = list(pool.map(_run_chunk, chunks))
    winners: list[str] = []
    for chunk_winners, _tokens in chunk_results:
        winners.extend(chunk_winners)
    if has_none:
        winners.append(NONE_OPTION)
    qualifying_tokens = sum(tokens for _winners, tokens in chunk_results)

    final_probabilities, final_coverage, final_tokens = _single_round_choice(
        state_text, instructions_text, winners, descriptions, backend_url, resolved_model, timeout,
        images=_images_of(state),
    )
    final_best = max(final_probabilities, key=final_probabilities.get)

    probabilities = dict.fromkeys(options, 0.0)
    probabilities.update(final_probabilities)  # 決勝出場者のみ上書き。予選落ちは 0.0 のまま

    result = {
        "choice": final_best,
        "probabilities": probabilities,
        "coverage": final_coverage,
        "stages": 2,
        "_input_tokens": qualifying_tokens + final_tokens,
    }
    if final_coverage == 0:
        result["degraded"] = True
    return result


def noul(
    state: Any,
    instructions: Any,
    criteria: dict[str, str] | None = None,
    *,
    backend_url: str = DEFAULT_BACKEND_URL,
    model: str | None = None,
    timeout: float = BACKEND_TIMEOUT_SEC,
) -> dict[str, Any]:
    """yes/no の 2 択。criteria は省略可の {"true": 説明, "false": 説明}。

    Returns:
        {"noul": p_yes, "coverage": p, "stages": 1,
         "degraded": True(coverage==0の時のみ), "_input_tokens": n}
    """
    described = criteria if isinstance(criteria, dict) else {}
    menu = f"yes: {described.get('true', 'yes')}\nno: {described.get('false', 'no')}"
    prompt = _build_prompt(_render(instructions), menu, _render(state), "yes or no.", ["yes", "no"])
    resolved_model = model or resolve_model(backend_url, timeout=timeout)
    dist, coverage, prompt_tokens = _query_distribution(
        prompt, ["yes", "no"], backend_url, resolved_model, timeout, _images_of(state), case_insensitive=True
    )
    result: dict[str, Any] = {
        "noul": dist["yes"],
        "probability": dist["yes"],  # Jev(Gateway 表記)互換の別名
        "coverage": coverage,
        "stages": 1,
        "_input_tokens": prompt_tokens,
    }
    if coverage == 0:
        result["degraded"] = True
    return result


def score(
    state: Any,
    instructions: Any,
    criteria: list[str],
    *,
    backend_url: str = DEFAULT_BACKEND_URL,
    model: str | None = None,
    timeout: float = BACKEND_TIMEOUT_SEC,
) -> dict[str, Any]:
    """段階(低→高)のリストから、期待値を 0..1 に正規化して返す(最大 52 段階)。

    Returns:
        {"score": 0..1, "probabilities": {level: p}, "coverage": p, "stages": 1,
         "degraded": True(coverage==0の時のみ), "_input_tokens": n}
    """
    levels = list(criteria)
    labels = labels_for(len(levels))
    menu = "\n".join(f"{label}: {level}" for label, level in zip(labels, levels, strict=True))
    prompt = _build_prompt(_render(instructions), menu, _render(state), "rate on this scale.", labels)
    resolved_model = model or resolve_model(backend_url, timeout=timeout)
    dist, coverage, prompt_tokens = _query_distribution(
        prompt, labels, backend_url, resolved_model, timeout, _images_of(state), case_insensitive=len(labels) <= 26
    )
    probabilities = {level: dist[label] for label, level in zip(labels, levels, strict=True)}
    n = len(labels)
    expected = sum(i * dist[label] for i, label in enumerate(labels)) / (n - 1) if n > 1 else 0.0
    result: dict[str, Any] = {
        "score": expected,
        "probabilities": probabilities,
        "coverage": coverage,
        "stages": 1,
        "_input_tokens": prompt_tokens,
    }
    if coverage == 0:
        result["degraded"] = True
    return result


_HANDLERS = {"choice": choice, "noul": noul, "boolean": noul, "score": score}  # "boolean" は Jev(Vercel AI Gateway 表記)の別名


def evaluate(
    state: Any,
    questions: dict[str, dict[str, Any]],
    *,
    backend_url: str = DEFAULT_BACKEND_URL,
    model: str | None = None,
    timeout: float = BACKEND_TIMEOUT_SEC,
) -> dict[str, Any]:
    """同一 state に対する複数質問をスレッドで並列評価し、Jev 互換のレスポンスを返す。

    Returns:
        {"answers": {name: {...}}, "usage": {"input_tokens": n},
         "model": str, "latency_ms": n}

    Raises:
        ValueError: questions が空、または質問の type が不明・選択肢過多などの場合。
        BackendError: backend 呼び出しが失敗した場合。
    """
    if not questions:
        raise ValueError("at least one question is required")
    resolved_model = model or resolve_model(backend_url, timeout=timeout)
    state_text = _render(state)

    def _run(item: tuple[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        name, question = item
        kind = question.get("type")
        handler = _HANDLERS.get(kind)
        if handler is None:
            raise ValueError(f"unknown question type: {kind!r}")
        answer = handler(
            state_text,
            question.get("instructions", ""),
            question.get("criteria"),
            backend_url=backend_url,
            model=resolved_model,
            timeout=timeout,
        )
        return name, answer

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, len(questions))) as pool:
        results = list(pool.map(_run, questions.items()))

    answers: dict[str, Any] = {}
    total_tokens = 0
    for name, answer in results:
        total_tokens += answer.pop("_input_tokens", 0)
        answers[name] = answer

    return {
        "answers": answers,
        "usage": {"input_tokens": total_tokens},
        "model": resolved_model,
        "latency_ms": round((time.perf_counter() - t0) * 1000),
    }
