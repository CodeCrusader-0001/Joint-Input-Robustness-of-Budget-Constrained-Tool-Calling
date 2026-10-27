"""
Resilient LLM client for Google Gemini API with native function calling.
Features:
- Request timeouts & exponential backoff
- Rate-limit handling (HTTP 429)
- Disk-backed response caching for budget protection and reproducibility
- Hard budget constraints (max API calls, max tokens)
- Dry-run / offline demonstration mode
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from pydantic import BaseModel

from src.config import (
    CACHE_DIR,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    MAX_API_BUDGET,
    MAX_RETRIES,
    MAX_TOKEN_BUDGET,
    RATE_LIMIT_PACING_SECONDS,
    REQUEST_TIMEOUT,
    RETRY_BASE_DELAY,
    is_api_configured,
)
from src.dataset import ToolDefinition
from src.function_calling import build_gemini_tools_payload, parse_gemini_response
from src.token_accounting import extract_token_counts


class ModelResponse(BaseModel):
    """Structured result returned by ModelClient for any tool-calling query."""
    predicted_tool: Optional[str] = None
    predicted_arguments: Dict[str, Any] = {}
    api_success: bool = False
    input_token_count: int = 0
    output_token_count: int = 0
    token_count_source: str = "none"
    latency_seconds: float = 0.0
    error_message: Optional[str] = None
    is_cached: bool = False
    is_dry_run: bool = False
    raw_response: Dict[str, Any] = {}


class ModelClient:
    """
    Client managing Gemini API requests with caching, budgeting, and retry logic.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        dry_run: bool = False,
        enable_cache: bool = True
    ):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name or GEMINI_MODEL
        self.dry_run = dry_run or (not is_api_configured() and not self.api_key)
        self.enable_cache = enable_cache

        # Usage accounting
        self.total_calls_attempted = 0
        self.total_calls_succeeded = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _compute_cache_key(self, query: str, tools: List[ToolDefinition]) -> str:
        """Create deterministic MD5 hash for request caching."""
        tools_repr = [t.model_dump() for t in tools]
        payload_repr = {
            "model": self.model_name,
            "query": query.strip(),
            "tools": tools_repr
        }
        raw_str = json.dumps(payload_repr, sort_keys=True)
        return hashlib.md5(raw_str.encode("utf-8")).hexdigest()

    def _read_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        if not self.enable_cache:
            return None
        cache_file = CACHE_DIR / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _write_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        if not self.enable_cache:
            return
        cache_file = CACHE_DIR / f"{cache_key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def call_tool(
        self,
        query: str,
        tools: List[ToolDefinition],
        temperature: float = 0.0
    ) -> ModelResponse:
        """
        Execute tool-calling query against model or cache.
        """
        tools_payload = build_gemini_tools_payload(tools)
        cache_key = self._compute_cache_key(query, tools)

        # 1. Check cache first
        cached_data = self._read_cache(cache_key)
        if cached_data is not None:
            pred_tool, pred_args, parse_err = parse_gemini_response(cached_data)
            in_tok, out_tok, tok_src = extract_token_counts(
                cached_data, query, tools_payload
            )
            return ModelResponse(
                predicted_tool=pred_tool,
                predicted_arguments=pred_args,
                api_success=parse_err is None and pred_tool is not None,
                input_token_count=in_tok,
                output_token_count=out_tok,
                token_count_source=tok_src,
                latency_seconds=cached_data.get("_cached_latency", 0.0),
                error_message=parse_err,
                is_cached=True,
                is_dry_run=False,
                raw_response=cached_data
            )

        # 2. Check dry-run mode
        if self.dry_run:
            in_tok, out_tok, tok_src = extract_token_counts(None, query, tools_payload)
            return ModelResponse(
                predicted_tool=None,
                predicted_arguments={},
                api_success=False,
                input_token_count=in_tok,
                output_token_count=out_tok,
                token_count_source="heuristic_estimated",
                latency_seconds=0.0,
                error_message="Dry-run mode active; no live API calls made.",
                is_cached=False,
                is_dry_run=True,
                raw_response={}
            )

        # 3. Check budget exhaustion
        if self.total_calls_attempted >= MAX_API_BUDGET:
            return ModelResponse(
                api_success=False,
                error_message=f"API call budget exhausted ({MAX_API_BUDGET} calls reached).",
                token_count_source="none"
            )
        if self.total_input_tokens >= MAX_TOKEN_BUDGET:
            return ModelResponse(
                api_success=False,
                error_message=f"Token budget exhausted ({MAX_TOKEN_BUDGET} tokens reached).",
                token_count_source="none"
            )

        # 4. Execute HTTP API Call with Retries
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={self.api_key}"
        )
        request_body = {
            "contents": [{"parts": [{"text": query}]}],
            "tools": tools_payload,
            "generationConfig": {
                "temperature": temperature
            }
        }

        last_error = ""
        for attempt in range(1, MAX_RETRIES + 1):
            self.total_calls_attempted += 1
            t_start = time.perf_counter()

            try:
                resp = requests.post(
                    url,
                    json=request_body,
                    timeout=REQUEST_TIMEOUT
                )
                latency = round(time.perf_counter() - t_start, 3)

                if resp.status_code == 200:
                    resp_json = resp.json()
                    resp_json["_cached_latency"] = latency
                    self._write_cache(cache_key, resp_json)

                    pred_tool, pred_args, parse_err = parse_gemini_response(resp_json)
                    in_tok, out_tok, tok_src = extract_token_counts(
                        resp_json, query, tools_payload
                    )

                    self.total_calls_succeeded += 1
                    self.total_input_tokens += in_tok
                    self.total_output_tokens += out_tok

                    # Pace subsequent requests to respect free-tier per-minute rate limits
                    if RATE_LIMIT_PACING_SECONDS > 0:
                        time.sleep(RATE_LIMIT_PACING_SECONDS)

                    return ModelResponse(
                        predicted_tool=pred_tool,
                        predicted_arguments=pred_args,
                        api_success=parse_err is None and pred_tool is not None,
                        input_token_count=in_tok,
                        output_token_count=out_tok,
                        token_count_source=tok_src,
                        latency_seconds=latency,
                        error_message=parse_err,
                        is_cached=False,
                        is_dry_run=False,
                        raw_response=resp_json
                    )

                elif resp.status_code == 429:
                    last_error = f"HTTP 429 Rate Limited: {resp.text[:120]}"
                    # On 429, wait at least 15s for the free-tier per-minute quota window to reset
                    backoff = max(15.0, RETRY_BASE_DELAY * (2 ** (attempt - 1)))
                    time.sleep(backoff)
                elif resp.status_code == 503:
                    last_error = f"HTTP 503 Service Unavailable (model overloaded): {resp.text[:120]}"
                    # On 503, wait 10s before retrying
                    backoff = max(10.0, RETRY_BASE_DELAY * (2 ** (attempt - 1)))
                    time.sleep(backoff)
                else:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:150]}"
                    if resp.status_code >= 500:
                        backoff = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                        time.sleep(backoff)
                    else:
                        break

            except requests.RequestException as e:
                last_error = f"Network Exception: {str(e)}"
                backoff = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                time.sleep(backoff)

        # If retries exhausted
        in_tok, out_tok, tok_src = extract_token_counts(None, query, tools_payload)
        return ModelResponse(
            api_success=False,
            input_token_count=in_tok,
            output_token_count=out_tok,
            token_count_source="heuristic_estimated",
            error_message=f"Request failed after {MAX_RETRIES} attempts. Error: {last_error}"
        )
