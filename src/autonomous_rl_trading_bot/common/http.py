from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HttpResult:
    data: Any
    status: int


class HttpRequestError(RuntimeError):
    pass


def get_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout_s: float = 10.0,
    max_retries: int = 6,
    backoff_s: float = 0.5,
) -> HttpResult:
    """
    Basic GET with retry/backoff for transient Binance-style errors.
    Stdlib only.
    """
    hdrs = headers or {}
    last_err: BaseException | None = None

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, method="GET", headers=hdrs)
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read()
                charset = resp.headers.get_content_charset() or "utf-8"
                text = raw.decode(charset, errors="replace")
                return HttpResult(data=json.loads(text), status=int(getattr(resp, "status", 200)))
        except urllib.error.HTTPError as e:
            last_err = e
            code = int(getattr(e, "code", 0) or 0)
            # Retry on common transient codes / rate limits
            if code in (418, 429, 500, 502, 503, 504):
                ra = e.headers.get("Retry-After")
                wait = float(ra) if ra and ra.isdigit() else backoff_s * (2**attempt)
                time.sleep(min(wait, 30.0))
                continue
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise HttpRequestError(f"HTTPError {code} for {url}. Body={body[:500]!r}") from e
        except urllib.error.URLError as e:
            last_err = e
            time.sleep(min(backoff_s * (2**attempt), 10.0))
            continue
        except json.JSONDecodeError as e:
            raise HttpRequestError(f"Invalid JSON from {url}: {e}") from e

    raise HttpRequestError(f"Failed GET after {max_retries} attempts: {url}. Last={last_err!r}")
