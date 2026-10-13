"""vision_api: how blindsight talks to the vision model. Shared by imgfix (captions) and vision-mcp (ask_image).

Pure stdlib and no I/O: it builds the HTTP request (url, headers, JSON body) for one image plus a prompt and parses
the reply; each component sends the request with its own HTTP client. Two wire formats (VISION_API):
  openai     OpenAI-compatible chat/completions, image as a data URL. Covers vLLM, SGLang, llama.cpp server, Ollama,
             LM Studio, OpenAI, OpenRouter, Gemini's OpenAI endpoint, Groq, Together, Mistral, Azure OpenAI, ...
  anthropic  Anthropic Messages API (Claude), image as a base64 source block.

Environment (read by from_env):
  VISION_API          openai (default) | anthropic
  VISION_BASE_URL     http(s) URL. openai: a path ending in /chat/completions is used as is (query kept, e.g. Azure's
                      ?api-version=...); a bare host[:port] gets /v1/chat/completions; any other path gets
                      /chat/completions (.../v1, Gemini's .../v1beta/openai, OpenRouter's .../api/v1). anthropic: a
                      path ending in /messages is used as is; one ending in /v1 gets /messages; anything else (bare
                      host, or an Anthropic-style base URL such as https://api.moonshot.ai/anthropic) gets /v1/messages,
                      like ANTHROPIC_BASE_URL. Default: http://host.docker.internal:8001/v1 (openai) or
                      https://api.anthropic.com (anthropic).
  VISION_MODEL        required
  VISION_API_KEY      optional. openai: "Authorization: Bearer <key>"; anthropic: "x-api-key: <key>".
  VISION_HEADERS      optional JSON object of extra request headers (single-line printable-ASCII string values,
                      surrounding whitespace dropped), applied last, so it can replace the defaults (e.g. Azure's
                      {"api-key":"..."} with VISION_API_KEY left empty).
  VISION_TOKEN_PARAM  openai only: max_tokens (default) | max_completion_tokens (newer OpenAI models reject max_tokens).
  VISION_EXTRA_BODY   optional JSON object merged into every request body (top-level keys). A null value deletes
                      that key (e.g. {"temperature":null} for models that reject a temperature). It may not set
                      model, messages, stream or system.
Header values (keys) are secrets: describe() names them but never prints values.
"""
import base64
import json
import os
import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

APIS = ("openai", "anthropic")
DEFAULT_BASE_URL = {"openai": "http://host.docker.internal:8001/v1", "anthropic": "https://api.anthropic.com"}
ENDPOINT = {"openai": "/chat/completions", "anthropic": "/messages"}
ANTHROPIC_VERSION = "2023-06-01"
TOKEN_PARAMS = ("max_tokens", "max_completion_tokens")
RESERVED_BODY_KEYS = ("model", "messages", "stream", "system")  # VISION_EXTRA_BODY may not set these
_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")  # RFC 9110 token
# Header values we send: printable ASCII + space/tab. Anything else (control characters, line breaks, non-ASCII such as
# a pasted zero-width space or smart quote) would only fail later, at request time, inside the HTTP client.
_HEADER_VALUE = re.compile(r"[\t\x20-\x7e]*")


class VisionConfigError(ValueError):
    """An operator-facing configuration problem (reported at startup, before serving)."""


class VisionReplyError(ValueError):
    """The vision model's reply did not have the expected shape."""


@dataclass
class VisionAPI:
    kind: str                                      # "openai" | "anthropic"
    url: str                                       # full endpoint URL
    model: str
    headers: dict = field(default_factory=dict)    # complete request headers incl. auth: never log the values
    extra: dict = field(default_factory=dict)      # VISION_EXTRA_BODY (None values delete keys)
    token_param: str = "max_tokens"

    def describe(self):
        """One log line: where and how, without secrets (header names only, no userinfo in the URL)."""
        u = urlsplit(self.url)
        host = f"[{u.hostname}]" if ":" in u.hostname else u.hostname  # IPv6 literals keep their brackets
        url = urlunsplit((u.scheme, host + (f":{u.port}" if u.port else ""), u.path, "", ""))
        auth = any(k in self.headers for k in ("authorization", "x-api-key", "api-key"))
        names = ",".join(sorted(k for k in self.headers if k not in ("content-type", "anthropic-version"))) or "-"
        return (f"api={self.kind} url={url}{'?...' if u.query else ''} model={self.model} "
                f"auth={'set' if auth else 'unset'} headers={names} extra_body_keys={','.join(sorted(self.extra)) or '-'}")


def endpoint_url(kind, base):
    """VISION_BASE_URL -> the full endpoint URL for this API kind (see the module doc). Raises VisionConfigError
    (whose message hides any user:password@ in the URL)."""
    shown = re.sub(r"//[^/?#]*@", "//***@", base or "")
    try:
        u = urlsplit((base or "").strip())  # raises ValueError on a malformed [IPv6] host
    except ValueError:
        raise VisionConfigError(f"VISION_BASE_URL is not a valid URL: {shown!r}") from None
    if u.scheme not in ("http", "https") or not u.hostname:
        raise VisionConfigError(f"VISION_BASE_URL must be an http(s) URL like http://host:8001/v1; got {shown!r}")
    try:
        u.port  # noqa: B018 - raises ValueError on a bad port
    except ValueError:
        raise VisionConfigError(f"VISION_BASE_URL has an invalid port: {shown!r}") from None
    leaf, path = ENDPOINT[kind], u.path.rstrip("/")
    if not path.endswith(leaf):
        # openai: the base URL includes the version (OpenAI SDK convention); anthropic: it does not (Anthropic SDK /
        # ANTHROPIC_BASE_URL convention: https://api.anthropic.com, https://api.deepseek.com/anthropic, ...).
        if not path or (kind == "anthropic" and not path.endswith("/v1")):
            path += "/v1"
        path += leaf
    return urlunsplit((u.scheme, u.netloc, path, u.query, ""))


def parse_json_object(name, s):
    """An optional JSON-object env value ("" -> {}). Raises VisionConfigError."""
    s = (s or "").strip()
    if not s:
        return {}
    try:
        d = json.loads(s)
    except ValueError as e:
        raise VisionConfigError(f"{name} is not valid JSON ({e}); put it in single quotes in .env, "
                                f"""e.g. {name}='{{"key":"value"}}'""") from None
    if not isinstance(d, dict):
        raise VisionConfigError(f"{name} must be a JSON object, got {type(d).__name__}")
    return d


def parse_extra_body(s):
    d = parse_json_object("VISION_EXTRA_BODY", s)
    bad = sorted(k for k in d if k in RESERVED_BODY_KEYS)
    if bad:
        raise VisionConfigError(f"VISION_EXTRA_BODY may not set {', '.join(bad)} (fixed by blindsight)")
    return d


def parse_headers(s):
    d = parse_json_object("VISION_HEADERS", s)
    out = {}
    for k, v in d.items():
        if not _HEADER_NAME.fullmatch(k):
            raise VisionConfigError(f"VISION_HEADERS: {k!r} is not a valid header name")
        if not isinstance(v, str) or not _HEADER_VALUE.fullmatch(v):  # the message never shows the value (a secret)
            raise VisionConfigError(f"VISION_HEADERS: the value of {k!r} must be a single-line printable-ASCII string")
        out[k.lower()] = v.strip()
    return out


def from_env(env=None):
    """VisionAPI from the environment (see the module doc). Raises VisionConfigError."""
    env = os.environ if env is None else env
    kind = (env.get("VISION_API") or "").strip().lower() or "openai"
    if kind not in APIS:
        raise VisionConfigError(f"VISION_API must be one of {', '.join(APIS)}; got {kind!r}")
    model = (env.get("VISION_MODEL") or "").strip()
    if not model:
        raise VisionConfigError("VISION_MODEL is not set (the model name your vision endpoint serves)")
    url = endpoint_url(kind, (env.get("VISION_BASE_URL") or "").strip() or DEFAULT_BASE_URL[kind])
    token_param = (env.get("VISION_TOKEN_PARAM") or "").strip() or "max_tokens"
    if token_param not in TOKEN_PARAMS:
        raise VisionConfigError(f"VISION_TOKEN_PARAM must be one of {', '.join(TOKEN_PARAMS)}; got {token_param!r}")
    if kind == "anthropic":
        token_param = "max_tokens"  # the Messages API has only this one
    key = (env.get("VISION_API_KEY") or "").strip()
    if not _HEADER_VALUE.fullmatch(key):
        raise VisionConfigError("VISION_API_KEY must be a single-line printable-ASCII string (check for pasted "
                                "invisible or typographic characters)")
    headers = {"content-type": "application/json"}
    if kind == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
        if key:
            headers["x-api-key"] = key
    elif key:
        headers["authorization"] = f"Bearer {key}"
    headers.update(parse_headers(env.get("VISION_HEADERS")))
    return VisionAPI(kind, url, model, headers, parse_extra_body(env.get("VISION_EXTRA_BODY")), token_param)


def build_body(api, png, prompt, *, max_tokens, temperature, system=None):
    """JSON body bytes asking about one PNG image. VISION_EXTRA_BODY is merged last (null deletes a key) but never
    touches the reserved keys."""
    data = base64.b64encode(png).decode("ascii")
    if api.kind == "anthropic":
        body = {"model": api.model,
                "messages": [{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
                    {"type": "text", "text": prompt}]}],
                "temperature": temperature, "max_tokens": max_tokens}
        if system:
            body["system"] = system
    else:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}},
            {"type": "text", "text": prompt}]})
        body = {"model": api.model, "messages": messages, "temperature": temperature, api.token_param: max_tokens}
    for k, v in api.extra.items():
        if k in RESERVED_BODY_KEYS:
            continue
        if v is None:
            body.pop(k, None)
        else:
            body[k] = v
    return json.dumps(body).encode()


def parse_reply(api, payload):
    """Decoded JSON reply -> (text, truncated). text is the raw model text (callers strip any reasoning);
    truncated is True when the reply stopped at the token limit. Raises VisionReplyError on an unexpected shape."""
    try:
        if api.kind == "anthropic":
            blocks = payload["content"]
            if not isinstance(blocks, list):  # iterating a string/object here would silently yield ""
                raise TypeError("content is not a list of blocks")
            text = "".join(b.get("text") or "" for b in blocks if isinstance(b, dict) and b.get("type") == "text")
            return text, payload.get("stop_reason") == "max_tokens"
        choice = payload["choices"][0]
        content = choice["message"]["content"]
        if isinstance(content, list):  # some servers return content parts
            content = "".join(p.get("text") or "" for p in content if isinstance(p, dict))
        if content is not None and not isinstance(content, str):
            raise TypeError("content is not text")
        return content or "", choice.get("finish_reason") == "length"
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        raise VisionReplyError(f"unexpected {api.kind} reply shape ({type(e).__name__})") from e
