"""Unit tests for shared/vision_api.py (pure stdlib, no network): VISION_BASE_URL -> endpoint URL, environment parsing,
request headers and bodies for both wire formats, reply parsing, and that describe() never leaks a secret.

Runs inside the vision-mcp test image (copied next to its tests; see vision-mcp/Dockerfile), or from the repo with
python -m unittest discover -s shared/tests."""

import base64
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

try:
    import vision_api
except ImportError:  # run from the repo: the module is one directory up
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import vision_api

from vision_api import VisionAPI, VisionConfigError, VisionReplyError, build_body, from_env, parse_reply

PNG = b"\x89PNG\r\n\x1a\n-not-really-a-png-"
KEY = "sk-test-0123456789abcdef"


def env(**kv):
    return {"VISION_MODEL": "m", **kv}


def url(kind, base):
    return vision_api.endpoint_url(kind, base)


class TestEndpointUrl(unittest.TestCase):
    def test_openai(self):
        cases = {
            "http://vlm:8001": "http://vlm:8001/v1/chat/completions",  # bare host -> /v1
            "http://vlm:8001/": "http://vlm:8001/v1/chat/completions",
            "http://vlm:8001/v1": "http://vlm:8001/v1/chat/completions",
            "http://vlm:8001/v1/": "http://vlm:8001/v1/chat/completions",
            " http://vlm:8001/v1 ": "http://vlm:8001/v1/chat/completions",  # surrounding whitespace ignored
            "https://api.openai.com/v1": "https://api.openai.com/v1/chat/completions",
            "https://generativelanguage.googleapis.com/v1beta/openai":
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",  # Gemini: no /v1 added
            "https://generativelanguage.googleapis.com/v1beta/openai/":
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            "https://openrouter.ai/api/v1": "https://openrouter.ai/api/v1/chat/completions",
            "https://api.groq.com/openai/v1": "https://api.groq.com/openai/v1/chat/completions",
            "http://vlm:8001/v1/chat/completions": "http://vlm:8001/v1/chat/completions",  # full URL kept
            "http://vlm:8001/v1/chat/completions/": "http://vlm:8001/v1/chat/completions",
            "http://vlm:8001/v1?x=1#frag": "http://vlm:8001/v1/chat/completions?x=1",  # query kept, fragment dropped
            "HTTP://VLM:8001/v1": "http://VLM:8001/v1/chat/completions",  # scheme case-insensitive
            "http://[::1]:8001/v1": "http://[::1]:8001/v1/chat/completions",  # IPv6 literal
            "http://192.168.1.5/v1": "http://192.168.1.5/v1/chat/completions",
            "http://user:pw@vlm:8001/v1": "http://user:pw@vlm:8001/v1/chat/completions",  # userinfo is the client's
        }
        for base, want in cases.items():
            self.assertEqual(url("openai", base), want, base)

    def test_azure_full_url_with_query(self):
        az = "https://res.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21"
        self.assertEqual(url("openai", az), az)
        base = "https://res.openai.azure.com/openai/deployments/gpt-4o?api-version=2024-10-21"
        self.assertEqual(url("openai", base), az)  # path gets the leaf, the query stays at the end

    def test_anthropic(self):
        cases = {
            "https://api.anthropic.com": "https://api.anthropic.com/v1/messages",
            "https://api.anthropic.com/": "https://api.anthropic.com/v1/messages",
            "https://api.anthropic.com/v1": "https://api.anthropic.com/v1/messages",
            "https://api.anthropic.com/v1/": "https://api.anthropic.com/v1/messages",
            "https://api.anthropic.com/v1/messages": "https://api.anthropic.com/v1/messages",
            # Anthropic-style base URLs (what ANTHROPIC_BASE_URL is set to) never include /v1
            "https://api.moonshot.ai/anthropic": "https://api.moonshot.ai/anthropic/v1/messages",
            "https://api.deepseek.com/anthropic/": "https://api.deepseek.com/anthropic/v1/messages",
            "https://api.z.ai/api/anthropic": "https://api.z.ai/api/anthropic/v1/messages",
            "https://gw.example/anthropic/v1/messages?beta=true": "https://gw.example/anthropic/v1/messages?beta=true",
            "http://127.0.0.1:4000": "http://127.0.0.1:4000/v1/messages",
        }
        for base, want in cases.items():
            self.assertEqual(url("anthropic", base), want, base)

    def test_bad_urls(self):
        for base in ("", "   ", None, "vlm:8001", "ftp://vlm/v1", "file:///etc/passwd", "http://", "http:///v1",
                     "//vlm:8001/v1", "http://:8001/v1", "vlm", "unix:/run/x.sock"):
            with self.assertRaisesRegex(VisionConfigError, "must be an http\\(s\\) URL", msg=repr(base)):
                url("openai", base)
        for base in ("http://vlm:99999/v1", "http://vlm:port/v1", "http://vlm:-1/v1", "http://vlm:8001:2/v1"):
            with self.assertRaisesRegex(VisionConfigError, "invalid port", msg=base):
                url("openai", base)
        for base in ("http://[::1/v1", "http://[not-ip]:80/v1"):  # urlsplit itself raises ValueError on these
            with self.assertRaisesRegex(VisionConfigError, "not a valid URL", msg=base):
                url("anthropic", base)

    def test_errors_hide_userinfo(self):
        for base in ("ftp://user:hunter2@vlm/v1", "http://user:hunter2@vlm:99999/v1", "http://user:hunter2@[::1/v1"):
            with self.assertRaises(VisionConfigError) as cm:
                url("openai", base)
            self.assertNotIn("hunter2", str(cm.exception))
            self.assertIn("***@", str(cm.exception))

    def test_config_error_is_value_error(self):
        self.assertTrue(issubclass(VisionConfigError, ValueError))
        self.assertTrue(issubclass(VisionReplyError, ValueError))


class TestFromEnv(unittest.TestCase):
    def test_defaults_openai(self):
        api = from_env(env())
        self.assertEqual((api.kind, api.url, api.model, api.token_param, api.extra),
                         ("openai", "http://host.docker.internal:8001/v1/chat/completions", "m", "max_tokens", {}))
        self.assertEqual(api.headers, {"content-type": "application/json"})  # no key -> no auth header

    def test_defaults_anthropic(self):
        api = from_env(env(VISION_API="anthropic"))
        self.assertEqual((api.kind, api.url), ("anthropic", "https://api.anthropic.com/v1/messages"))
        self.assertEqual(api.headers, {"content-type": "application/json", "anthropic-version": "2023-06-01"})

    def test_empty_values_mean_default(self):
        api = from_env(env(VISION_API=" ", VISION_BASE_URL="", VISION_TOKEN_PARAM="", VISION_API_KEY=" ",
                           VISION_HEADERS=" ", VISION_EXTRA_BODY=""))
        self.assertEqual(api, from_env(env()))

    def test_kind(self):
        self.assertEqual(from_env(env(VISION_API=" Anthropic ")).kind, "anthropic")
        self.assertEqual(from_env(env(VISION_API="OPENAI")).kind, "openai")
        for bad in ("gemini", "claude", "open-ai"):
            with self.assertRaisesRegex(VisionConfigError, "VISION_API must be one of openai, anthropic", msg=bad):
                from_env(env(VISION_API=bad))

    def test_model_required(self):
        for e in ({}, {"VISION_MODEL": ""}, {"VISION_MODEL": "  "}):
            with self.assertRaisesRegex(VisionConfigError, "VISION_MODEL is not set", msg=e):
                from_env(e)
        self.assertEqual(from_env(env(VISION_MODEL=" qwen3-vl ")).model, "qwen3-vl")

    def test_reads_os_environ_by_default(self):
        clean = {k: v for k, v in os.environ.items() if not k.startswith("VISION_")}
        with mock.patch.dict(os.environ, {**clean, "VISION_MODEL": "from-os", "VISION_API": "anthropic"}, clear=True):
            api = from_env()
        self.assertEqual((api.kind, api.model), ("anthropic", "from-os"))

    def test_bad_base_url_reported(self):
        with self.assertRaisesRegex(VisionConfigError, "VISION_BASE_URL"):
            from_env(env(VISION_BASE_URL="vlm:8001"))

    def test_auth_header_per_kind(self):
        api = from_env(env(VISION_API_KEY=f" {KEY} "))
        self.assertEqual(api.headers, {"content-type": "application/json", "authorization": f"Bearer {KEY}"})
        api = from_env(env(VISION_API="anthropic", VISION_API_KEY=KEY))
        self.assertEqual(api.headers, {"content-type": "application/json", "anthropic-version": "2023-06-01",
                                       "x-api-key": KEY})
        self.assertNotIn("authorization", api.headers)

    def test_api_key_must_be_printable_ascii(self):
        for bad in ("sk-​abc", "sk-’abc", "sk-a\nb", "sk-a\x00b", "sk-\x7f"):
            with self.assertRaisesRegex(VisionConfigError, "VISION_API_KEY must be a single-line", msg=repr(bad)):
                from_env(env(VISION_API_KEY=bad))
            try:
                from_env(env(VISION_API_KEY=bad))
            except VisionConfigError as e:
                self.assertNotIn(bad.strip(), str(e))  # the key itself is never in the message

    def test_headers_override_and_extend(self):
        api = from_env(env(VISION_API_KEY=KEY, VISION_HEADERS='{"Authorization": "Bearer other", "X-Title": " bs "}'))
        self.assertEqual(api.headers["authorization"], "Bearer other")  # applied last: replaces the default
        self.assertEqual(api.headers["x-title"], "bs")  # names lower-cased, values stripped
        azure = from_env(env(VISION_HEADERS='{"api-key": "az-secret"}'))  # Azure: no VISION_API_KEY
        self.assertEqual(azure.headers, {"content-type": "application/json", "api-key": "az-secret"})
        anth = from_env(env(VISION_API="anthropic", VISION_API_KEY=KEY,
                            VISION_HEADERS='{"anthropic-version": "2024-01-01", "anthropic-beta": "x"}'))
        self.assertEqual((anth.headers["anthropic-version"], anth.headers["anthropic-beta"], anth.headers["x-api-key"]),
                         ("2024-01-01", "x", KEY))

    def test_headers_validation(self):
        cases = {
            "{bad": "VISION_HEADERS is not valid JSON",
            '["a"]': "VISION_HEADERS must be a JSON object, got list",
            '"x"': "VISION_HEADERS must be a JSON object, got str",
            '{"bad name": "v"}': "is not a valid header name",
            '{"x:y": "v"}': "is not a valid header name",
            '{"": "v"}': "is not a valid header name",
            '{"x-n": 1}': "must be a single-line printable-ASCII string",
            '{"x-n": null}': "must be a single-line printable-ASCII string",
            '{"x-n": ["v"]}': "must be a single-line printable-ASCII string",
            '{"x-n": "a\\r\\nInjected: 1"}': "must be a single-line printable-ASCII string",
            '{"x-n": "a\\u0000b"}': "must be a single-line printable-ASCII string",
            '{"x-n": "caf\\u00e9"}': "must be a single-line printable-ASCII string",
        }
        for value, msg in cases.items():
            with self.assertRaisesRegex(VisionConfigError, msg, msg=value):
                from_env(env(VISION_HEADERS=value))
        with self.assertRaises(VisionConfigError) as cm:
            from_env(env(VISION_HEADERS='{"api-key": "secret-value\\n"}'))
        self.assertNotIn("secret-value", str(cm.exception))  # header values are secrets
        self.assertEqual(from_env(env(VISION_HEADERS='{"x-tab": "a\\tb", "x-empty": ""}')).headers["x-tab"], "a\tb")

    def test_json_error_example_is_formatted(self):
        with self.assertRaises(VisionConfigError) as cm:
            from_env(env(VISION_EXTRA_BODY="{chat_template_kwargs: 1}"))
        msg = str(cm.exception)
        self.assertIn("VISION_EXTRA_BODY is not valid JSON", msg)
        self.assertIn("""e.g. VISION_EXTRA_BODY='{"key":"value"}'""", msg)  # not a literal "{name}" placeholder
        self.assertNotIn("{name}", msg)

    def test_token_param(self):
        self.assertEqual(from_env(env(VISION_TOKEN_PARAM="max_completion_tokens")).token_param, "max_completion_tokens")
        self.assertEqual(from_env(env(VISION_TOKEN_PARAM=" max_tokens ")).token_param, "max_tokens")
        anth = from_env(env(VISION_API="anthropic", VISION_TOKEN_PARAM="max_completion_tokens"))
        self.assertEqual(anth.token_param, "max_tokens")  # the Messages API has only max_tokens
        for bad in ("max_output_tokens", "MAX_TOKENS", "tokens"):
            with self.assertRaisesRegex(VisionConfigError, "VISION_TOKEN_PARAM must be one of", msg=bad):
                from_env(env(VISION_TOKEN_PARAM=bad))

    def test_extra_body(self):
        api = from_env(env(VISION_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false},"temperature":null}'))
        self.assertEqual(api.extra, {"chat_template_kwargs": {"enable_thinking": False}, "temperature": None})
        for value, msg in (("{bad", "VISION_EXTRA_BODY is not valid JSON"), ("[]", "must be a JSON object, got list"),
                           ("1", "must be a JSON object, got int"), ("null", "must be a JSON object, got NoneType"),
                           ('{"model":"x"}', "may not set model"), ('{"messages":[]}', "may not set messages"),
                           ('{"stream":true}', "may not set stream"), ('{"system":"x"}', "may not set system"),
                           ('{"stream":true,"model":"x","top_k":1}', "may not set model, stream \\(fixed by")):
            with self.assertRaisesRegex(VisionConfigError, msg, msg=value):
                from_env(env(VISION_EXTRA_BODY=value))


class TestBuildBody(unittest.TestCase):
    def body(self, api, **kw):
        kw.setdefault("max_tokens", 700)
        kw.setdefault("temperature", 0.2)
        raw = build_body(api, PNG, "What is shown?", **kw)
        self.assertIsInstance(raw, bytes)
        return json.loads(raw)

    def test_openai(self):
        api = from_env(env(VISION_MODEL="vlm"))
        b = self.body(api, system="SYS")
        self.assertEqual(list(b), ["model", "messages", "temperature", "max_tokens"])
        self.assertEqual((b["model"], b["temperature"], b["max_tokens"]), ("vlm", 0.2, 700))
        self.assertEqual(b["messages"], [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(PNG).decode()}},
                {"type": "text", "text": "What is shown?"}]}])
        self.assertEqual([m["role"] for m in self.body(api)["messages"]], ["user"])  # no system -> no system message
        self.assertEqual([m["role"] for m in self.body(api, system="")["messages"]], ["user"])

    def test_openai_token_param(self):
        b = self.body(from_env(env(VISION_TOKEN_PARAM="max_completion_tokens")), max_tokens=60)
        self.assertEqual(b["max_completion_tokens"], 60)
        self.assertNotIn("max_tokens", b)

    def test_anthropic(self):
        api = from_env(env(VISION_API="anthropic", VISION_MODEL="claude-x"))
        b = self.body(api, system="SYS", max_tokens=600, temperature=0.0)
        self.assertEqual((b["model"], b["system"], b["max_tokens"], b["temperature"]), ("claude-x", "SYS", 600, 0.0))
        self.assertEqual(b["messages"], [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.b64encode(PNG).decode()}},
            {"type": "text", "text": "What is shown?"}]}])
        self.assertNotIn("system", self.body(api))  # no system -> no system key
        self.assertNotIn("max_completion_tokens", b)

    def test_extra_merge_and_null_delete(self):
        for kind in ("openai", "anthropic"):
            api = from_env(env(VISION_API=kind, VISION_EXTRA_BODY='{"temperature":null,"top_k":20,"max_tokens":900,'
                                                                  '"chat_template_kwargs":{"enable_thinking":false},'
                                                                  '"not_there":null}'))
            b = self.body(api, system="SYS")
            self.assertNotIn("temperature", b, kind)  # null deletes
            self.assertNotIn("not_there", b, kind)  # deleting a missing key is fine
            self.assertEqual((b["top_k"], b["max_tokens"], b["chat_template_kwargs"]),
                             (20, 900, {"enable_thinking": False}), kind)  # extra wins over our defaults

    def test_reserved_keys_never_overridden(self):
        # from_env refuses them; a hand-built VisionAPI still cannot replace or delete them
        for kind in ("openai", "anthropic"):
            api = VisionAPI(kind, "http://x/v1/x", "real", {}, {"model": "hijack", "messages": None, "stream": True,
                                                                  "system": "hijack"})
            b = self.body(api, system="SYS")
            self.assertEqual(b["model"], "real", kind)
            self.assertIn("messages", b, kind)
            self.assertNotIn("stream", b, kind)
            if kind == "anthropic":
                self.assertEqual(b["system"], "SYS")
            else:
                self.assertNotIn("system", b)
                self.assertEqual(b["messages"][0], {"role": "system", "content": "SYS"})

    def test_deterministic_and_does_not_mutate(self):
        api = from_env(env(VISION_EXTRA_BODY='{"temperature":null}'))
        a = build_body(api, PNG, "q", max_tokens=1, temperature=0.5, system="s")
        self.assertEqual(a, build_body(api, PNG, "q", max_tokens=1, temperature=0.5, system="s"))
        self.assertEqual(api.extra, {"temperature": None})
        a.decode("ascii")  # non-ASCII prompt text is escaped: the body is plain ASCII JSON
        self.assertIn("\\u00e9", build_body(api, PNG, "café", max_tokens=1, temperature=0).decode("ascii"))


class TestParseReply(unittest.TestCase):
    OA = VisionAPI("openai", "http://x/v1/chat/completions", "m")
    AN = VisionAPI("anthropic", "https://x/v1/messages", "m")

    def test_openai(self):
        def reply(content, finish="stop"):
            return {"choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                 "finish_reason": finish}]}
        self.assertEqual(parse_reply(self.OA, reply("A red square.")), ("A red square.", False))
        self.assertEqual(parse_reply(self.OA, reply("A red", "length")), ("A red", True))
        self.assertEqual(parse_reply(self.OA, reply(None)), ("", False))  # e.g. refusal-only / empty
        self.assertEqual(parse_reply(self.OA, reply("<think>x</think>Red.")), ("<think>x</think>Red.", False))  # raw
        parts = [{"type": "text", "text": "A "}, {"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": None},
                 "junk", {"type": "text", "text": "red square."}]
        self.assertEqual(parse_reply(self.OA, reply(parts)), ("A red square.", False))  # content parts joined
        no_finish = {"choices": [{"message": {"content": "x"}}]}
        self.assertEqual(parse_reply(self.OA, no_finish), ("x", False))

    def test_openai_bad_shapes(self):
        for payload in ({}, {"choices": []}, {"choices": None}, {"choices": [{}]}, {"choices": [{"message": {}}]},
                        {"choices": [{"message": None}]}, {"choices": ["x"]}, {"choices": [{"message": {"content": 5}}]},
                        {"choices": [{"message": {"content": {"text": "x"}}}]},
                        {"choices": [{"message": {"content": [{"text": 5}]}}]}, [], "x", None,
                        {"content": [{"type": "text", "text": "anthropic shape"}]}):
            with self.assertRaisesRegex(VisionReplyError, "unexpected openai reply shape", msg=repr(payload)):
                parse_reply(self.OA, payload)

    def test_anthropic(self):
        def reply(content, stop="end_turn"):
            return {"type": "message", "role": "assistant", "content": content, "stop_reason": stop}
        self.assertEqual(parse_reply(self.AN, reply([{"type": "text", "text": "Red."}])), ("Red.", False))
        self.assertEqual(parse_reply(self.AN, reply([{"type": "text", "text": "Re"}], "max_tokens")), ("Re", True))
        blocks = [{"type": "thinking", "thinking": "hmm", "signature": "s"}, {"type": "text", "text": "A "},
                  {"type": "redacted_thinking", "data": "x"}, "junk", {"type": "text", "text": "red square."},
                  {"type": "text"}]
        self.assertEqual(parse_reply(self.AN, reply(blocks)), ("A red square.", False))  # only text blocks
        self.assertEqual(parse_reply(self.AN, reply([])), ("", False))
        self.assertEqual(parse_reply(self.AN, {"content": [{"type": "text", "text": "x"}]}), ("x", False))

    def test_anthropic_bad_shapes(self):
        for payload in ({}, {"content": None}, {"content": "Red."}, {"content": {"type": "text", "text": "x"}},
                        {"content": [{"type": "text", "text": 5}]}, [], "x", None,
                        {"choices": [{"message": {"content": "openai shape"}}]}):
            with self.assertRaisesRegex(VisionReplyError, "unexpected anthropic reply shape", msg=repr(payload)):
                parse_reply(self.AN, payload)


class TestDescribe(unittest.TestCase):
    def test_no_secrets(self):
        api = from_env(env(VISION_BASE_URL="https://user:hunter2@vlm.example:8443/v1?key=qsecret", VISION_API_KEY=KEY,
                           VISION_HEADERS='{"X-Org": "org-secret-value", "api-key": "az-secret-value"}',
                           VISION_EXTRA_BODY='{"top_k": 20, "api_token": "body-secret-value"}'))
        line = api.describe()
        for secret in (KEY, "hunter2", "user", "qsecret", "org-secret-value", "az-secret-value", "body-secret-value",
                       "Bearer"):
            self.assertNotIn(secret, line)
        self.assertEqual(line, "api=openai url=https://vlm.example:8443/v1/chat/completions?... model=m auth=set "
                               "headers=api-key,authorization,x-org extra_body_keys=api_token,top_k")

    def test_plain(self):
        self.assertEqual(from_env(env()).describe(),
                         "api=openai url=http://host.docker.internal:8001/v1/chat/completions model=m auth=unset "
                         "headers=- extra_body_keys=-")
        self.assertEqual(from_env(env(VISION_API="anthropic", VISION_API_KEY=KEY)).describe(),
                         "api=anthropic url=https://api.anthropic.com/v1/messages model=m auth=set headers=x-api-key "
                         "extra_body_keys=-")

    def test_ipv6_and_azure(self):
        self.assertIn("url=http://[::1]:8001/v1/chat/completions ",
                      from_env(env(VISION_BASE_URL="http://[::1]:8001/v1")).describe())
        az = from_env(env(VISION_BASE_URL="https://r.openai.azure.com/openai/deployments/d/chat/completions"
                                          "?api-version=2024-10-21", VISION_HEADERS='{"api-key": "az-key-123"}'))
        line = az.describe()
        self.assertIn("url=https://r.openai.azure.com/openai/deployments/d/chat/completions?... ", line)
        self.assertIn("auth=set headers=api-key ", line)
        self.assertNotIn("az-key-123", line)


if __name__ == "__main__":
    unittest.main()
