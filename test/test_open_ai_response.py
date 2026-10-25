"""Offline checks: .recai/bin/python framework/test/test_open_ai_response.py"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx

spec = importlib.util.spec_from_file_location(
    "open_ai_under_test", Path(__file__).resolve().parents[1] / "utils/open_ai.py"
)
llm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm)


class ResponseTests(unittest.TestCase):
    def client(self, content, finish="stop", reasoning=""):
        body = {
            "choices": [{"message": {"content": content, "reasoning_content": reasoning},
                         "finish_reason": finish}],
            "usage": {"completion_tokens": 1024},
        }
        client = llm.OpenAICall(model="offline", api_key="test", retry_limits=5)
        self.addCleanup(client.openai_client.close)
        create = Mock(return_value=SimpleNamespace(json=lambda: json.dumps(body)))
        client.openai_client.chat.completions.create = create
        return client, create

    def test_success(self):
        client, create = self.client(" Yes. ")
        self.assertEqual(client.call("test", max_tokens=1024), "Yes.")
        create.assert_called_once()

    def test_invalid_response_fails_without_repeating_request(self):
        for content, finish in [("", "length"), ("partial", "length"), (None, "stop")]:
            with self.subTest(content=content, finish=finish):
                client, create = self.client(content, finish, "private reasoning")
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(llm.LLMCallFailed) as caught:
                        client.call("private prompt", max_tokens=1024)
                message = str(caught.exception)
                self.assertIn(f"finish_reason={finish!r}", message)
                self.assertIn('"completion_tokens": 1024', message)
                self.assertIn("reasoning_chars=17", message)
                self.assertNotIn("private reasoning", message)
                self.assertNotIn("private prompt", message)
                create.assert_called_once()

    def test_network_retry_does_not_swallow_later_invalid_response(self):
        client, create = self.client("", "length")
        response = create.return_value
        create.side_effect = [llm.openai.APIConnectionError(
            request=httpx.Request("POST", "https://example.invalid")
        ), response]
        with patch.object(llm.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(llm.LLMCallFailed, "Invalid LLM response"):
                client.call("test", max_tokens=1024)
        self.assertEqual(create.call_count, 2)


if __name__ == "__main__":
    unittest.main()
