import requests
import json

class OllamaChatCompletion:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url

    def create(self, model, messages, stream=False):
        # Convert OpenAI-style messages to Ollama prompt
        prompt = self._format_prompt(messages)

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }

        if stream:
            return self._stream_response(payload)
        else:
            return self._standard_response(payload)

    def _format_prompt(self, messages):
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"[System]: {content}\n"
            elif role == "user":
                formatted += f"[User]: {content}\n"
            elif role == "assistant":
                formatted += f"[Assistant]: {content}\n"
        formatted += "[Assistant]:"
        return formatted

    def _standard_response(self, payload):
        res = requests.post(f"{self.base_url}/api/generate", json=payload)
        res.raise_for_status()
        data = res.json()

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": data["response"]
                    },
                    "finish_reason": "stop"
                }
            ],
            "model": payload["model"],
            "usage": {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            }
        }

    def _stream_response(self, payload):
        res = requests.post(f"{self.base_url}/api/generate", json=payload, stream=True)
        res.raise_for_status()

        def generate():
            for line in res.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        yield {
                            "choices": [{
                                "delta": {"content": chunk["response"]},
                                "finish_reason": None if not chunk.get("done") else "stop"
                            }]
                        }
        return generate()
