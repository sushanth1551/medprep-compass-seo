"""Groq provider using the official `groq` Python SDK (OpenAI-style chat completions).

The model id always comes from GROQ_MODEL; nothing is hardcoded or substituted.
No retries, no fallback to mock text.
"""

import groq

from app.models import DEFAULT_MAX_OUTPUT_TOKENS, ProviderError
from app.providers.base import Completion, JsonChatProvider
from app.providers.prompts import DEFAULT_PROMPT_STYLE


class GroqProvider(JsonChatProvider):
    provider_id = "groq"
    label = "GroqProvider"

    def __init__(self, api_key, model, timeout=120.0, max_tokens=DEFAULT_MAX_OUTPUT_TOKENS, client=None,
                 prompt_style=DEFAULT_PROMPT_STYLE, target_words=None,
                 max_continuations=1, reasoning_effort=None):
        if not api_key:
            raise ProviderError("GroqProvider needs both an API key and a model name.")
        super().__init__(model, max_tokens, prompt_style, target_words, max_continuations)
        self.reasoning_effort = reasoning_effort  # None = not sent
        self._client = client or groq.Groq(api_key=api_key, max_retries=0, timeout=timeout)

    def _complete(self, system, user):
        try:
            options = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
            response = self._client.chat.completions.create(
                model=self.model,
                max_completion_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **options,
            )
        except groq.AuthenticationError:
            raise ProviderError(
                "Groq rejected the API key (authentication failed). Check GROQ_API_KEY "
                "in your .env file."
            ) from None
        except groq.PermissionDeniedError:
            raise ProviderError("Groq denied permission for this key or model.") from None
        except (groq.NotFoundError, groq.BadRequestError):
            raise ProviderError(
                f"Groq did not accept model '{self.model}' or the request. Check GROQ_MODEL "
                "against the model list in your Groq console."
            ) from None
        except groq.RateLimitError:
            raise ProviderError(
                "Groq rate limit reached. Wait a while and run again; nothing was retried."
            ) from None
        except groq.APITimeoutError:
            raise ProviderError("The Groq request timed out. Nothing was retried.") from None
        except groq.APIConnectionError:
            raise ProviderError("Could not reach the Groq API (network problem).") from None
        except groq.APIStatusError as exc:
            raise ProviderError(f"Groq API error (HTTP status {exc.status_code}).") from None
        choices = getattr(response, "choices", None)
        if not choices:
            raise ProviderError("Groq returned no choices; no draft saved.")
        usage = getattr(response, "usage", None)
        details = getattr(usage, "prompt_tokens_details", None)
        reasoning = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None)
        finish = getattr(choices[0], "finish_reason", None)
        return Completion(
            text=choices[0].message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            cached_tokens=getattr(details, "cached_tokens", None),
            # Only an explicit "length" means the output limit was hit. A missing or unknown
            # finish_reason is not truncation (a malformed reply fails JSON parsing instead).
            truncated=finish == "length",
            reasoning_tokens=reasoning,
            finish_reason=finish,
        )
