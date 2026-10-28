"""Claude provider using Anthropic's official Python SDK.

A failed request is NOT retried and NOT replaced by mock text: it raises ProviderError.
"""

import anthropic

from app.models import DEFAULT_MAX_OUTPUT_TOKENS, ProviderError
from app.providers.base import Completion, JsonChatProvider
from app.providers.prompts import DEFAULT_PROMPT_STYLE


class ClaudeProvider(JsonChatProvider):
    provider_id = "claude"
    label = "ClaudeProvider"

    def __init__(self, api_key, model, timeout=120.0, max_tokens=DEFAULT_MAX_OUTPUT_TOKENS, client=None,
                 prompt_style=DEFAULT_PROMPT_STYLE, target_words=None,
                 max_continuations=1):
        if not api_key:
            raise ProviderError("ClaudeProvider needs both an API key and a model name.")
        super().__init__(model, max_tokens, prompt_style, target_words, max_continuations)
        # max_retries=0: we never repeat a paid request automatically.
        self._client = client or anthropic.Anthropic(
            api_key=api_key, max_retries=0, timeout=timeout
        )

    def _complete(self, system, user):
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.AuthenticationError:
            raise ProviderError(
                "Claude rejected the API key (authentication failed). Check "
                "ANTHROPIC_API_KEY in your .env file."
            ) from None
        except anthropic.PermissionDeniedError:
            raise ProviderError("Claude denied permission for this key or model.") from None
        except anthropic.NotFoundError:
            raise ProviderError(
                f"Model '{self.model}' was not found. Check ANTHROPIC_MODEL."
            ) from None
        except anthropic.RateLimitError:
            raise ProviderError(
                "Claude rate limit reached. Wait a while and run again; nothing was retried."
            ) from None
        except anthropic.APITimeoutError:
            raise ProviderError("The Claude request timed out. Nothing was retried.") from None
        except anthropic.APIConnectionError:
            raise ProviderError("Could not reach the Claude API (network problem).") from None
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"Claude API error (HTTP status {exc.status_code}).") from None
        text = "".join(getattr(b, "text", "") for b in getattr(response, "content", []))
        usage = getattr(response, "usage", None)
        return Completion(
            text=text,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            truncated=getattr(response, "stop_reason", None) == "max_tokens",
        )
