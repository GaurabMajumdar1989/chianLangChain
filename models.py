"""Centralized model configuration for this chapter's code companion.

Every script in this folder does `from models import model` instead of naming a
model inline. That is not tidiness for its own sake: it means the chapter's
labs are provider-agnostic, and swapping model or provider is a one-line edit
here rather than a find-and-replace across seven files.

Default: OpenAI gpt-5.6-sol. Requires OPENAI_API_KEY in a local .env file
(see .env.example).

To swap providers, change the init_chat_model call below:
    init_chat_model("anthropic:claude-haiku-4-5")
    init_chat_model("google_genai:gemini-2.5-flash")
    init_chat_model("ollama:llama3.2")
"""

from pathlib import Path

from dotenv import load_dotenv

# Load .env *before* the model import, and let it win over OS env vars, so a
# stale shell export can never shadow the key in your .env file.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

from langchain.chat_models import init_chat_model  # noqa: E402

# `use_responses_api=True` is required here, not stylistic.
#
# gpt-5.6-sol is a reasoning model, and on the older /v1/chat/completions
# endpoint it rejects function tools outright:
#
#     Function tools with reasoning_effort are not supported for gpt-5.6-sol
#     in /v1/chat/completions. To use function tools, use /v1/responses or set
#     reasoning_effort to 'none'.
#
# Every lab in this chapter calls tools, so the choice is between the Responses
# API and switching reasoning off. We take the Responses API: disabling
# reasoning to keep an older endpoint happy would trade away the thing we are
# paying for.
model = init_chat_model("openai:gpt-5.6-sol", use_responses_api=True)

# A second handle for steps that deserve more deliberation. Kept separate so a
# lab can opt into it without changing the default everywhere.
strong_model = init_chat_model("openai:gpt-5.6-sol", use_responses_api=True)


def text_of(message) -> str:
    """Return a message's text, whatever shape the provider used.

    Providers do not agree on this. A chat-completions model puts a plain
    string on `.content`; a Responses-API model puts a list of typed content
    blocks there, only some of which are text. Every lab in this folder prints
    model output, so the normalisation lives here rather than seven times over.
    """
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


if __name__ == "__main__":
    reply = model.invoke("Reply with exactly: OK")
    print(f"text_of: {text_of(reply)!r}")
    print(f"model  : {getattr(model, 'model_name', '?')}")
    print(f"reply  : {reply.content!r}")
    print(f"usage  : {reply.usage_metadata}")
    profile = getattr(model, "profile", None) or {}
    print(f"profile: max_input_tokens={profile.get('max_input_tokens')}")
