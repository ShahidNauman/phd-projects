from __future__ import annotations


DEFAULT_PROMPT_TEMPLATE = "{text}"


def format_judge_text(text: str, template: str | None = None) -> str:
    template = template or DEFAULT_PROMPT_TEMPLATE
    if "{text}" not in template:
        raise ValueError("Prompt template must contain a {text} placeholder")
    return template.format(text=text)
