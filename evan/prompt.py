from pathlib import Path

from jinja2 import Environment, StrictUndefined, Template

FORMAT_PROMPT = Path(__file__).with_name("format_prompt.jinja")


def load_format_prompt(path: str | None = None) -> Template:
    return Environment(undefined=StrictUndefined).from_string(Path(path or FORMAT_PROMPT).read_text(encoding="utf-8"))
