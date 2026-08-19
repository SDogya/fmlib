import argparse
import json
import os
import re
import sys
import warnings

TRIGGER_LIST = list(map(re.compile, [".*epk.*", ".*епк.*"]))
IGNORE_LIST = [".*unclean.*", ".*ipynb_checkpoints.*"]


def warning_triggers(text: str, triggers: list = TRIGGER_LIST) -> bool:
    any_triggered = False
    lower_text = text.lower()
    for trigger in triggers:
        if match := trigger.match(lower_text):
            msg: str = f"Caution: {match.group()=} triggered warning in {text=}."
            warnings.warn(msg, stacklevel=2)
            any_triggered = True
    return any_triggered


def clean_up_text(text: str, mask_char: str = "🔐", threshold: int = 5) -> str:
    warning_triggers(text)

    def replace_match(match):
        msg: str = f"Caution: {match.group()=} triggered substitution."
        warnings.warn(msg, stacklevel=2)
        return mask_char * len(match.group())

    return re.sub(rf"\d{{{threshold},}}", replace_match, text)


def clean_up_output(output: dict) -> dict:
    result = {}

    if data := output.get("data", {}):
        for data_key, data_value in data.items():
            if data_key == "image/png":
                result[data_key] = data_value
            elif data_key == "text/plain":
                result[data_key] = [clean_up_text(text) for text in data_value]

    return {
        "data": result,
        "output_type": "display_data",
        "metadata": {},
    }


def clean_up_notebook(path: str, unclean_postfix: str = ".unclean", strict: bool = False) -> str:
    assert os.path.exists(path)
    assert path.endswith(".ipynb")

    with open(path, "r", encoding="utf-8") as file:
        notebook_json = json.load(file)

    unclean_path: str = path.removesuffix(".ipynb") + unclean_postfix + ".ipynb"
    if os.path.exists(unclean_path):
        msg: str = f"File {unclean_path=} already exists."
        if strict:
            raise IOError(msg)
        else:
            warnings.warn(msg, stacklevel=2)
            os.remove(unclean_path)

    os.rename(path, unclean_path)

    for cell in notebook_json["cells"]:
        if "outputs" in cell:
            cleaned_outputs = []
            for output in cell["outputs"]:
                cleaned = clean_up_output(output)
                cleaned_outputs.append(cleaned)
            cell["outputs"] = cleaned_outputs
        if (metadata := cell.get("metadata", None)) and len(metadata) > 0:
            cell["metadata"] = {"tags": []}

    with open(path, "w") as file:
        json.dump(notebook_json, file, sort_keys=True, indent=1, ensure_ascii=False)

    assert os.path.exists(unclean_path)
    assert os.path.exists(path)

    return path


def filter_notebooks(paths: list[str], ignores: list[str] = IGNORE_LIST) -> list[str]:
    ignore_patterns = [re.compile(ignore) for ignore in sorted(ignores)]

    def is_valid_notebook(path: str) -> bool:
        ignore: bool = any(pattern.match(path) for pattern in ignore_patterns)
        return not ignore

    files = []

    for path in paths:
        if path.endswith(".ipynb") and is_valid_notebook(path):
            files.append(path)

    return sorted(set(files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    notebooks = filter_notebooks(args.paths)

    for notebook in sorted(notebooks):
        _ = clean_up_notebook(notebook)

    sys.exit(0)
