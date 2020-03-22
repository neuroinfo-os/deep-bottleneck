from pathlib import Path
import json
import shutil
from copy import deepcopy
from typing import *

Config = Dict[str, Any]

def main():
    convert_configs(
        src=Path("cohort_18_paper/"), dst=Path("cohort_19_adam_paper/"), adapt=change
    )


def convert_configs(
    src: Path, dst: Path, adapt: Callable[[Config], Config]
):
    shutil.rmtree(dst, ignore_errors=True)
    relevant_configs = (p for p in src.glob("**/*.json") if not p.parts[-2].startswith("."))
    for p in relevant_configs:
        d = read_json(p)
        d = adapt(d)
        new_path = dst.joinpath(*p.parts[1:])
        new_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(new_path, d)


def read_json(p: Path) -> Config:
    return json.loads(p.read_text())


def write_json(p: Path, d: Config):
    return p.write_text(json.dumps(d, indent=4))


def change(d: Config) -> Config:
    d = deepcopy(d)
    d["optimizer"] = "adam"
    return d


if __name__ == "__main__":
    main()
