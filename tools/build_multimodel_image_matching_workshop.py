"""Generate the DIMER multi-model image-matching workshop notebook."""
# ruff: noqa: E501
from __future__ import annotations
import argparse
import json
from pathlib import Path
from multimodel_image_matching_workshop_source import CELLS

NOTEBOOK_NAME="DIMER_MultiModel_Image_Matching_Workshop.ipynb"

def build_notebook():
    rendered=[]
    for index,cell in enumerate(CELLS):
        base={"id":f"dimer-matching-workshop-{index:02d}","metadata":{},"source":cell["source"].splitlines(keepends=True)}
        if cell["kind"]=="markdown":
            rendered.append({"cell_type":"markdown",**base})
        else:
            rendered.append({"cell_type":"code","execution_count":None,"outputs":[],**base})
    return {
        "cells":rendered,
        "metadata":{
            "accelerator":"GPU",
            "colab":{"gpuType":"T4","provenance":[]},
            "dimer":{
                "notebook_profile":"TASK-INFERENCE",
                "notebook_mode":"WORKSHOP",
                "notebook_spec":"2.1",
                "standalone":True,
                "carrier":"self-contained comparative matcher benchmark with two embedded pinned DIMER packages",
                "clean_runtime_evidence":"pending",
                "generated_from":{
                    "repository":"kurtvalcorza/lightglue-matching-pipeline",
                    "source":"tools/multimodel_image_matching_workshop_source.py",
                    "generator":"tools/build_multimodel_image_matching_workshop.py",
                },
            },
            "kernelspec":{"display_name":"Python 3","name":"python3"},
            "language_info":{"name":"python"},
        },
        "nbformat":4,
        "nbformat_minor":5,
    }

def serialized():
    return json.dumps(build_notebook(),indent=1,ensure_ascii=False)+"\n"

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path)
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    out=args.out or repo/"tutorials"/NOTEBOOK_NAME
    content=serialized()
    if args.check:
        if not out.exists() or out.read_text(encoding="utf-8")!=content:
            raise SystemExit(f"STALE: {out}; regenerate the workshop notebook")
        print(f"OK: {out}")
        return 0
    out.write_text(content,encoding="utf-8")
    print(out)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
