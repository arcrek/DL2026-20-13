"""Fetch only public test annotations; images and raw OCR are not downloaded."""
import argparse
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path


DATASET = "Ahren09/MMSoc_Memotion"
LABELS = {"very_negative": 0, "negative": 0, "neutral": 1,
          "positive": 2, "very_positive": 2}


def get_json(url):
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "dl2026-role-e-analysis"})
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def fetch(output):
    output = Path(output)
    annotations, revisions = [], set()
    for offset in range(0, 700, 100):
        query = urllib.parse.urlencode({"dataset": DATASET, "config": "default",
                                       "split": "test", "offset": offset, "length": 100})
        payload = get_json("https://datasets-server.huggingface.co/rows?" + query)
        if payload["num_rows_total"] != 700:
            raise ValueError("Unexpected test split size")
        for item in payload["rows"]:
            row, index = item["row"], item["row_idx"]
            if index != len(annotations) or item.get("truncated_cells"):
                raise ValueError("Unexpected ordering or truncated annotation")
            source = row.get("image", {}).get("src", "")
            match = re.search(r"/--/([0-9a-f]{40})/--/", source)
            if match:
                revisions.add(match.group(1))
            annotations.append({"id": f"test_{index:05d}", "label": LABELS[row["sentiment"]],
                                "text": row["text_corrected"] or "", "humor": row["humor"],
                                "sarcasm": row["sarcasm"], "motivational": row["motivational"]})
        print(f"Fetched {len(annotations)}/700 annotations", flush=True)
    if len(annotations) != 700 or len(revisions) != 1:
        raise ValueError("Incomplete test metadata or inconsistent dataset revision")
    output.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in annotations)
    output.write_text(content, encoding="utf-8")
    provenance = {"dataset": DATASET, "config": "default", "split": "test", "rows": 700,
                  "dataset_revision": next(iter(revisions)),
                  "source": "https://datasets-server.huggingface.co/rows",
                  "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                  "fields": list(annotations[0])}
    output.with_suffix(".source.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="report/inputs/test_metadata.jsonl")
    args = parser.parse_args()
    print(fetch(args.output))
