"""Pull message metadata (From/Subject/Date/snippet/labels) from the Gmail API into a JSONL file.

Usage: python -m email_triage.pull OUT.jsonl
Needs an authenticated Gmail client: set GMAIL_BUILD_SERVICE_DIR to a directory containing a
`google_api.py` that exposes build_service(name, version). Metadata only, no bodies. Output is private
data: keep it out of version control (see .gitignore).
"""
from __future__ import annotations

import json
import os
import sys


def main(out_path: str, limit: int = 1500) -> int:
    sys.path.insert(0, os.environ["GMAIL_BUILD_SERVICE_DIR"])
    from google_api import build_service  # type: ignore

    s = build_service("gmail", "v1")
    labs = {lab["id"]: lab["name"] for lab in s.users().labels().list(userId="me").execute()["labels"]}
    ids, tok = [], None
    while len(ids) < limit:
        r = s.users().messages().list(userId="me", maxResults=500, pageToken=tok, q="-in:sent -in:drafts").execute()
        ids += [m["id"] for m in r.get("messages", [])]
        tok = r.get("nextPageToken")
        if not tok:
            break
    with open(out_path, "w") as out:
        for i in ids:
            m = s.users().messages().get(userId="me", id=i, format="metadata",
                                         metadataHeaders=["From", "Subject", "Date"]).execute()
            h = {x["name"]: x["value"] for x in m["payload"]["headers"]}
            out.write(json.dumps({"id": i, "from": h.get("From", ""), "subject": h.get("Subject", ""),
                                  "date": h.get("Date", ""), "snippet": m.get("snippet", ""),
                                  "labels": [labs.get(lab, lab) for lab in m.get("labelIds", [])]}) + "\n")
    print(len(ids))
    return len(ids)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
