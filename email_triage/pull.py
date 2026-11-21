"""Pull message metadata (From/Subject/Date/snippet/labels) from the Gmail API into a JSONL file.

Usage: python -m email_triage.pull OUT.jsonl [LIMIT]

Needs requirements-pull.txt and an OAuth token for the Gmail API with the gmail.readonly scope, saved as an
authorized-user JSON file (the token.json that Google's Python quickstart writes). Set GMAIL_TOKEN to its path
(default: token.json). Metadata only, no bodies. Output is private data: keep it out of version control
(see .gitignore). Next step: python -m email_triage.prepare (turns this output into the dataset schema).
"""
from __future__ import annotations

import json
import os
import sys

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
RETRIES = 5  # googleapiclient retries 429/5xx with exponential backoff


def build_service(token_path: str):
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    if not os.path.exists(token_path):
        raise SystemExit(f"Gmail token not found at {token_path!r}. Create one with Google's OAuth quickstart "
                         "(scope gmail.readonly) and set GMAIL_TOKEN to its path.")
    creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def list_ids(s, limit: int) -> list[str]:
    """Ids of up to `limit` received messages, newest first."""
    ids, tok = [], None
    while len(ids) < limit:
        r = s.users().messages().list(userId="me", maxResults=min(500, limit - len(ids)), pageToken=tok,
                                      q="-in:sent -in:drafts").execute(num_retries=RETRIES)
        ids += [m["id"] for m in r.get("messages", [])]
        tok = r.get("nextPageToken")
        if not tok:
            break
    return ids[:limit]


def main(out_path: str, limit: int = 1500) -> int:
    s = build_service(os.environ.get("GMAIL_TOKEN", "token.json"))
    labs = {lab["id"]: lab["name"]
            for lab in s.users().labels().list(userId="me").execute(num_retries=RETRIES)["labels"]}
    ids = list_ids(s, int(limit))
    with open(out_path, "w") as out:
        for i in ids:
            m = s.users().messages().get(userId="me", id=i, format="metadata",
                                         metadataHeaders=["From", "Subject", "Date"]).execute(num_retries=RETRIES)
            h = {x["name"]: x["value"] for x in m["payload"]["headers"]}
            out.write(json.dumps({"id": i, "from": h.get("From", ""), "subject": h.get("Subject", ""),
                                  "date": h.get("Date", ""), "snippet": m.get("snippet", ""),
                                  "labels": [labs.get(lab, lab) for lab in m.get("labelIds", [])]}) + "\n")
    print(len(ids))
    return len(ids)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    main(*sys.argv[1:])
