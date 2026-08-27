"""
Crawl StackOverflow java + python posts created on or after 2025-01-01.

For each question we need:
  - Title
  - Accepted Answer Body (must have accepted_answer_id; skip otherwise)
  - Catalog = JAVA / PYTHON (tag)
  - creation_date, question_id, score, view_count

Produces:
  data/unseen_post2025_raw.csv

Crawl plan:
  - Stack Exchange API v2.3
  - `fromdate` = unix(2025-01-01 00:00:00 UTC)
  - sort=creation, order=desc (paginate until `has_more` == false or page cap)
  - Uses filter !9_bDDY3A3 (question with accepted answer inline) — fallback to
    per-answer fetch if body not included.
  - Incremental save every 200 posts; resumable via --skip-existing.

Usage:
    python3 crawl_post_2025.py --fromdate 2025-01-01 --max-pages 200 --key <api_key_optional>
"""
import os
import sys
import time
import argparse
import datetime
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "unseen_post2025_raw.csv")

BASE = "https://api.stackexchange.com/2.3"
# Filter that includes: question.body, answers.body, accepted_answer inline
# Built from StackExchange "create filter" — equivalent to "withbody"
FILTER_Q = "withbody"


def to_unix(date_str):
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
    return int(dt.timestamp())


def month_windows(fromdate_str, todate_str):
    """Yield (label, fromdate_ts, todate_ts) monthly windows."""
    start = datetime.datetime.strptime(fromdate_str, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
    end = datetime.datetime.strptime(todate_str, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
    cur = start
    while cur < end:
        if cur.month == 12:
            nxt = cur.replace(year=cur.year + 1, month=1, day=1)
        else:
            nxt = cur.replace(month=cur.month + 1, day=1)
        win_end = min(nxt, end)
        yield f"{cur.strftime('%Y-%m')}", int(cur.timestamp()), int(win_end.timestamp())
        cur = nxt


def fetch_questions(tag, fromdate_str, todate_str, max_pages, page_size, api_key=None, sleep=0.5):
    """Paginate through questions for a single tag, using monthly date windows to
    work around the 25-page hard cap.
    """
    all_rows = []
    for label, fd, td in month_windows(fromdate_str, todate_str):
        rows = []
        page = 1
        while page <= max_pages:
            params = {
                "site": "stackoverflow",
                "tagged": tag,
                "fromdate": fd,
                "todate": td,
                "order": "desc",
                "sort": "creation",
                "page": page,
                "pagesize": page_size,
                "filter": FILTER_Q,
            }
            if api_key:
                params["key"] = api_key
            try:
                r = requests.get(f"{BASE}/questions", params=params, timeout=30)
                data = r.json()
                if r.status_code >= 400:
                    err_id = data.get("error_id") if isinstance(data, dict) else None
                    err_msg = data.get("error_message") if isinstance(data, dict) else r.text[:150]
                    print(f"[{tag} {label} page {page}] HTTP {r.status_code} err_id={err_id} {err_msg}")
                    # error_id 502 is "too many ids"; for 400 on pagination, just move on
                    break
            except Exception as e:
                print(f"[{tag} {label} page {page}] ERROR {e}")
                time.sleep(5); break
            items = data.get("items", [])
            if not items:
                break
            for it in items:
                if it.get("accepted_answer_id"):
                    rows.append({
                        "question_id": it["question_id"],
                        "Title": it["title"],
                        "Votes": it.get("score", 0),
                        "Views": it.get("view_count", 0),
                        "creation_date": it["creation_date"],
                        "tags": ",".join(it.get("tags", [])),
                        "accepted_answer_id": it["accepted_answer_id"],
                        "Catalog": tag.upper(),
                    })
            quota = data.get("quota_remaining", -1)
            backoff = data.get("backoff", 0)
            print(f"[{tag} {label}] p{page:<2} items={len(items):<4} acc={len(rows):<4} "
                  f"qleft={quota} bf={backoff}")
            if backoff:
                time.sleep(backoff)
            if not data.get("has_more"):
                break
            page += 1
            time.sleep(sleep)
        all_rows.extend(rows)
        print(f"[{tag} {label}] -> +{len(rows)} (cumulative {len(all_rows)})")
    return all_rows


def fetch_answer_bodies(answer_ids, api_key=None, sleep=0.5):
    """Batch-fetch answer bodies (up to 100 ids per request)."""
    bodies = {}
    for i in range(0, len(answer_ids), 100):
        chunk = answer_ids[i:i + 100]
        ids_str = ";".join(str(x) for x in chunk)
        params = {"site": "stackoverflow", "filter": FILTER_Q}
        if api_key:
            params["key"] = api_key
        try:
            r = requests.get(f"{BASE}/answers/{ids_str}", params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"[answers {i}..] ERROR {e}")
            time.sleep(5)
            continue
        for it in data.get("items", []):
            bodies[it["answer_id"]] = it.get("body", "")
        quota = data.get("quota_remaining", -1)
        backoff = data.get("backoff", 0)
        print(f"[answers] batch {i//100+1} fetched {len(bodies)} total  quota_left={quota}  backoff={backoff}")
        if backoff:
            time.sleep(backoff)
        time.sleep(sleep)
    return bodies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fromdate", default="2025-01-01")
    ap.add_argument("--todate", default=datetime.datetime.utcnow().strftime("%Y-%m-%d"))
    ap.add_argument("--max-pages", type=int, default=25)
    ap.add_argument("--page-size", type=int, default=100)
    ap.add_argument("--key", default=None, help="StackExchange API key (raises quota)")
    args = ap.parse_args()

    print(f"Crawling questions created {args.fromdate} -> {args.todate} "
          f"(monthly windows) ...")

    all_rows = []
    for tag in ["java", "python"]:
        print(f"\n=== tag={tag} ===")
        rows = fetch_questions(tag, args.fromdate, args.todate, args.max_pages,
                               args.page_size, args.key)
        print(f"[{tag}] total accepted: {len(rows)}")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if len(df) == 0:
        print("No posts returned."); return
    # Dedup across tags if a question has both java+python tags
    df = df.drop_duplicates(subset=["question_id"]).reset_index(drop=True)
    print(f"\nTotal unique accepted-answer questions: {len(df)}")

    print("Fetching accepted-answer bodies ...")
    ans_ids = df["accepted_answer_id"].tolist()
    bodies = fetch_answer_bodies(ans_ids, api_key=args.key)
    df["Accepted_Answer_Body"] = df["accepted_answer_id"].map(bodies).fillna("")
    df = df[df["Accepted_Answer_Body"].str.strip() != ""].reset_index(drop=True)
    print(f"With non-empty accepted answer bodies: {len(df)}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {OUT}  ({len(df)} rows)")
    print("\nPer Catalog:")
    print(df["Catalog"].value_counts())


if __name__ == "__main__":
    main()
