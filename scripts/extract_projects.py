#!/usr/bin/env python3
"""Stream the TravisTorrent dump and emit compact build level CSVs.

The published dump stores one row per (job, built commit) pair, so a single
build appears many times. This script performs the job to build aggregation
while streaming and writes one chronologically ordered file per project.

The extraction records, per build:

  n_jobs_passed / n_jobs_failed / n_jobs_errored / n_jobs_canceled
      job status counts, so that both the lenient label (any non passed job)
      and the strict label (any failed or errored job) can be derived later
  job_minutes_total
      the sum of job durations, which is the compute a build consumes, as
      opposed to tr_duration (the maximum job duration, a wall clock proxy)
  latency_rows_disagree
      1 if tr_ci_latency differs between the rows of the same build, so the
      claim that latency is a build level quantity can be audited
  status_rows_disagree
      1 if tr_status differs between the rows of the same build

It also counts distinct builds for every project in the dump and writes the
ranking to data/build_level/PROJECT_RANKING.json, so that the project
selection rule stated in the paper can be checked against the dump.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parent.parent

# The ten primary study projects. They are drawn from the sixteen most active
# projects of the snapshot (by distinct builds, apache/sling excluded) so that
# the sample spans failure prevalence from under ten percent to over sixty.
PRIMARY_PROJECTS = [
    "jruby/jruby",
    "apache/jackrabbit-oak",
    "gradle/gradle",
    "opf/openproject",
    "rails/rails",
    "ruby/ruby",
    "Graylog2/graylog2-server",
    "cloudfoundry/cloud_controller_ng",
    "facebook/presto",
    "github/linguist",
]

# The four more active projects that the primary rule skipped. They are
# extracted and evaluated as a robustness set.
ROBUSTNESS_PROJECTS = [
    "openSUSE/open-build-service",
    "mitchellh/vagrant",
    "rubinius/rubinius",
    "orbeon/orbeon-forms",
]

DEFAULT_PROJECTS = PRIMARY_PROJECTS + ROBUSTNESS_PROJECTS

KEEP_COLUMNS = [
    "gh_project_name", "gh_lang", "gh_is_pr", "gh_team_size",
    "git_num_commits", "git_num_committers", "gh_src_churn", "gh_test_churn",
    "gh_files_added", "gh_files_deleted", "gh_files_modified",
    "gh_tests_added", "gh_tests_deleted",
    "gh_src_files", "gh_doc_files", "gh_other_files",
    "gh_commits_on_files_touched", "gh_sloc",
    "gh_test_lines_per_kloc", "gh_test_cases_per_kloc", "gh_asserts_cases_per_kloc",
    "gh_by_core_team_member", "gh_num_issue_comments", "gh_num_commit_comments",
    "gh_num_pr_comments", "gh_description_complexity",
    "tr_build_id", "tr_status", "tr_duration", "tr_started_at",
    "tr_setup_time", "tr_tests_ok", "tr_tests_fail", "tr_tests_run",
    "tr_tests_skipped", "tr_testduration", "tr_purebuildduration",
    "tr_num_jobs", "tr_prev_build", "tr_ci_latency", "git_branch",
]

EXTRA_COLUMNS = [
    "tr_job_count", "n_jobs_passed", "n_jobs_failed", "n_jobs_errored",
    "n_jobs_canceled", "n_jobs_other", "job_minutes_total",
    "latency_rows_disagree", "status_rows_disagree",
    "is_failure", "is_failure_strict",
]

OUT_COLUMNS = KEEP_COLUMNS + EXTRA_COLUMNS


def _to_float(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:
        return None
    return v


def extract(dump_path: Path, projects: list[str], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = set(projects)
    acc: dict[str, dict[str, dict]] = {p: {} for p in projects}
    builds_per_project: dict[str, set] = {}
    scanned = 0

    with gzip.open(dump_path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scanned += 1
            project = row.get("gh_project_name")
            build_id = row.get("tr_build_id")
            if not build_id:
                continue
            builds_per_project.setdefault(project, set()).add(build_id)
            if project not in targets:
                continue

            builds = acc[project]
            record = builds.get(build_id)
            status = str(row.get("tr_status", "")).strip().lower()
            duration = _to_float(row.get("tr_duration"))
            latency = _to_float(row.get("tr_ci_latency"))
            job_id = row.get("tr_job_id")

            if record is None:
                record = {col: row.get(col, "") for col in KEEP_COLUMNS}
                record["_durations"] = {}
                record["_latencies"] = []
                record["_statuses"] = Counter()
                record["_rows"] = 0
                builds[build_id] = record

            record["_rows"] += 1
            record["_statuses"][status] += 1
            # One duration per job id (a job can appear on several commit rows).
            if duration is not None and job_id:
                record["_durations"][job_id] = duration
            if latency is not None:
                record["_latencies"].append(latency)

            if scanned % 1_000_000 == 0:
                held = sum(len(v) for v in acc.values())
                print(f"  scanned {scanned:,} rows, {held:,} builds buffered", flush=True)

    manifest = {}
    for project in projects:
        builds = acc[project]
        rows = []
        for build_id, record in builds.items():
            out = {col: record.get(col, "") for col in KEEP_COLUMNS}
            durations = list(record["_durations"].values())
            latencies = record["_latencies"]
            statuses = record["_statuses"]
            n_pass = statuses.get("passed", 0)
            n_fail = statuses.get("failed", 0)
            n_err = statuses.get("errored", 0)
            n_can = statuses.get("canceled", 0) + statuses.get("cancelled", 0)
            n_other = record["_rows"] - n_pass - n_fail - n_err - n_can
            out["tr_build_id"] = build_id
            out["tr_duration"] = max(durations) if durations else ""
            out["job_minutes_total"] = round(sum(durations) / 60.0, 3) if durations else ""
            out["tr_ci_latency"] = statistics.median(latencies) if latencies else ""
            out["latency_rows_disagree"] = int(len(set(latencies)) > 1)
            out["status_rows_disagree"] = int(len([k for k, v in statuses.items() if v > 0]) > 1)
            out["tr_job_count"] = record["_rows"]
            out["n_jobs_passed"] = n_pass
            out["n_jobs_failed"] = n_fail
            out["n_jobs_errored"] = n_err
            out["n_jobs_canceled"] = n_can
            out["n_jobs_other"] = n_other
            lenient = int((record["_rows"] - n_pass) > 0)
            strict = int((n_fail + n_err) > 0)
            out["tr_status"] = "failed" if lenient else "passed"
            out["is_failure"] = lenient
            out["is_failure_strict"] = strict
            rows.append(out)

        rows.sort(key=lambda r: (str(r.get("tr_started_at") or ""), str(r["tr_build_id"])))
        safe = project.replace("/", "_")
        dest = out_dir / f"{safe}.csv"
        with dest.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        failures = sum(int(r["is_failure"]) for r in rows)
        strict_failures = sum(int(r["is_failure_strict"]) for r in rows)
        manifest[project] = {
            "file": dest.name,
            "role": "primary" if project in PRIMARY_PROJECTS else "robustness",
            "builds": len(rows),
            "failures": failures,
            "failure_rate_pct": round(failures / max(len(rows), 1) * 100, 2),
            "failures_strict": strict_failures,
            "failure_rate_strict_pct": round(strict_failures / max(len(rows), 1) * 100, 2),
            "source_records": sum(int(r["tr_job_count"]) for r in rows),
            "builds_with_latency_disagreement": sum(int(r["latency_rows_disagree"]) for r in rows),
            "builds_with_status_disagreement": sum(int(r["status_rows_disagree"]) for r in rows),
            "first_build": rows[0]["tr_started_at"] if rows else None,
            "last_build": rows[-1]["tr_started_at"] if rows else None,
            "size_bytes": dest.stat().st_size,
        }
        print(f"  {project:36s} {len(rows):6d} builds  {manifest[project]['failure_rate_pct']:5.2f}% failing "
              f"(strict {manifest[project]['failure_rate_strict_pct']:5.2f}%) -> {dest.name}", flush=True)

    ranking = sorted(((p, len(b)) for p, b in builds_per_project.items()), key=lambda kv: -kv[1])
    ranking_payload = {
        "source_dump": dump_path.name,
        "projects_in_dump": len(ranking),
        "top_30_by_distinct_builds": [
            {"rank": i + 1, "project": p, "builds": n,
             "role": ("primary" if p in PRIMARY_PROJECTS else
                      "robustness" if p in ROBUSTNESS_PROJECTS else
                      "excluded_degenerate" if p == "apache/sling" else "not_used")}
            for i, (p, n) in enumerate(ranking[:30])
        ],
    }
    (out_dir / "PROJECT_RANKING.json").write_text(json.dumps(ranking_payload, indent=2), encoding="utf-8")

    meta = {
        "source_dump": dump_path.name,
        "source": "TravisTorrent 5/3/2016 snapshot",
        "rows_scanned": scanned,
        "aggregation": "job records grouped by tr_build_id",
        "label_lenient": "is_failure = 1 if any job row of the build has a status other than passed",
        "label_strict": "is_failure_strict = 1 if any job row of the build has status failed or errored",
        "projects": manifest,
    }
    (out_dir / "EXTRACTION_MANIFEST.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract build level project slices from TravisTorrent.")
    parser.add_argument("--dump", default=str(ROOT / "data/raw/travistorrent_5_3_2016.csv.gz"))
    parser.add_argument("--out_dir", default=str(ROOT / "data/build_level"))
    parser.add_argument("--projects", nargs="+", default=DEFAULT_PROJECTS)
    args = parser.parse_args()

    dump = Path(args.dump)
    if not dump.exists():
        raise SystemExit(f"Dump not found at {dump}. Run scripts/download_data.py first.")

    print(f"Extracting {len(args.projects)} projects from {dump.name}")
    meta = extract(dump, args.projects, Path(args.out_dir))
    total = sum(v["builds"] for v in meta["projects"].values())
    print(f"\nTotal build level records written: {total:,}")


if __name__ == "__main__":
    main()
