#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WF-P read-only source preflight.

Purpose
-------
Verify authoritative upstream source/cohort identity for
"WF-P: Population Geometry of Arterial Waveform Morphology" without opening
waveform arrays and without calculating any morphology/population scientific
effect.

This utility reads metadata, manifests, JSON/text provenance, source-code
hashes, and file existence/size only. It never calls np.load, wfdb, PCA,
covariance, waveform reconstruction, or clinical-effect code.

Decision semantics
------------------
WFP_SOURCE_PREFLIGHT_PASS:
    Legacy WF1/WF2 authoritative sources pass and the optional future bank is
    present, complete, and metadata-consistent. Cohort/split freeze may proceed.
WFP_SOURCE_PREFLIGHT_PARTIAL:
    Legacy WF1/WF2 authoritative sources pass, but the preferred future-bank
    source is absent/incomplete/not metadata-clean. Scientific effects remain
    blocked; a cohort-source decision is still required.
WFP_SOURCE_PREFLIGHT_FAIL:
    One or more required authoritative legacy/WF2 checks failed.

Passing this preflight NEVER authorizes scientific-effect analysis. A separate
WF-P cohort/split freeze and final scientific-spec freeze are required first.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_VERSION = "wfp_source_preflight_v0.1_2026-08-20"

# Authoritative values recovered from the already-passed upstream source audit.
EXPECTED = {
    "pilot_manifest_sha256": "ad02f524e94b0c8c268218bf9fdd088f2b104a8c52f210f6e9ee42fa37403af2",
    "validation_manifest_sha256": "d85e58aea6067565d3e52a82c54a1c28eaaa84ceed12f5af41a78df0806aae68",
    "pilot_extractor_sha256": "64126b940bde0f05a6bd995b51789e74e651c4c8e311cb4fb90eac25e2ae587e",
    "validation_extractor_sha256": "2a4a3b77e8db277b3f332147ade3d7c9b3e7e25be8b99c01d52145a1a732a581",
    "wp2_run1_sha256": "811775f50283a8f5d813d517f6c8c4bc3ed846fa994c3145eda96404ff04ee01",
    "wp2_frozen_spec_json_sha256": "c72c05ffbf26e50f60d548f5a6d86dd392c134d568b26abae2f38c6863705ab7",
    "validation_record_list_sha256": "48de2a92e38d1ea76281352f26f0cf8ea5022c801ac136c7532648607cfbef9d",
}

PILOT_REQUIRED_FIELDS = {
    "patient_id", "record_path", "record_name", "pn_dir", "abp_source_name",
    "fs", "duration_sec", "start_sample", "file",
}
VALIDATION_REQUIRED_FIELDS = PILOT_REQUIRED_FIELDS | {"selection_order_index"}
SCAN_REQUIRED_FIELDS = {
    "selection_order_index", "patient_id", "outcome", "records_inspected", "detail"
}
FUTURE_REQUIRED_FIELDS = {
    "patient_id", "selection_order_index", "record_path", "record_name", "pn_dir",
    "abp_source_name", "fs", "duration_sec", "start_sample", "file",
    "overlap_labels_json",
}
ABP_ALIASES = {"ABP", "ART"}

FORBIDDEN_OUTPUT_TERMS = {
    "mean_morphology", "morphology_vector", "population_pca", "pca_score",
    "covariance_matrix", "eigenvalue", "effective_rank", "deformation_score",
    "harmonic_score", "z_morph", "z_state", "clinical_effect", "treatment_effect",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_lines(lines: Iterable[str]) -> str:
    h = hashlib.sha256()
    for x in sorted(str(s) for s in lines):
        h.update(x.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        return list(r.fieldnames or []), rows


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def as_int(x: Any) -> Optional[int]:
    try:
        return int(float(x))
    except Exception:
        return None


def expanded(path: str | Path) -> Path:
    return Path(os.path.expanduser(str(path))).resolve()


@dataclass
class Audit:
    checks: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str, category: str = "required") -> None:
        self.checks.append({
            "name": name,
            "pass": bool(passed),
            "detail": str(detail),
            "category": category,
        })

    def required_pass(self) -> bool:
        return all(x["pass"] for x in self.checks if x.get("category") == "required")

    def future_pass(self) -> bool:
        items = [x for x in self.checks if x.get("category") == "future"]
        return bool(items) and all(x["pass"] for x in items)


def locate_named_file(
    basename: str,
    explicit: Optional[str],
    roots: Sequence[Path],
) -> Tuple[Optional[Path], List[str]]:
    if explicit:
        p = expanded(explicit)
        return (p if p.is_file() else None), ([str(p)] if p.is_file() else [])

    candidates: List[Path] = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        direct = [
            root / basename,
            root / "code" / basename,
            root / "docs" / basename,
            root / "freeze" / basename,
        ]
        for p in direct:
            if p.is_file() and str(p) not in seen:
                candidates.append(p)
                seen.add(str(p))
        # Search only named research roots supplied below; no home-wide recursion.
        try:
            for p in root.rglob(basename):
                if p.is_file() and str(p) not in seen:
                    candidates.append(p)
                    seen.add(str(p))
        except Exception:
            pass

    candidates = sorted(candidates, key=lambda p: str(p))
    return (candidates[0] if candidates else None), [str(x) for x in candidates]


def validate_rows_basic(
    audit: Audit,
    label: str,
    fields: List[str],
    rows: List[Dict[str, str]],
    required_fields: set[str],
    expected_n: int,
    category: str = "required",
) -> set[str]:
    missing = sorted(required_fields - set(fields))
    audit.add(f"{label}_required_fields", not missing,
              "complete" if not missing else "missing=" + ",".join(missing), category)

    pids = [str(r.get("patient_id", "")).strip() for r in rows]
    pids_nonblank = [x for x in pids if x]
    unique = set(pids_nonblank)
    audit.add(f"{label}_row_count", len(rows) == expected_n,
              f"rows={len(rows)} expected={expected_n}", category)
    audit.add(f"{label}_unique_patient_count", len(unique) == expected_n,
              f"unique={len(unique)} expected={expected_n}", category)
    audit.add(f"{label}_blank_patient_ids", len(pids_nonblank) == len(rows),
              f"blank={len(rows)-len(pids_nonblank)}", category)

    bad_fs = 0
    bad_dur = 0
    bad_alias = 0
    for r in rows:
        fs = as_float(r.get("fs"))
        dur = as_int(r.get("duration_sec"))
        alias = str(r.get("abp_source_name", "")).strip()
        if fs is None or abs(fs - 125.0) > 1e-9:
            bad_fs += 1
        if dur != 1800:
            bad_dur += 1
        if alias not in ABP_ALIASES:
            bad_alias += 1
    audit.add(f"{label}_all_125hz", bad_fs == 0, f"bad_rows={bad_fs}", category)
    audit.add(f"{label}_all_30min", bad_dur == 0, f"bad_rows={bad_dur}", category)
    audit.add(f"{label}_abp_aliases", bad_alias == 0, f"bad_rows={bad_alias}", category)
    return unique


def audit_future_bank(
    audit: Audit,
    future_dir: Path,
    legacy_pids: set[str],
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": str(future_dir),
        "present": future_dir.is_dir(),
        "complete": False,
        "ready_for_cohort_freeze": False,
    }
    if not future_dir.is_dir():
        # Absence/incompletion is not a failure of upstream WF1/WF2 integrity.
        audit.add("future_bank_present", False, "directory absent", category="future")
        return info

    paths = {
        "manifest": future_dir / "manifest.csv",
        "config": future_dir / "FROZEN_EXTRACTION_CONFIG.json",
        "state": future_dir / "resume_state.json",
        "record_hash": future_dir / "FROZEN_RECORD_LIST_SHA256.txt",
        "provenance": future_dir / "PROVENANCE.json",
    }
    for k, p in paths.items():
        audit.add(f"future_{k}_present", p.is_file(), str(p), category="future")
    if not all(p.is_file() for p in paths.values()):
        return info

    fields, rows = read_csv(paths["manifest"])
    missing = sorted(FUTURE_REQUIRED_FIELDS - set(fields))
    audit.add("future_manifest_required_fields", not missing,
              "complete" if not missing else "missing=" + ",".join(missing), category="future")

    pids = [str(r.get("patient_id", "")).strip() for r in rows]
    unique = {x for x in pids if x}
    dup = len(pids) - len(unique)
    audit.add("future_unique_patients", dup == 0 and len(unique) == len(rows),
              f"rows={len(rows)} unique={len(unique)} duplicates={dup}", category="future")

    bad_fs = sum(1 for r in rows if as_float(r.get("fs")) is None or abs(float(r.get("fs")) - 125.0) > 1e-9)
    bad_dur = sum(1 for r in rows if as_int(r.get("duration_sec")) != 1800)
    bad_alias = sum(1 for r in rows if str(r.get("abp_source_name", "")).strip() not in ABP_ALIASES)
    audit.add("future_all_125hz", bad_fs == 0, f"bad_rows={bad_fs}", category="future")
    audit.add("future_all_30min", bad_dur == 0, f"bad_rows={bad_dur}", category="future")
    audit.add("future_abp_aliases", bad_alias == 0, f"bad_rows={bad_alias}", category="future")

    missing_case_files = 0
    empty_case_files = 0
    for r in rows:
        rel = str(r.get("file", "")).strip()
        if not rel:
            missing_case_files += 1
            continue
        p = Path(rel)
        if not p.is_absolute():
            p = future_dir / p
        if not p.exists():
            missing_case_files += 1
        elif p.stat().st_size == 0:
            empty_case_files += 1
    audit.add("future_case_files_exist_without_opening", missing_case_files == 0 and empty_case_files == 0,
              f"missing={missing_case_files} empty={empty_case_files}; arrays_opened=NO", category="future")

    cfg = read_json(paths["config"])
    st = read_json(paths["state"])
    prov = read_json(paths["provenance"])
    target_n = as_int(cfg.get("target_n"))
    cfg_fs = as_float(cfg.get("required_fs"))
    cfg_dur = as_int(cfg.get("duration_min"))
    cfg_scan = as_float(cfg.get("max_scan_hours"))
    cfg_version = str(cfg.get("script_version", ""))
    completed = bool(st.get("completed", False))
    completed_reason = str(st.get("completed_reason", ""))

    audit.add("future_config_required_fs", cfg_fs is not None and abs(cfg_fs - 125.0) < 1e-9,
              f"required_fs={cfg_fs}", category="future")
    audit.add("future_config_duration", cfg_dur == 30, f"duration_min={cfg_dur}", category="future")
    audit.add("future_config_scan_extent", cfg_scan is not None and abs(cfg_scan - 6.0) < 1e-9,
              f"max_scan_hours={cfg_scan}", category="future")
    audit.add("future_config_script_version", cfg_version == "future_bank5000_v1_2026-08-18",
              f"script_version={cfg_version}", category="future")
    audit.add("future_completed", completed,
              f"completed={completed} reason={completed_reason!r} target_n={target_n} rows={len(rows)}",
              category="future")

    record_hash = paths["record_hash"].read_text(encoding="utf-8").strip()
    prov_record_hash = str(prov.get("record_list_sha256", "")).strip()
    audit.add("future_record_hash_provenance_match", bool(record_hash) and record_hash == prov_record_hash,
              "match" if record_hash == prov_record_hash else "mismatch", category="future")

    local_future_script = future_dir.parent.parent / "code" / "00c_extract_mimic_future_bank5000.py"
    cfg_script_hash = str(cfg.get("script_sha256", "")).strip()
    prov_script_hash = str(prov.get("script_sha256", "")).strip()
    if local_future_script.is_file():
        local_hash = sha256_file(local_future_script)
        ok = bool(cfg_script_hash) and local_hash == cfg_script_hash and (
            not prov_script_hash or prov_script_hash == local_hash
        )
        audit.add("future_extractor_hash_chain", ok,
                  f"local={local_hash} config={cfg_script_hash} provenance={prov_script_hash or 'NA'}",
                  category="future")
    else:
        audit.add("future_extractor_hash_chain", False,
                  f"missing={local_future_script}", category="future")

    overlap_n = len(unique.intersection(legacy_pids))
    label_counts: Dict[str, int] = {}
    bad_overlap_json = 0
    for r in rows:
        try:
            labs = json.loads(r.get("overlap_labels_json", "[]") or "[]")
            if not isinstance(labs, list):
                raise ValueError("not list")
        except Exception:
            bad_overlap_json += 1
            labs = []
        for lab in labs:
            label_counts[str(lab)] = label_counts.get(str(lab), 0) + 1
    audit.add("future_overlap_json_parse", bad_overlap_json == 0,
              f"bad_rows={bad_overlap_json}", category="future")

    info.update({
        "manifest_rows": len(rows),
        "unique_patients": len(unique),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "config_sha256": sha256_file(paths["config"]),
        "record_list_sha256": record_hash,
        "target_n": target_n,
        "completed": completed,
        "completed_reason": completed_reason,
        "legacy_overlap_count_direct": overlap_n,
        "overlap_label_counts": dict(sorted(label_counts.items())),
        "patient_set_sha256": sha256_lines(unique),
        "ready_for_cohort_freeze": completed and audit.future_pass(),
    })
    return info


def run_preflight(
    project_root: Path,
    outdir: Path,
    wp2_run1_arg: Optional[str] = None,
    wp2_spec_arg: Optional[str] = None,
    future_bank_arg: Optional[str] = None,
    pin_authoritative_hashes: bool = True,
    expected_pilot: int = 50,
    expected_validation: int = 1000,
    write_outputs: bool = True,
) -> Dict[str, Any]:
    audit = Audit()
    project_root = expanded(project_root)
    outdir = expanded(outdir)
    audit.add("project_root_exists", project_root.is_dir(), str(project_root))

    paths = {
        "pilot_manifest": project_root / "data" / "abp125_pilot50" / "manifest.csv",
        "validation_manifest": project_root / "data" / "abp125_validation1000" / "manifest.csv",
        "validation_config": project_root / "data" / "abp125_validation1000" / "FROZEN_EXTRACTION_CONFIG.json",
        "validation_record_hash": project_root / "data" / "abp125_validation1000" / "FROZEN_RECORD_LIST_SHA256.txt",
        "validation_provenance": project_root / "data" / "abp125_validation1000" / "PROVENANCE.json",
        "validation_scan_log": project_root / "data" / "abp125_validation1000" / "patient_scan_log.csv",
        "pilot_extractor": project_root / "code" / "00_extract_abp125.py",
        "validation_extractor": project_root / "code" / "00b_extract_abp125_validation1000.py",
    }
    for k, p in paths.items():
        audit.add(f"required_{k}", p.is_file(), str(p))

    roots = [
        project_root,
        expanded("~/Documents/abp_wp2"),
        expanded("~/Documents/abp_information_study"),
    ]
    wp2_run1, run1_copies = locate_named_file("wp2_run1_development50.py", wp2_run1_arg, roots)
    wp2_spec, spec_copies = locate_named_file("WP2_VALIDATION1000_FROZEN_SPEC.json", wp2_spec_arg, roots)
    audit.add("wp2_run1_resolved", wp2_run1 is not None,
              f"selected={wp2_run1}; copies={len(run1_copies)}")
    audit.add("wp2_frozen_spec_resolved", wp2_spec is not None,
              f"selected={wp2_spec}; copies={len(spec_copies)}")

    if not all(p.is_file() for p in paths.values()) or wp2_run1 is None or wp2_spec is None:
        result = make_result(
            audit=audit,
            project_root=project_root,
            paths=paths,
            wp2_run1=wp2_run1,
            wp2_spec=wp2_spec,
            run1_copies=run1_copies,
            spec_copies=spec_copies,
            future_info={"present": False, "ready_for_cohort_freeze": False},
            hashes={}, counts={}, manifest_fields={},
        )
        if write_outputs:
            write_result(outdir, result)
        return result

    p_fields, p_rows = read_csv(paths["pilot_manifest"])
    v_fields, v_rows = read_csv(paths["validation_manifest"])
    s_fields, scan_rows = read_csv(paths["validation_scan_log"])

    p_ids = validate_rows_basic(
        audit, "pilot", p_fields, p_rows, PILOT_REQUIRED_FIELDS, expected_pilot
    )
    v_ids = validate_rows_basic(
        audit, "validation", v_fields, v_rows, VALIDATION_REQUIRED_FIELDS, expected_validation
    )
    overlap = p_ids.intersection(v_ids)
    audit.add("development_validation_no_overlap", len(overlap) == 0,
              f"overlap_count={len(overlap)}")

    missing_scan = sorted(SCAN_REQUIRED_FIELDS - set(s_fields))
    audit.add("scan_log_required_fields", not missing_scan,
              "complete" if not missing_scan else "missing=" + ",".join(missing_scan))
    accepted_scan = {
        str(r.get("patient_id", "")).strip()
        for r in scan_rows
        if str(r.get("outcome", "")).strip() == "accepted"
    }
    audit.add("scan_log_accepted_matches_validation", accepted_scan == v_ids,
              f"accepted_log={len(accepted_scan)} manifest={len(v_ids)}")

    cfg = read_json(paths["validation_config"])
    prov = read_json(paths["validation_provenance"])
    cfg_checks = [
        ("validation_config_duration_min", as_int(cfg.get("duration_min")) == 30, cfg.get("duration_min")),
        ("validation_config_required_fs", as_float(cfg.get("required_fs")) is not None and abs(float(cfg.get("required_fs")) - 125.0) < 1e-9, cfg.get("required_fs")),
        ("validation_config_max_scan_hours", as_float(cfg.get("max_scan_hours")) is not None and abs(float(cfg.get("max_scan_hours")) - 6.0) < 1e-9, cfg.get("max_scan_hours")),
        ("validation_config_seed", as_int(cfg.get("seed")) == 20260810, cfg.get("seed")),
        ("validation_config_script_version", str(cfg.get("script_version", "")) == "validation1000_v1_frozen_2026-08-09", cfg.get("script_version")),
    ]
    for name, ok, val in cfg_checks:
        audit.add(name, ok, repr(val))

    pilot_sha = sha256_file(paths["pilot_manifest"])
    validation_sha = sha256_file(paths["validation_manifest"])
    pilot_script_sha = sha256_file(paths["pilot_extractor"])
    validation_script_sha = sha256_file(paths["validation_extractor"])
    run1_sha = sha256_file(wp2_run1)
    wp2_spec_sha = sha256_file(wp2_spec)
    record_sha = paths["validation_record_hash"].read_text(encoding="utf-8").strip()

    if pin_authoritative_hashes:
        pin_checks = [
            ("pilot_manifest_authoritative_hash", pilot_sha, EXPECTED["pilot_manifest_sha256"]),
            ("validation_manifest_authoritative_hash", validation_sha, EXPECTED["validation_manifest_sha256"]),
            ("pilot_extractor_authoritative_hash", pilot_script_sha, EXPECTED["pilot_extractor_sha256"]),
            ("validation_extractor_authoritative_hash", validation_script_sha, EXPECTED["validation_extractor_sha256"]),
            ("wp2_run1_authoritative_hash", run1_sha, EXPECTED["wp2_run1_sha256"]),
            ("wp2_frozen_spec_authoritative_hash", wp2_spec_sha, EXPECTED["wp2_frozen_spec_json_sha256"]),
            ("validation_record_list_authoritative_hash", record_sha, EXPECTED["validation_record_list_sha256"]),
        ]
        for name, observed, expected in pin_checks:
            audit.add(name, observed == expected,
                      f"observed={observed} expected={expected}")

    cfg_excl = str(cfg.get("exclude_manifest_sha256", "")).strip()
    cfg_script = str(cfg.get("script_sha256", "")).strip()
    prov_excl = str(prov.get("exclude_manifest_sha256", "")).strip()
    prov_script = str(prov.get("script_sha256", "")).strip()
    prov_record = str(prov.get("record_list_sha256", "")).strip()
    audit.add("validation_exclusion_hash_chain", bool(cfg_excl) and cfg_excl == pilot_sha and prov_excl == pilot_sha,
              f"pilot={pilot_sha} config={cfg_excl} provenance={prov_excl}")
    audit.add("validation_script_hash_chain", bool(cfg_script) and cfg_script == validation_script_sha and prov_script == validation_script_sha,
              f"local={validation_script_sha} config={cfg_script} provenance={prov_script}")
    audit.add("validation_record_hash_chain", bool(record_sha) and prov_record == record_sha,
              f"file={record_sha} provenance={prov_record}")

    legacy_all = p_ids.union(v_ids)
    future_dir = expanded(future_bank_arg) if future_bank_arg else project_root / "data" / "mimic_future_bank5000"
    future_info = audit_future_bank(audit, future_dir, legacy_all)

    hashes = {
        "pilot_manifest_sha256": pilot_sha,
        "validation_manifest_sha256": validation_sha,
        "pilot_extractor_sha256": pilot_script_sha,
        "validation_extractor_sha256": validation_script_sha,
        "wp2_run1_sha256": run1_sha,
        "wp2_frozen_spec_json_sha256": wp2_spec_sha,
        "validation_record_list_sha256": record_sha,
        "pilot_patient_set_sha256": sha256_lines(p_ids),
        "validation_patient_set_sha256": sha256_lines(v_ids),
    }
    counts = {
        "pilot_rows": len(p_rows),
        "pilot_unique_patients": len(p_ids),
        "validation_rows": len(v_rows),
        "validation_unique_patients": len(v_ids),
        "development_validation_overlap": len(overlap),
        "scan_log_rows": len(scan_rows),
        "scan_log_accepted_patients": len(accepted_scan),
    }
    manifest_fields = {
        "pilot": p_fields,
        "validation": v_fields,
        "scan_log": s_fields,
    }

    result = make_result(
        audit=audit,
        project_root=project_root,
        paths=paths,
        wp2_run1=wp2_run1,
        wp2_spec=wp2_spec,
        run1_copies=run1_copies,
        spec_copies=spec_copies,
        future_info=future_info,
        hashes=hashes,
        counts=counts,
        manifest_fields=manifest_fields,
    )
    if write_outputs:
        write_result(outdir, result)
    return result


def make_result(
    audit: Audit,
    project_root: Path,
    paths: Dict[str, Path],
    wp2_run1: Optional[Path],
    wp2_spec: Optional[Path],
    run1_copies: List[str],
    spec_copies: List[str],
    future_info: Dict[str, Any],
    hashes: Dict[str, Any],
    counts: Dict[str, Any],
    manifest_fields: Dict[str, Any],
) -> Dict[str, Any]:
    required_ok = audit.required_pass()
    future_ready = bool(future_info.get("ready_for_cohort_freeze", False)) and audit.future_pass()
    if not required_ok:
        decision = "WFP_SOURCE_PREFLIGHT_FAIL"
        next_action = "RESOLVE_AUTHORITATIVE_SOURCE_FAILURES"
    elif future_ready:
        decision = "WFP_SOURCE_PREFLIGHT_PASS"
        next_action = "COHORT_AND_SPLIT_FREEZE_AUTHORIZED"
    else:
        decision = "WFP_SOURCE_PREFLIGHT_PARTIAL"
        next_action = "COHORT_SOURCE_SELECTION_OR_FUTURE_BANK_COMPLETION_REQUIRED"

    return {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "decision": decision,
        "next_action": next_action,
        "project_root": str(project_root),
        "waveform_arrays_opened": False,
        "scientific_effects_opened": False,
        "scientific_effect_analysis_authorized": False,
        "cohort_split_freeze_authorized": bool(required_ok and future_ready),
        "forbidden_output_terms": sorted(FORBIDDEN_OUTPUT_TERMS),
        "authoritative_sources": {
            **{k: str(v) for k, v in paths.items()},
            "wp2_run1": str(wp2_run1) if wp2_run1 else None,
            "wp2_frozen_spec_json": str(wp2_spec) if wp2_spec else None,
            "wp2_run1_candidate_copies": run1_copies,
            "wp2_frozen_spec_candidate_copies": spec_copies,
        },
        "hashes": hashes,
        "counts": counts,
        "manifest_fields": manifest_fields,
        "future_bank": future_info,
        "checks": audit.checks,
    }


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_result(outdir: Path, result: Dict[str, Any]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    atomic_json(outdir / "WFP_SOURCE_PREFLIGHT.json", result)
    failed_required = [x for x in result["checks"] if x["category"] == "required" and not x["pass"]]
    failed_future = [x for x in result["checks"] if x["category"] == "future" and not x["pass"]]
    f = result.get("future_bank", {})
    lines = [
        "WF-P SOURCE PREFLIGHT",
        "=====================",
        f"Decision: {result['decision']}",
        f"Next action: {result['next_action']}",
        "Waveform arrays opened: NO",
        "Scientific effects opened: NO",
        "Scientific effect analysis authorized: NO",
        f"Cohort/split freeze authorized: {'YES' if result['cohort_split_freeze_authorized'] else 'NO'}",
        f"Required checks failed: {len(failed_required)}",
        f"Future-bank checks failed: {len(failed_future)}",
        "",
        "Legacy cohort counts:",
        f"  Development50: {result.get('counts', {}).get('pilot_unique_patients', 'NA')}",
        f"  Validation1000: {result.get('counts', {}).get('validation_unique_patients', 'NA')}",
        f"  overlap: {result.get('counts', {}).get('development_validation_overlap', 'NA')}",
        "",
        "Preferred future-bank source:",
        f"  present: {f.get('present', False)}",
        f"  complete: {f.get('complete', False)}",
        f"  unique patients: {f.get('unique_patients', 'NA')}",
        f"  direct overlap with legacy 1050: {f.get('legacy_overlap_count_direct', 'NA')}",
        f"  ready for cohort freeze: {f.get('ready_for_cohort_freeze', False)}",
    ]
    if failed_required:
        lines.append("\nRequired failures:")
        lines.extend(f"  - {x['name']}: {x['detail']}" for x in failed_required)
    if failed_future:
        lines.append("\nFuture-bank pending/failures:")
        lines.extend(f"  - {x['name']}: {x['detail']}" for x in failed_future)
    lines += [
        "",
        "Boundary: This output authorizes NO waveform-effect calculation.",
        "Return WFP_SOURCE_PREFLIGHT.txt and .json for cohort/split-freeze review.",
    ]
    atomic_text(outdir / "WFP_SOURCE_PREFLIGHT.txt", "\n".join(lines) + "\n")


def build_self_test_fixture(root: Path) -> None:
    pdir = root / "data" / "abp125_pilot50"
    vdir = root / "data" / "abp125_validation1000"
    fdir = root / "data" / "mimic_future_bank5000"
    code = root / "code"
    pdir.mkdir(parents=True)
    vdir.mkdir(parents=True)
    fdir.mkdir(parents=True)
    code.mkdir(parents=True)

    fields_p = ["patient_id", "record_path", "record_name", "pn_dir", "abp_source_name", "fs", "duration_sec", "start_sample", "file"]
    fields_v = ["patient_id", "selection_order_index", "record_path", "record_name", "pn_dir", "abp_source_name", "fs", "duration_sec", "start_sample", "file"]

    def write_manifest(path: Path, fields: List[str], prefix: str, n: int, offset: int) -> None:
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for i in range(n):
                pid = f"{prefix}{offset+i:03d}"
                row = {
                    "patient_id": pid,
                    "selection_order_index": str(i),
                    "record_path": f"x/{pid}/rec",
                    "record_name": "rec",
                    "pn_dir": "db/x",
                    "abp_source_name": "ABP",
                    "fs": "125",
                    "duration_sec": "1800",
                    "start_sample": "0",
                    "file": f"cases/{pid}.npz",
                }
                w.writerow({k: row[k] for k in fields})

    write_manifest(pdir / "manifest.csv", fields_p, "p", 2, 0)
    write_manifest(vdir / "manifest.csv", fields_v, "v", 3, 100)

    (code / "00_extract_abp125.py").write_text("# pilot\n", encoding="utf-8")
    (code / "00b_extract_abp125_validation1000.py").write_text("# validation\n", encoding="utf-8")
    (code / "wp2_run1_development50.py").write_text("# wp2\n", encoding="utf-8")
    (root / "WP2_VALIDATION1000_FROZEN_SPEC.json").write_text("{}\n", encoding="utf-8")
    pilot_sha = sha256_file(pdir / "manifest.csv")
    val_script_sha = sha256_file(code / "00b_extract_abp125_validation1000.py")
    record_sha = "a" * 64
    cfg = {
        "duration_min": 30, "required_fs": 125.0, "max_scan_hours": 6.0,
        "seed": 20260810, "script_version": "validation1000_v1_frozen_2026-08-09",
        "exclude_manifest_sha256": pilot_sha, "script_sha256": val_script_sha,
    }
    (vdir / "FROZEN_EXTRACTION_CONFIG.json").write_text(json.dumps(cfg), encoding="utf-8")
    (vdir / "FROZEN_RECORD_LIST_SHA256.txt").write_text(record_sha + "\n", encoding="utf-8")
    (vdir / "PROVENANCE.json").write_text(json.dumps({
        "exclude_manifest_sha256": pilot_sha, "script_sha256": val_script_sha,
        "record_list_sha256": record_sha,
    }), encoding="utf-8")
    with (vdir / "patient_scan_log.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["selection_order_index", "patient_id", "outcome", "records_inspected", "detail"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for i in range(3):
            w.writerow({"selection_order_index": i, "patient_id": f"v{100+i:03d}", "outcome": "accepted", "records_inspected": 1, "detail": "selftest"})

    # Future bank complete synthetic metadata. Create empty nonzero placeholder case files;
    # preflight only stats them and never opens them.
    (fdir / "cases").mkdir()
    ffields = sorted(FUTURE_REQUIRED_FIELDS)
    with (fdir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ffields); w.writeheader()
        for i in range(4):
            pid = f"f{i:03d}"
            rel = f"cases/{pid}.npz"
            (fdir / rel).write_bytes(b"placeholder")
            row = {k: "" for k in ffields}
            row.update({
                "patient_id": pid, "selection_order_index": str(i),
                "record_path": f"x/{pid}/rec", "record_name": "rec", "pn_dir": "db/x",
                "abp_source_name": "ABP", "fs": "125", "duration_sec": "1800",
                "start_sample": "0", "file": rel, "overlap_labels_json": "[]",
            })
            w.writerow(row)
    future_script = code / "00c_extract_mimic_future_bank5000.py"
    future_script.write_text("# future bank\n", encoding="utf-8")
    future_script_sha = sha256_file(future_script)
    fcfg = {
        "target_n": 4, "required_fs": 125.0, "duration_min": 30,
        "max_scan_hours": 6.0, "script_version": "future_bank5000_v1_2026-08-18",
        "script_sha256": future_script_sha,
    }
    (fdir / "FROZEN_EXTRACTION_CONFIG.json").write_text(json.dumps(fcfg), encoding="utf-8")
    (fdir / "resume_state.json").write_text(json.dumps({"completed": True, "completed_reason": "target_reached"}), encoding="utf-8")
    frecord = "b" * 64
    (fdir / "FROZEN_RECORD_LIST_SHA256.txt").write_text(frecord + "\n", encoding="utf-8")
    (fdir / "PROVENANCE.json").write_text(json.dumps({"record_list_sha256": frecord, "script_sha256": future_script_sha}), encoding="utf-8")


def self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="wfp_preflight_") as tmp:
        root = Path(tmp) / "project"
        build_self_test_fixture(root)
        result = run_preflight(
            project_root=root,
            outdir=root / "out",
            wp2_run1_arg=str(root / "code" / "wp2_run1_development50.py"),
            wp2_spec_arg=str(root / "WP2_VALIDATION1000_FROZEN_SPEC.json"),
            future_bank_arg=str(root / "data" / "mimic_future_bank5000"),
            pin_authoritative_hashes=False,
            expected_pilot=2,
            expected_validation=3,
            write_outputs=False,
        )
        if result["decision"] != "WFP_SOURCE_PREFLIGHT_PASS":
            print("SELF-TEST FAIL: expected PASS", result["decision"], file=sys.stderr)
            bad = [x for x in result["checks"] if not x["pass"]]
            print(json.dumps(bad, indent=2), file=sys.stderr)
            return 1
        if result["waveform_arrays_opened"] or result["scientific_effects_opened"]:
            print("SELF-TEST FAIL: effect-access invariant", file=sys.stderr)
            return 1
    print("WF-P source preflight self-test: PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="WF-P read-only source preflight")
    ap.add_argument("--project-root", default="~/Documents/abp_information_study")
    ap.add_argument("--out", default="~/Documents/abp_information_study/results/wfp_source_preflight")
    ap.add_argument("--wp2-run1", default=None, help="optional explicit authoritative wp2_run1_development50.py")
    ap.add_argument("--wp2-spec-json", default=None, help="optional explicit WP2_VALIDATION1000_FROZEN_SPEC.json")
    ap.add_argument("--future-bank-dir", default=None, help="optional explicit future-bank directory")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    result = run_preflight(
        project_root=Path(args.project_root),
        outdir=Path(args.out),
        wp2_run1_arg=args.wp2_run1,
        wp2_spec_arg=args.wp2_spec_json,
        future_bank_arg=args.future_bank_dir,
        pin_authoritative_hashes=True,
    )
    print(result["decision"])
    print("Waveform arrays opened: NO")
    print("Scientific effects opened: NO")
    print("Scientific effect analysis authorized: NO")
    print("Cohort/split freeze authorized:", "YES" if result["cohort_split_freeze_authorized"] else "NO")
    print("Output:", expanded(args.out))
    return 0 if result["decision"] != "WFP_SOURCE_PREFLIGHT_FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
