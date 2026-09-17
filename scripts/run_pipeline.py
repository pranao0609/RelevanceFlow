from __future__ import annotations

import sys
from pathlib import Path


from relevanceflow.data.pipeline_stages import (
    clean_wands_dataset,
    validate_raw_wands,
)
from relevanceflow.features.cache import (
    build_feature_fingerprint,
    build_manifest,
    is_cache_valid,
    save_manifest,
    validate_artifacts,
)
from relevanceflow.utils.config import load_configs
from relevanceflow.utils.pipeline import (
    build_pipeline_manifest,
    run_stage,
    utc_now,
    write_json,
)

# Define python_executable pointing to the active virtualenv interpreter
python_executable = sys.executable
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "wands"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "wands"

PIPELINE_DIR = PROJECT_ROOT / "experiments" / "pipeline"

MANIFEST_PATH = PIPELINE_DIR / "pipeline_manifest.json"
SUMMARY_PATH = PIPELINE_DIR / "run_summary.json"


def run_python_script(script_path: str) -> None:
    """Run a project Python script using the active interpreter."""

    run_stage(
        name=Path(script_path).stem,
        command=[
            sys.executable,
            script_path,
        ],
        project_root=PROJECT_ROOT,
    )


def main() -> int:
    """Execute the complete RelevanceFlow training pipeline."""

    print("=" * 70)
    print("RelevanceFlow — End-to-End Training Pipeline")
    print("=" * 70)

    config = load_configs()

    PIPELINE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Pipeline manifest
    # ---------------------------------------------------------------

    manifest = build_pipeline_manifest(
        project_root=PROJECT_ROOT,
        config=config,
    )

    write_json(
        MANIFEST_PATH,
        manifest,
    )

    print(f"\nPipeline manifest: {MANIFEST_PATH}")

    stage_results = []

    # ---------------------------------------------------------------
    # 1. VALIDATE
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("STAGE 1 — VALIDATE")
    print("=" * 70)

    validation_summary = validate_raw_wands(RAW_DIR)

    validation_report_path = PIPELINE_DIR / "validation_report.json"

    write_json(
        validation_report_path,
        validation_summary,
    )

    print(f"Valid: {validation_summary['valid']}")

    print(f"Passed checks: " f"{validation_summary['passed_checks']}")

    print(f"Warnings: " f"{validation_summary['warning_checks']}")

    print(f"Errors: " f"{validation_summary['failed_checks']}")

    # ---------------------------------------------------------------
    # Validation policy
    # ---------------------------------------------------------------
    #
    # Raw WANDS contains duplicate query-product annotations.
    # Conflicting labels are intentionally reported by the standalone
    # validator and are resolved by the cleaning stage.
    #
    # Therefore:
    # - structural validation errors stop the pipeline
    # - query-product label conflicts are allowed to proceed to cleaning
    # ---------------------------------------------------------------

    cleaning_resolvable_checks = {
        "query_product_pair_label_consistency",
    }

    blocking_errors = [
        check
        for check in validation_summary["checks"]
        if (
            check["status"] == "FAIL"
            and check["check"] not in cleaning_resolvable_checks
        )
    ]

    resolvable_errors = [
        check
        for check in validation_summary["checks"]
        if (check["status"] == "FAIL" and check["check"] in cleaning_resolvable_checks)
    ]

    if blocking_errors:
        print("\nValidation failed.")

        print("\nBlocking errors:")
        for error in blocking_errors:
            print(f"  - {error['check']}: {error['message']}")

        write_json(
            SUMMARY_PATH,
            {
                "status": "failed",
                "failed_stage": "validate",
                "timestamp": utc_now(),
                "blocking_errors": blocking_errors,
                "resolvable_errors": resolvable_errors,
            },
        )

        return 1

    if resolvable_errors:
        print(
            "\nValidation detected annotation conflicts "
            "that are explicitly handled by the cleaning stage."
        )

        for error in resolvable_errors:
            print(f"  [DEFERRED TO CLEANING] " f"{error['check']}: {error['message']}")

    print("\nRaw-data validation accepted for pipeline continuation.")

    stage_results.append(
        {
            "stage": "validate",
            "status": "success",
            "validation_valid": validation_summary["valid"],
            "blocking_errors": len(blocking_errors),
            "deferred_errors": len(resolvable_errors),
        }
    )

    stage_results.append(
        {
            "stage": "validate",
            "status": "success",
        }
    )

    # ---------------------------------------------------------------
    # 2. CLEAN
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("STAGE 2 — CLEAN")
    print("=" * 70)

    cleaning_summary = clean_wands_dataset(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
    )

    print(f"Products: {cleaning_summary['products']:,}")

    print(f"Queries: {cleaning_summary['queries']:,}")

    print(f"Judgments: {cleaning_summary['judgments']:,}")

    print(f"Conflicts: {cleaning_summary['conflicts']:,}")

    stage_results.append(
        {
            "stage": "clean",
            "status": "success",
            **cleaning_summary,
        }
    )

    # ---------------------------------------------------------------
    # 3. SPLIT
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("STAGE 3 — SPLIT")
    print("=" * 70)

    run_python_script("scripts/create_split.py")

    stage_results.append(
        {
            "stage": "split",
            "status": "success",
        }
    )

    # ---------------------------------------------------------------
    # 4. FEATURES
    # ---------------------------------------------------------------
    print()
    print("=" * 70)
    print("STAGE 4 — FEATURE GENERATION")
    print("=" * 70)

    cache_config = config["features_cache"]["features_cache"]

    if not cache_config.get("enabled", True):
        print()
        print("Feature caching disabled.")
        run_stage(
            "generate_features",
            [
                python_executable,
                "scripts/generate_features.py",
            ],
            PROJECT_ROOT,
        )
    else:
        manifest_path = PROJECT_ROOT / cache_config["manifest_path"]

        artifact_paths = {
            name: PROJECT_ROOT / path
            for name, path in cache_config["artifacts"].items()
        }

        processed_data_paths = {
            "products": (PROJECT_ROOT / "data/processed/wands/products.parquet"),
            "queries": (PROJECT_ROOT / "data/processed/wands/queries.parquet"),
            "judgments": (PROJECT_ROOT / "data/processed/wands/judgments.parquet"),
        }

        split_paths = {
            "train": (PROJECT_ROOT / "data/processed/wands/train.parquet"),
            "validation": (PROJECT_ROOT / "data/processed/wands/validation.parquet"),
            "test": (PROJECT_ROOT / "data/processed/wands/test.parquet"),
        }

        feature_config = config.get("features", {})
        retrieval_config = config.get("retrieval", {})

        source_paths = [
            PROJECT_ROOT / "scripts/generate_features.py",
            PROJECT_ROOT / "src/relevanceflow/features/pipeline.py",
            PROJECT_ROOT / "src/relevanceflow/features/lexical.py",
            PROJECT_ROOT / "src/relevanceflow/features/text.py",
            PROJECT_ROOT / "src/relevanceflow/features/metadata.py",
            PROJECT_ROOT / "src/relevanceflow/features/compatibility.py",
            PROJECT_ROOT / "src/relevanceflow/features/semantic.py",
        ]

        fingerprint, fingerprint_payload = build_feature_fingerprint(
            project_root=PROJECT_ROOT,
            processed_data_paths=processed_data_paths,
            split_paths=split_paths,
            feature_config=feature_config,
            retrieval_config=retrieval_config,
            source_paths=source_paths,
        )

        cache_valid, cache_reason = is_cache_valid(
            project_root=PROJECT_ROOT,
            manifest_path=manifest_path,
            expected_fingerprint=fingerprint,
            artifact_paths=artifact_paths,
        )

        print()

        if cache_valid:
            print("FEATURE CACHE HIT")
            print("Reusing existing feature artifacts.")
            print()
            print(f"Fingerprint: {fingerprint}")
            print(f"Manifest:    {manifest_path}")

            for name, path in artifact_paths.items():
                print(f"  {name:12s} {path}")

        else:
            print(f"FEATURE CACHE MISS: {cache_reason}")
            print("Generating feature artifacts...")

            run_stage(
                "generate_features",
                [
                    python_executable,
                    "scripts/generate_features.py",
                ],
                PROJECT_ROOT,
            )

            artifacts_valid, artifact_metadata = validate_artifacts(
                project_root=PROJECT_ROOT,
                artifact_paths=artifact_paths,
            )

            if not artifacts_valid:
                raise RuntimeError(
                    "Feature generation completed, but one or more "
                    "feature artifacts are missing or empty."
                )

            manifest = build_manifest(
                fingerprint=fingerprint,
                fingerprint_payload=fingerprint_payload,
                artifact_metadata=artifact_metadata,
                source="pipeline_generated",
            )

            save_manifest(
                manifest_path,
                manifest,
            )

            print()
            print("Feature cache manifest updated.")
            print(f"Manifest: {manifest_path}")

        # ---------------------------------------------------------------
        # 5. TRAIN + EVALUATE + REGISTER
        # ---------------------------------------------------------------

        print("\n" + "=" * 70)
    print("STAGE 5 — TRAIN / EVALUATE / REGISTER")
    print("=" * 70)

    run_python_script("scripts/train_ltr.py")

    stage_results.append(
        {
            "stage": "train",
            "status": "success",
        }
    )

    stage_results.append(
        {
            "stage": "evaluate",
            "status": "success",
        }
    )

    stage_results.append(
        {
            "stage": "register",
            "status": "success",
        }
    )

    # ---------------------------------------------------------------
    # Pipeline summary
    # ---------------------------------------------------------------

    summary = {
        "status": "success",
        "timestamp": utc_now(),
        "stages": stage_results,
        "manifest": str(MANIFEST_PATH),
        "validation_report": str(validation_report_path),
    }

    write_json(
        SUMMARY_PATH,
        summary,
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(f"\nSummary: {SUMMARY_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
