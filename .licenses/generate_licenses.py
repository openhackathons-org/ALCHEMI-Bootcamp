#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the installed-package license inventory for the locked runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import sysconfig
import tomllib
from importlib.metadata import Distribution, distributions
from pathlib import Path
from typing import Any
from urllib.parse import quote

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

UNKNOWN_VALUES = {"", "UNKNOWN", "UNKNOWN LICENSE"}

CLASSIFIER_LICENSES = {
    "License :: OSI Approved :: Apache Software License": "Apache-2.0",
    "License :: OSI Approved :: GNU General Public License (GPL)": "GPL",
    "License :: OSI Approved :: ISC License (ISCL)": "ISC",
    "License :: OSI Approved :: MIT License": "MIT",
    "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "License :: OSI Approved :: Python Software Foundation License": "PSF-2.0",
    "License :: Other/Proprietary License": "LicenseRef-Proprietary",
    "License :: Public Domain": "LicenseRef-Public-Domain",
}

SHORT_LICENSES = {
    "3-Clause BSD License": "BSD-3-Clause",
    "Apache 2.0": "Apache-2.0",
    "Apache License 2.0": "Apache-2.0",
    "Apache Software License": "Apache-2.0",
    "BSD 2-Clause License": "BSD-2-Clause",
    "BSD 3-Clause License": "BSD-3-Clause",
    "GPL2+": "GPL-2.0-or-later",
    "ISC license": "ISC",
    "MIT License": "MIT",
    "MIT license": "MIT",
    "Modified BSD License": "BSD-3-Clause",
    "MPL 2.0": "MPL-2.0",
    "NVIDIA Proprietary Software": "LicenseRef-NVIDIA-Proprietary",
    "PSFL": "PSF-2.0",
}

REQUIRED_REVIEW_LICENSES = {
    "ase": "LGPL-2.1-or-later",
    "cuda-toolkit": "NVIDIA CUDA Toolkit EULA (proprietary)",
    "matscipy": "LGPL-2.1-or-later",
    "python-hostlist": "GPL-2.0-or-later",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def declared_dependencies(pyproject_path: Path) -> dict[str, dict[str, Any]]:
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    sections: list[tuple[str, str, list[str]]] = [
        (
            "runtime",
            "pyproject.toml [project.dependencies]",
            payload.get("project", {}).get("dependencies", []),
        )
    ]
    for group, requirements in payload.get("dependency-groups", {}).items():
        sections.append(
            (
                group,
                f"pyproject.toml [dependency-groups.{group}]",
                requirements,
            )
        )

    declared: dict[str, dict[str, Any]] = {}
    order = 0
    for group, source, requirements in sections:
        for declaration in requirements:
            requirement = Requirement(declaration)
            normalized = canonicalize_name(requirement.name)
            if normalized in declared:
                raise SystemExit(
                    f"Direct dependency is declared more than once: {requirement.name}"
                )
            declared[normalized] = {
                "name": requirement.name,
                "declaration": declaration,
                "declared_in": source,
                "dependency_group": group,
                "order": order,
            }
            order += 1
    return declared


def load_overrides(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        canonicalize_name(name): value
        for name, value in payload.get("overrides", {}).items()
    }


def license_paths(distribution: Distribution) -> list[tuple[Path, Path]]:
    found: dict[str, tuple[Path, Path]] = {}
    for relative in distribution.files or []:
        relative_path = Path(str(relative))
        basename = relative_path.name.upper()
        parts = {part.upper() for part in relative_path.parts}
        if "LICENSES" not in parts and not basename.startswith(
            ("LICENSE", "COPYING", "NOTICE")
        ):
            continue
        located = Path(distribution.locate_file(relative))
        if located.is_file():
            found[str(relative_path)] = (relative_path, located)
    return [found[key] for key in sorted(found)]


def environment_path(path: Path, environment_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(environment_root.resolve())
    except (OSError, ValueError):
        return str(path)
    return f"<inventory-environment>/{relative.as_posix()}"


def read_license_text(
    distribution: Distribution,
    metadata_license: str,
    environment_root: Path,
) -> tuple[str, str, str]:
    paths = license_paths(distribution)
    if paths:
        text = "\n\n".join(
            f"===== {relative.as_posix()} =====\n"
            + located.read_text(encoding="utf-8", errors="replace").strip()
            for relative, located in paths
        )
        files = ", ".join(
            environment_path(located, environment_root) for _relative, located in paths
        )
        return text, files, "bundled"
    if "\n" in metadata_license:
        return metadata_license, "Installed package metadata", "metadata"
    return "UNKNOWN", "UNKNOWN", "missing"


def bsd_identifier(license_text: str) -> str:
    normalized = " ".join(license_text.split()).lower()
    if "redistribution and use in source and binary forms" not in normalized:
        return "BSD License"
    if "neither the name" in normalized or "neither the names" in normalized:
        return "BSD-3-Clause"
    return "BSD-2-Clause"


def classifier_identifier(metadata: Any, license_text: str) -> str:
    classifiers = [
        value
        for value in metadata.get_all("Classifier") or []
        if value.startswith("License ::")
    ]
    identifiers: list[str] = []
    for classifier in classifiers:
        if classifier == "License :: OSI Approved :: BSD License":
            identifier = bsd_identifier(license_text)
        else:
            identifier = CLASSIFIER_LICENSES.get(classifier, "")
        if identifier and identifier not in identifiers:
            identifiers.append(identifier)
    if not identifiers:
        return "UNKNOWN"
    if len(identifiers) == 1:
        return identifiers[0]
    return " OR ".join(identifiers)


def license_identifier(metadata: Any, license_text: str) -> str:
    expression = (metadata.get("License-Expression") or "").strip()
    if expression:
        return expression

    metadata_license = (metadata.get("License") or "").strip()
    if metadata_license == "Dual License":
        return classifier_identifier(metadata, license_text)
    if metadata_license in {"BSD", "BSD License"}:
        return bsd_identifier(license_text)
    if metadata_license in SHORT_LICENSES:
        return SHORT_LICENSES[metadata_license]
    if (
        metadata_license
        and "\n" not in metadata_license
        and len(metadata_license) <= 100
        and metadata_license.upper() not in UNKNOWN_VALUES
    ):
        return metadata_license
    return classifier_identifier(metadata, license_text)


def distribution_url(distribution: Distribution) -> str:
    for relative in distribution.files or []:
        if relative.name != "direct_url.json":
            continue
        path = Path(distribution.locate_file(relative))
        if not path.is_file():
            continue
        direct = json.loads(path.read_text(encoding="utf-8"))
        url = direct.get("url", "")
        vcs = direct.get("vcs_info", {})
        if url and vcs.get("vcs") and vcs.get("commit_id"):
            return f"{vcs['vcs']}+{url}@{vcs['commit_id']}"
        if url:
            return url

    metadata = distribution.metadata
    project_urls = []
    for value in metadata.get_all("Project-URL") or []:
        label, separator, url = value.partition(",")
        if separator and url.strip():
            project_urls.append((label.strip().lower(), url.strip()))
    for preferred_label in ("repository", "source", "homepage", "documentation"):
        for label, url in project_urls:
            if label == preferred_label:
                return url
    homepage = (metadata.get("Home-page") or "").strip()
    if homepage:
        return homepage
    if project_urls:
        return project_urls[0][1]
    name = quote(distribution.metadata["Name"], safe="")
    version = quote(distribution.version, safe="")
    return f"https://pypi.org/project/{name}/{version}/"


def build_records(
    repo_root: Path,
    declared: dict[str, dict[str, Any]],
    overrides: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    environment_root = Path(sys.prefix).resolve()
    records: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}

    installed = sorted(
        distributions(),
        key=lambda item: canonicalize_name(item.metadata["Name"]),
    )
    for distribution in installed:
        metadata = distribution.metadata
        name = metadata["Name"]
        normalized = canonicalize_name(name)
        if normalized in by_name:
            raise SystemExit(f"Duplicate installed distribution: {name}")
        metadata_license = (metadata.get("License") or "").strip()
        license_text, license_file, text_status = read_license_text(
            distribution,
            metadata_license,
            environment_root,
        )
        direct = declared.get(normalized)
        record = {
            "Declaration": direct["declaration"] if direct else "",
            "DeclaredIn": direct["declared_in"] if direct else "",
            "DependencyGroup": direct["dependency_group"] if direct else "",
            "Direct": direct is not None,
            "License": license_identifier(metadata, license_text),
            "LicenseFile": license_file,
            "LicenseText": license_text,
            "LicenseTextStatus": text_status,
            "Name": name,
            "URL": distribution_url(distribution),
            "Version": distribution.version,
            "_direct_order": direct["order"] if direct else None,
        }
        records.append(record)
        by_name[normalized] = record

    applied: list[str] = []
    for normalized, override in overrides.items():
        record = by_name.get(normalized)
        if record is None:
            continue
        expected_version = override.get("version")
        if expected_version and record["Version"] != expected_version:
            raise SystemExit(
                f"License override for {record['Name']} expects {expected_version}, "
                f"found {record['Version']}"
            )
        record["License"] = override["license"]
        if override.get("license_text"):
            record["LicenseText"] = override["license_text"]
            record["LicenseFile"] = "Manual text in license-overrides.json"
            record["LicenseTextStatus"] = "manual"
        elif override.get("license_text_file"):
            license_path = repo_root / override["license_text_file"]
            record["LicenseText"] = license_path.read_text(encoding="utf-8").strip()
            record["LicenseFile"] = override["license_text_file"]
            record["LicenseTextStatus"] = "manual"
        elif record["LicenseText"].strip().upper() in UNKNOWN_VALUES:
            record["LicenseText"] = override.get("evidence", override["license"])
            record["LicenseFile"] = override.get("license_text_url", "Official source")
            record["LicenseTextStatus"] = "reference-only"
        record["LicenseEvidence"] = override.get("evidence", "")
        applied.append(record["Name"])

    missing_direct = sorted(set(declared) - set(by_name))
    if missing_direct:
        names = ", ".join(declared[name]["name"] for name in missing_direct)
        raise SystemExit(f"Declared dependencies missing from inventory: {names}")

    for normalized, expected in REQUIRED_REVIEW_LICENSES.items():
        record = by_name.get(normalized)
        if record is not None and record["License"] != expected:
            raise SystemExit(
                f"Review-required license mismatch for {record['Name']}: "
                f"expected {expected}, found {record['License']}"
            )

    records.sort(
        key=lambda record: (
            not record["Direct"],
            record["_direct_order"]
            if record["_direct_order"] is not None
            else sys.maxsize,
            canonicalize_name(record["Name"]),
        )
    )
    for record in records:
        record.pop("_direct_order")
    return records, applied


def cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.replace("|", r"\|").split())


def markdown_table(records: list[dict[str, Any]], direct_only: bool) -> str:
    if direct_only:
        columns = ["Group", "Name", "Version", "License", "Declaration", "URL"]
    else:
        columns = ["Scope", "Name", "Version", "License", "Declared in", "URL"]
    rows = []
    for record in records:
        if direct_only:
            row = [
                record["DependencyGroup"],
                record["Name"],
                record["Version"],
                record["License"],
                record["Declaration"],
                record["URL"],
            ]
        else:
            row = [
                "direct" if record["Direct"] else "transitive",
                record["Name"],
                record["Version"],
                record["License"],
                record["DeclaredIn"],
                record["URL"],
            ]
        rows.append([cell(value) for value in row])
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return f"{header}\n{divider}\n{body}\n"


def review_notes(records: list[dict[str, Any]], markdown: bool) -> str:
    by_name = {canonicalize_name(record["Name"]): record for record in records}
    code = (lambda value: f"`{value}`") if markdown else (lambda value: value)
    lines: list[str] = []

    ase = by_name.get("ase")
    if ase:
        lines.append(
            f"ASE {ase['Version']} is a direct dependency under {ase['License']}."
        )

    mace = by_name.get("mace-torch")
    matscipy = by_name.get("matscipy")
    hostlist = by_name.get("python-hostlist")
    if mace and matscipy and hostlist:
        lines.append(
            f"MACE {mace['Version']} is {mace['License']}. Its locked dependency "
            f"tree includes {code('matscipy')} {matscipy['Version']} under "
            f"{matscipy['License']} and {code('python-hostlist')} "
            f"{hostlist['Version']} under {hostlist['License']}."
        )

    cuda = by_name.get("cuda-toolkit")
    if cuda:
        eula = "https://docs.nvidia.com/cuda/eula/"
        linked_eula = (
            f"[NVIDIA CUDA Toolkit EULA]({eula})"
            if markdown
            else f"NVIDIA CUDA Toolkit EULA ({eula})"
        )
        lines.append(
            f"The CUDA packages installed through PyTorch include "
            f"{code('cuda-toolkit')} {cuda['Version']} and NVIDIA runtime wheels. "
            f"Their proprietary components are governed by the {linked_eula}; "
            "individual package records contain bundled terms when supplied."
        )

    notices = (
        "[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)"
        if markdown
        else "../THIRD_PARTY_NOTICES.md"
    )
    lines.append(
        "Toolkit builds the D3 parameter cache from the legacy Grimme reference "
        f"archive under GPL-1.0-or-later. The cache is a runtime asset covered by "
        f"{notices}; it is outside this installed-package inventory."
    )

    if markdown:
        return "\n".join(f"- {line}" for line in lines)
    return "\n".join(f"* {line}" for line in lines)


def snapshot_description(
    records: list[dict[str, Any]],
    lock_sha256: str,
    markdown: bool,
) -> str:
    direct_count = sum(record["Direct"] for record in records)
    lock_value = f"`{lock_sha256}`" if markdown else lock_sha256
    lines = [
        f"Source: pyproject.toml and uv.lock (SHA-256 {lock_value})",
        f"Environment: Python {platform.python_version()} on {sysconfig.get_platform()}",
        f"Packages: {len(records)} installed, {direct_count} direct",
    ]
    if markdown:
        return "\n".join(f"- {line}" for line in lines)
    return "\n".join(lines)


def write_outputs(
    repo_root: Path,
    out_dir: Path,
    records: list[dict[str, Any]],
) -> None:
    lock_sha256 = hashlib.sha256((repo_root / "uv.lock").read_bytes()).hexdigest()
    direct_records = [record for record in records if record["Direct"]]

    (out_dir / "details.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    direct_text = (
        "# Direct third-party dependencies\n\n"
        "Generated from the dependency declarations in `pyproject.toml` and the "
        "Linux environment synchronized from `uv.lock`. License texts and "
        "additional metadata are available in [details.json](details.json) and "
        "[Third_party_attr.txt](Third_party_attr.txt).\n\n"
        + snapshot_description(records, lock_sha256, markdown=True)
        + "\n\n## Terms requiring attention\n\n"
        + review_notes(records, markdown=True)
        + "\n\n## Declared packages\n\n"
        + markdown_table(direct_records, direct_only=True)
    )
    (out_dir / "direct-dependencies.md").write_text(direct_text, encoding="utf-8")

    summary_text = (
        "# Installed package license summary\n\n"
        "This generated snapshot includes direct and transitive Python packages "
        "installed for the locked Linux tutorial environment.\n\n"
        + snapshot_description(records, lock_sha256, markdown=True)
        + "\n\n## Terms requiring attention\n\n"
        + review_notes(records, markdown=True)
        + "\n\n## Installed packages\n\n"
        + markdown_table(records, direct_only=False)
    )
    (out_dir / "summary.md").write_text(summary_text, encoding="utf-8")

    header = (
        "Installed Python package third-party attributions\n"
        + ("=" * 51)
        + "\n\n"
        + snapshot_description(records, lock_sha256, markdown=False)
        + "\n\nTerms requiring attention\n"
        + "-" * 25
        + "\n"
        + review_notes(records, markdown=False)
        + "\n"
    )
    blocks = []
    for record in records:
        fields = [
            f"Name: {record['Name']}",
            f"Version: {record['Version']}",
            f"Scope: {'direct' if record['Direct'] else 'transitive'}",
            f"License: {record['License']}",
            f"URL: {record['URL']}",
            f"License file: {record['LicenseFile']}",
            f"License text status: {record['LicenseTextStatus']}",
        ]
        if record["DeclaredIn"]:
            fields.append(f"Declared in: {record['DeclaredIn']}")
            fields.append(f"Declaration: {record['Declaration']}")
        if record.get("LicenseEvidence"):
            fields.append(f"License evidence: {record['LicenseEvidence']}")
        fields.extend(["", record["LicenseText"]])
        blocks.append("\n".join(fields))

    separator = "\n\n" + ("-" * 79) + "\n\n"
    (out_dir / "Third_party_attr.txt").write_text(
        header + separator + separator.join(blocks) + "\n",
        encoding="utf-8",
    )


def validate_records(records: list[dict[str, Any]]) -> None:
    unknown = [
        record
        for record in records
        if record["License"].strip().upper() in UNKNOWN_VALUES
    ]
    missing_text = [
        record
        for record in records
        if record["LicenseText"].strip().upper() in UNKNOWN_VALUES
    ]
    if unknown:
        names = ", ".join(f"{record['Name']} {record['Version']}" for record in unknown)
        raise SystemExit(f"Unknown license identifiers: {names}")
    if missing_text:
        names = ", ".join(
            f"{record['Name']} {record['Version']}" for record in missing_text
        )
        raise SystemExit(f"Missing license text or reference: {names}")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    out_dir = args.out_dir.resolve()
    declared = declared_dependencies(repo_root / "pyproject.toml")
    overrides = load_overrides(out_dir / "license-overrides.json")
    records, applied = build_records(repo_root, declared, overrides)
    validate_records(records)
    write_outputs(repo_root, out_dir, records)
    direct_count = sum(record["Direct"] for record in records)
    print(
        f">>> done: {len(records)} installed packages; "
        f"{direct_count} direct; {len(applied)} overrides"
    )


if __name__ == "__main__":
    main()
