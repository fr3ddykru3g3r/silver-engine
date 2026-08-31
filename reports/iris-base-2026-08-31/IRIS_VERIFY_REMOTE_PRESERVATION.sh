#!/usr/bin/env bash
set -euo pipefail

# Verify the local IRIS package and the remote GitHub preservation copy.
# This script is intentionally verification-only: it never deletes files.

ROOT="${IRIS_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
AUDIT="$ROOT/IRIS_ARCHIVE_AUDIT_2026-09-01.json"
REPORT="$ROOT/IRIS_BASE_REPORT_PACKAGE_2026-08-31.zip"
FULL="$ROOT/IRIS_Colab_FULL_WITH_ACQUIRED_DATA_2026-08-31.zip"
PDF="$ROOT/iris_report/report.pdf"
STAGING="${IRIS_STAGING_ROOT:-/private/tmp/silver-engine-repo}"
REPO="fr3ddykru3g3r/silver-engine"
BRANCH="codex/iris-base-report-2026-08-31"
RELEASE="iris-2026-08-31"
ASSET="$(jq -r '.full_bundle.zip' "$AUDIT")"

for command_name in jq shasum unzip git gh; do
  command -v "$command_name" >/dev/null || {
    printf 'VERIFY_FAILED: missing command %s\n' "$command_name" >&2
    exit 1
  }
done

bytes() {
  wc -c < "$1" | tr -d '[:space:]'
}

sha256() {
  shasum -a 256 "$1" | awk '{print $1}'
}

expect_equal() {
  local label="$1" actual="$2" expected="$3"
  if [[ "$actual" != "$expected" ]]; then
    printf 'VERIFY_FAILED: %s (actual=%s expected=%s)\n' "$label" "$actual" "$expected" >&2
    exit 1
  fi
  printf 'OK: %s = %s\n' "$label" "$actual"
}

[[ -f "$AUDIT" && -f "$REPORT" && -f "$FULL" && -f "$PDF" ]] || {
  printf 'VERIFY_FAILED: required local artifact is missing\n' >&2
  exit 1
}

expected_pdf_bytes="$(jq -r '.report.pdf_bytes' "$AUDIT")"
expected_pdf_sha="$(jq -r '.report.pdf_sha256' "$AUDIT")"
expected_report_bytes="$(jq -r '.report.report_zip_bytes' "$AUDIT")"
expected_report_sha="$(jq -r '.report.report_zip_sha256' "$AUDIT")"
expected_full_bytes="$(jq -r '.full_bundle.zip_bytes' "$AUDIT")"
expected_full_sha="$(jq -r '.full_bundle.zip_sha256' "$AUDIT")"
expected_entries="$(jq -r '.full_bundle.zip_entries' "$AUDIT")"
expected_fits="$(jq -r '.full_bundle.verified_fits_entries' "$AUDIT")"

expect_equal 'report PDF bytes' "$(bytes "$PDF")" "$expected_pdf_bytes"
expect_equal 'report PDF SHA-256' "$(sha256 "$PDF")" "$expected_pdf_sha"
expect_equal 'report ZIP bytes' "$(bytes "$REPORT")" "$expected_report_bytes"
expect_equal 'report ZIP SHA-256' "$(sha256 "$REPORT")" "$expected_report_sha"
expect_equal 'full ZIP bytes' "$(bytes "$FULL")" "$expected_full_bytes"
expect_equal 'full ZIP SHA-256' "$(sha256 "$FULL")" "$expected_full_sha"

unzip -tq "$REPORT"
unzip -tq "$FULL"
expect_equal 'full ZIP entry count' "$(unzip -Z1 "$FULL" | wc -l | tr -d '[:space:]')" "$expected_entries"
expect_equal 'verified FITS entry count' "$(unzip -Z1 "$FULL" | grep -E '^fits/[^/]+\.fits$' | wc -l | tr -d '[:space:]')" "$expected_fits"

local_commit="$(git -C "$STAGING" rev-parse HEAD)"
expected_commit="$(jq -r '.publication_staging.local_commit' "$AUDIT")"
expect_equal 'local publication commit' "$local_commit" "$expected_commit"
[[ -z "$(git -C "$STAGING" status --porcelain=v1 --untracked-files=all)" ]] || {
  printf 'VERIFY_FAILED: publication staging tree is dirty\n' >&2
  exit 1
}

branch_json="$(gh api "repos/$REPO/branches/$BRANCH")"
remote_commit="$(jq -r '.commit.sha' <<<"$branch_json")"
expect_equal 'remote publication commit' "$remote_commit" "$local_commit"

release_json="$(gh api "repos/$REPO/releases/tags/$RELEASE")"
asset_json="$(jq -c --arg name "$ASSET" '.assets[] | select(.name == $name) | {state,size,digest}' <<<"$release_json" | head -n 1)"
[[ -n "$asset_json" ]] || {
  printf 'VERIFY_FAILED: release asset %s not found\n' "$ASSET" >&2
  exit 1
}
expect_equal 'remote asset state' "$(jq -r '.state' <<<"$asset_json")" 'uploaded'
expect_equal 'remote asset bytes' "$(jq -r '.size' <<<"$asset_json")" "$expected_full_bytes"
remote_digest="$(jq -r '.digest // empty' <<<"$asset_json")"
[[ "$remote_digest" == sha256:* ]] || {
  printf 'VERIFY_FAILED: GitHub did not expose a SHA-256 asset digest\n' >&2
  exit 1
}
expect_equal 'remote asset SHA-256' "${remote_digest#sha256:}" "$expected_full_sha"

printf 'REMOTE_PRESERVATION_VERIFIED=1\n'
printf 'No deletion was performed.\n'
