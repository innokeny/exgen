#!/usr/bin/env bash
# Functional + contract testing of the SAYIT generator service.
#
# Two layers:
#   1. pytest smoke tests (test_smoke.py) — happy paths and explicit error cases
#      that map directly onto §3.4 Table 17 (ФТ-01 .. ФТ-14).
#   2. Schemathesis property-based testing — generates thousands of inputs
#      from the OpenAPI schema and verifies every response conforms to the
#      schema and never returns a 5xx.
#
# Both layers emit JUnit XML so that a CI pipeline (or the thesis appendix)
# can reference structured results.
#
# Prerequisite: the service is running and /health is OK.

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p reports

URL=${SAYIT_GENERATOR_URL:-http://localhost:8000}

echo "[1/2] pytest smoke tests..."
SAYIT_GENERATOR_URL="$URL" pytest -v \
  --junit-xml=reports/pytest-junit.xml \
  --tb=short \
  test_smoke.py

echo
echo "[2/2] Schemathesis contract testing (property-based fuzzing)..."
schemathesis run "$URL/openapi.json" \
  --config-file schemathesis.toml

echo
echo "All functional artefacts in $(pwd)/reports/"
