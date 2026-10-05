#!/usr/bin/env bash
# Run every Mojo test file under test/ against the TestSuite runner.
# Mojo removed the `mojo test` subcommand, so each file has a `main()` that
# calls TestSuite.discover_tests(...). We add `-I src` so the kdigest
# package resolves on the import path.
set -euo pipefail

fail=0
for f in test/test_*.mojo; do
    echo "=== ${f} ==="
    if ! mojo run -I src "${f}"; then
        fail=1
    fi
done
exit "${fail}"
