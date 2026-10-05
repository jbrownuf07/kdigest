"""Smoke test: verifies the package imports and the test runner works.

Run with:
    pixi run mojo run test/test_smoke.mojo
"""

from std.testing import assert_true, TestSuite


def test_package_imports() raises:
    # If this file compiles, the stream_fire package resolved on the import
    # path (configured via `-I src` in the pixi `test` task).
    assert_true(True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
