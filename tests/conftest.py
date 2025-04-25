import pytest
import os


def pytest_addoption(parser):
    parser.addoption(
        "--high-tol", action="store_true", help="Use high tolerance for certain tests."
    )
    parser.addoption("--all", action="store_true", help="Run all tests.")
    parser.addoption("--local-only", action="store_true", help="Just run local-only test cases.")
    parser.addoption(
        "--action",
        default="all",
        help="Select the action to run. Can be: "
             "\"generate\" - just generate outputs and store them; "
             "\"compare\" - just run comparison with reference data; "
             "\"all\" - run everything.",
    )
    parser.addoption(
        "--save-timing", action="store_true", help="Enable function timing and save results."
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "local: mark test as local-only")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--all"):
        return
    skip_local = pytest.mark.skip(reason="need --all option to run")
    skip_non_local = pytest.mark.skip(reason="skipped due to --local-only flag")
    for item in items:
        if "local" in item.keywords and not config.getoption("--local-only"):
            item.add_marker(skip_local)
        elif "local" not in item.keywords and config.getoption("--local-only"):
            item.add_marker(skip_non_local)
