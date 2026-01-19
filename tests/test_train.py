"""
tests/test_train.py

REQ: Verify that the package can be imported.
WHY: Minimal correctness requirement.


WHY: Each stage should be able to be run independently on any platform.
"""

import importlib
import sys

PACKAGE_NAME = "toy_gpt_train"  # CUSTOM: Use package name.

# Modules that can run with no arguments
DEMO_MODULES_NO_ARGS = [
    "a_tokenizer",
    "b_vocab",
    "c_model",
    "d_train",
]

# Modules that require CLI arguments (tested separately)
DEMO_MODULES_WITH_ARGS = [
    "e_infer",
]


def test_package_imports() -> None:
    """Test that the package itself can be imported."""
    module = importlib.import_module(PACKAGE_NAME)
    assert module is not None


def test_demo_modules_run() -> None:
    """
    Test that each demo module can be imported and run.

    OBS: This test does not validate outputs.
         It only checks that demo entry points execute without error.
    """
    for module_name in DEMO_MODULES_NO_ARGS:
        module_path = f"{PACKAGE_NAME}.{module_name}"
        module = importlib.import_module(module_path)

        if hasattr(module, "main"):
            # Temporarily replace sys.argv so argparse doesn't see pytest args
            original_argv = sys.argv
            sys.argv = [module_name]
            try:
                module.main()
            finally:
                sys.argv = original_argv


def test_demo_modules_with_args_import() -> None:
    """
    Test that modules requiring CLI arguments can at least be imported.

    OBS: These modules need specific arguments to run main().
         We verify import works; full execution requires integration tests.
    """
    for module_name in DEMO_MODULES_WITH_ARGS:
        module_path = f"{PACKAGE_NAME}.{module_name}"
        module = importlib.import_module(module_path)
        assert module is not None
