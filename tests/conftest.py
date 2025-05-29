import pytest

# No shared fixtures are strictly necessary for the current set of tests,
# as dummy_wav_file and dummy_mp4_path are used by specific test files.
# However, this file makes `tests/` a pytest-discoverable test package root.

# If we had more shared fixtures, they would go here.
# For example, a more complex setup for a dummy video file if it needed specific metadata
# or if multiple test files needed the same kind of dummy data.

# Example of a shared fixture (not currently used by the tests above, but for illustration):
# @pytest.fixture(scope="session")
# def shared_resource():
#     print("\nSetting up shared_resource (session scope)")
#     yield "A shared resource"
#     print("\nTearing down shared_resource (session scope)")

# To ensure pytest-mock's 'mocker' fixture is available,
# it's good practice to note that pytest-mock should be in the testing requirements.
# (This is usually handled by adding it to requirements-dev.txt or similar)

# This file can also be used for global pytest hooks if needed.
# For now, its main purpose is to make `tests` a test package.
# and to house any future shared fixtures.

log_messages_store = {}

def pytest_runtest_setup(item):
    """Clear stored log messages before each test if needed or initialize."""
    log_messages_store[item.nodeid] = []

# This is a more advanced hook, not strictly needed for basic rich logging capture,
# but could be used if we wanted to intercept logs globally for all tests.
# Rich's handler usually captures output well for failed tests.
# However, if direct access to log records during tests is needed:
#
# import logging
#
# class CaptureHandler(logging.Handler):
#     def __init__(self, item_nodeid):
#         super().__init__()
#         self.item_nodeid = item_nodeid
#         if self.item_nodeid not in log_messages_store:
#             log_messages_store[self.item_nodeid] = []
#
#     def emit(self, record):
#         log_messages_store[self.item_nodeid].append(self.format(record))
#
# def pytest_runtest_setup(item):
#     # Assuming 'app.services.services.clipExtractor.log' is the target logger
#     logger_to_capture = logging.getLogger('app.services.services.clipExtractor.log') # Or just 'rich' if that's the main one
#     capture_handler = CaptureHandler(item.nodeid)
#     # Adjust formatter if needed, e.g., record.getMessage() for raw message
#     # capture_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s")) 
#     logger_to_capture.addHandler(capture_handler)
#     item.capture_handler = capture_handler # Store on item to remove later
#
# def pytest_runtest_teardown(item):
#     if hasattr(item, 'capture_handler'):
#         logger_to_capture = logging.getLogger('app.services.services.clipExtractor.log') # Or 'rich'
#         logger_to_capture.removeHandler(item.capture_handler)

# The above hooks are examples and might need refinement based on exact logging setup and needs.
# For the current tests using mocker.patch on 'app.services.services.clipExtractor.log',
# those mocks handle log assertion directly, so these global hooks are not strictly necessary.

# If 'pytest.ähnlich' was a real thing, it would be a custom matcher.
# Example of how one might define a custom matcher for use with mock.assert_called_with:
# class RegexMatcher:
#     def __init__(self, pattern):
#         self.pattern = pattern
#
#     def __eq__(self, other):
#         import re
#         return bool(re.search(self.pattern, str(other)))
#
#     def __repr__(self):
#         return f"RegexMatcher({self.pattern!r})"
#
# # Then in tests:
# # from conftest import RegexMatcher
# # mock_logger.info.assert_any_call(RegexMatcher(r"Pipeline finished .*"))
#
# # For now, the tests use basic string checks or rely on `assert_any_call` flexibility.
# # The `pytest.ähnlich` in test_end_to_end.py was a placeholder for such a concept.
# # Actual solution in that test was simplified to `any("Pipeline finished" in msg for msg in logs)`.
