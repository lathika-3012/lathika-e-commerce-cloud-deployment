"""Test configuration: keep tests offline, deterministic and free of side effects."""
import os

os.environ["TELEMETRY_PERSIST"] = "false"
os.environ["LLM_PROVIDER"] = "mock"
