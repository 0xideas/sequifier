import os

# ONNX Runtime 1.29's POSIX telemetry can crash during process shutdown.
# This must be set before onnxruntime is imported and initializes its telemetry.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
