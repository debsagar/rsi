# Sandbox for two things: the harness process (rsi/runner.py, stdlib only) and
# the `python` tool. Both are started with --network none and no environment
# keys, so nothing here needs or gets credentials.
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy pandas scipy scikit-learn statsmodels
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
