FROM python:3.12-slim

WORKDIR /app

# Dependencies first so edits to app code don't invalidate the layer.
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Caches (summaries, match contexts, headshots) live here. Mount a volume at
# this path in production — otherwise every deploy throws away paid analyses
# and re-fetches provider data that counts against the daily quota.
RUN mkdir -p /app/.cache
VOLUME ["/app/.cache"]

ENV PORT=8000
EXPOSE 8000

# --proxy-headers so request.client is the real caller behind a load balancer;
# the rate limiter keys on it. FORWARDED_ALLOW_IPS must name the platform's
# proxy and never "*": uvicorn rewrites request.client from X-Forwarded-For for
# any peer it trusts, so "*" lets a caller supply their own address and mint a
# fresh rate-limit identity on every request. The default here trusts only the
# loopback peer. Single worker keeps the in-process rate-limit counters
# authoritative — see LAUNCH.md before scaling out.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips=${FORWARDED_ALLOW_IPS:-127.0.0.1}"]
