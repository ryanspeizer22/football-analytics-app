# FORWARDED_ALLOW_IPS must be the platform's proxy address, never "*".
# Uvicorn rewrites request.client from X-Forwarded-For for any peer it trusts,
# and the per-client rate limit keys on request.client — so trusting "*" lets a
# caller mint a new identity per request and bypass the cap entirely. Verified:
# with "*", a spoofed header set client.host to an arbitrary address; with a
# restricted range it stayed the real peer. Default below trusts only the
# loopback peer, which is safe but shares one bucket across users if the proxy
# is not on loopback — set the real range on your host.
web: uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips=${FORWARDED_ALLOW_IPS:-127.0.0.1}
