# PitchSense — production launch checklist

Five phases, ordered so each one unblocks the next. Estimates assume one
person working through them.

---

## 1. Hosting & custom domain

**Do not deploy this to Vercel.** It is the wrong shape for this app, for
three reasons that will each bite on day one:

- Serverless functions cap request duration well below what a cold match
  analysis needs. Generation measured 20–80s; the narrative pass alone ran 80s
  on the Euro final. Those requests would be killed mid-flight.
- There is no persistent filesystem. `.cache/` holds summaries, contexts,
  crests, headshots, venue shots and OG cards. On Vercel each invocation starts
  empty, so every request re-pays the provider and the model.
- Pillow renders the OG cards at request time and needs real fonts. The
  Dockerfile installs `fonts-dejavu-core` precisely because a container without
  them silently falls back to a bitmap font.

**Use Railway or Render.** Both run the existing `Dockerfile` unchanged, allow
long requests, and offer a mountable volume.

```
Service       Docker, from the repo Dockerfile
Volume        mount at /app/.cache   (10GB is generous; today it is ~200MB)
Health check  GET /health
Start         already in the Dockerfile (--proxy-headers is set)
```

`--proxy-headers` is already configured, which matters: `_client_id()`
deliberately ignores `X-Forwarded-For` unless the proxy is trusted, and the
per-client rate limit depends on `request.client` being the real caller. Also
set `--forwarded-allow-ips` to the platform's proxy range.

**Domain.** Point `pitchsense.com` at the service, add `www` as a redirect, and
set `PITCHSENSE_BASE_URL=https://pitchsense.com`. That variable is not
cosmetic — canonical links, the sitemap and every OpenGraph image URL are built
from it, and crawlers will not resolve relative paths. Getting it wrong means
every shared link unfurls against `127.0.0.1`.

- [ ] Deploy from Dockerfile, volume mounted at `/app/.cache`
- [ ] `PITCHSENSE_BASE_URL` set to the real origin
- [ ] DNS + TLS, `www` → apex redirect
- [ ] Confirm `/og/match/979139.png` renders with real fonts in the container
- [ ] Confirm `/health` reports both quotas

---

## 2. Environment variables & key lockdown

Two secrets: `ANTHROPIC_API_KEY` and `FOOTBALL_API_KEY`. Both are read from the
environment at import time via `load_dotenv()`.

- [ ] Set both in the platform's secret store. Never in the image, never in the
      repo. `.env` is already gitignored, and history was checked clean.
- [ ] **Rotate both before going public.** They have been used in local
      development and pasted through terminals; treat them as burned.
- [ ] Restrict the Anthropic key to the minimum scope available, and set a
      hard monthly spend cap in the console. This is the single largest cost
      risk in the app — one uncached match is a real model call.
- [ ] Set `ADMIN_TOKEN` to a long random value. It is what grants founder
      access from a browser that is not on the server, and without it set,
      token auth is disabled entirely (which is the safe default). Never commit
      it; rotate it like any other secret.
- [ ] Confirm `PITCHSENSE_BASE_URL` is set to the real domain. It doubles as
      the switch that disables the loopback rate-limit bypass — without it, a
      reverse proxy on the same host makes every request look local.
      `/health` reports `local_bypass` and `admin_token_configured`; check both
      after the first deploy.
- [ ] Review the rate limiter for public traffic. `services/rate_limit.py`
      currently enforces a per-client sliding window, a global daily cap and a
      concurrency semaphore, counting only cache misses. The daily cap of 150
      is sized for one developer, not for launch — decide the number you are
      willing to pay for and set it deliberately.
- [ ] Consider putting the paid generation routes behind a soft gate (session
      or turnstile). Every `/api/match/{id}/narrative` miss costs money, and
      nothing currently stops a script from walking fixture ids.

---

## 3. Caching resilience (Redis / KV)

`services/summary_cache.py` is memory + on-disk JSON. That is correct for a
single instance and breaks the moment there are two: instance A generates an
analysis, instance B has never heard of it and pays to generate it again.

The abstraction is already the right shape — `get(key)` / `set(key, value)` —
so this is a backend swap, not a rewrite.

- [ ] Add Redis (Railway and Render both offer managed instances).
- [ ] Implement a Redis backend behind the existing `get`/`set`, with the disk
      cache as the fallback when `REDIS_URL` is unset, so local development
      keeps working with no Redis running.
- [ ] Keep the **media** proxies on disk. Crests, headshots and venue shots are
      binary, large, and immutable — the volume is the right home for them, not
      Redis.
- [ ] Set TTLs by kind, not globally. They have genuinely different lifetimes:
      finished-match analyses never change; upcoming fixture lists and the
      pre-match dossier go stale at kick-off; the leaderboard pool changes
      weekly.
- [ ] Add a cache-warm job for the eight heritage fixtures so the homepage
      never opens onto a cold generation.

---

## 4. Legal & compliance

Three third parties are involved, and the policy has to name all of them:

- **API-Football** supplies all match data, crests, competition logos and
  venue photography. Check the subscription terms for an attribution
  requirement; the footer already credits them.
- **Anthropic** receives match data and returns the analysis.
- **YouTube** — this one is easy to miss. The highlights panel embeds video.
  The player is a click-to-load facade, so nothing loads from YouTube until a
  viewer presses play, but **once they do, YouTube sets cookies.** That must be
  disclosed, and in the EU/UK it needs consent before the iframe mounts.

- [ ] `/terms` and `/privacy` pages, served through `_shell()` so they inherit
      the layout and metadata.
- [ ] Privacy policy covering: what is stored (currently nothing user-specific
      — no accounts, no personal data), the three processors above, and the
      YouTube cookie behaviour on play.
- [ ] Cookie/consent notice if you take EU or UK traffic. The facade helps —
      it means you only need consent at the point of play, not on page load.
- [ ] Footer links to both, plus the existing photography credit.
- [ ] Add both to `sitemap.xml` (it currently lists only `/`).
- [ ] State plainly that analysis is model-generated. The Match State Index
      already carries its methodology note; the same honesty belongs in the
      terms.

---

## 5. Analytics & monitoring

- [ ] **Plausible** over PostHog unless you need funnels. It is cookieless,
      which keeps the consent story simple, and this app has no user accounts
      to build product analytics around.
- [ ] **Sentry** on the FastAPI app. Set `traces_sample_rate` low (0.1) and
      scrub the request body — match contexts are large and pointless in an
      error report.
- [ ] Alert on the things that actually cost money or break the product:
      - Anthropic 4xx/5xx (credit exhaustion has already taken the app down once)
      - Football API quota below ~10%, from the numbers `/health` already exposes
      - Rate-limit rejections spiking, which means either real traffic or a script
      - p95 latency on `/api/match/{id}/narrative`
- [ ] Track, at minimum: matches opened, tab engagement, and cold-generation
      rate. That last one is the cost driver — if it is high, phase 3's warm
      job needs to cover more fixtures.
- [ ] Uptime check on `/health` from outside the platform.

---

## Known limitations to carry into launch

Worth stating up front rather than discovering in support:

- **Archive matches have thin data.** The 2014 World Cup has events but no
  team or player statistics, so those analyses render without stat bars or
  player cards. The heritage grid marks them "archive · limited stats".
- **No action photography.** Moment images are player portraits; the provider
  publishes no photographs of goals, and agency press photography requires a
  licence. `highlights_data.json` takes licensed stills per moment when you
  have them.
- **YouTube embeds depend on the rights holder.** A mapped video can be
  withdrawn or have embedding disabled at any time, and the panel will show
  YouTube's own unavailable state. Re-check the three mapped fixtures
  periodically.
- **The Match State Index is a model, not a measurement.** It distributes
  observed match totals across time by score state. It is labelled as such in
  the UI and should stay labelled.
