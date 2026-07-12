# Deploying worker code to production

> **TL;DR:** a `git push` does **not** deploy. Production workers boot from a **sha256-pinned
> tarball** served off the app host, not from GitHub. To ship a change you rebuild + republish that
> tarball, bump its pin in the Vast template, and **destroy** the running workers so fresh ones
> re-fetch. This file is the runbook. (Specific endpoint / workergroup / template IDs and the exact
> host live in the Vast console + private ops notes — they're mutable, so they're not committed here.)

## Why a tarball instead of `git clone`

Anonymous GitHub clones of the `deni-nzt` repos **404** (the account is shadow-flagged to
unauthenticated clients). A Vast worker boots unauthenticated, so a plain `git clone` in the
onstart fails on a fresh box. Production therefore ships code as a tarball fetched from our own
host, with the git clone kept only as a fallback.

## How a worker gets its code (boot)

The Vast endpoint template's `/root/onstart.sh` does, in order:

1. `curl` the bootstrap tarball from the app host → `https://<app-host>/static/vast-bootstrap.tar.gz`.
2. `sha256sum -c` it against a **hardcoded `BS_SHA`** in the onstart. Mismatch → fall back to
   `git clone` (which 404s) → worker crash-loop. **Keep the published tarball's sha and `BS_SHA` in
   lockstep.**
3. Extract to `/workspace/vast-ltx-worker`, then `cp -r` to `/workspace/vast-pyworker` so the Vast
   framework's own `start_server.sh` skips its clone step.
4. Run `provision.sh` → `worker.py` (see [ARCHITECTURE.md](ARCHITECTURE.md)).

The tarball is served by **Caddy `file_server`** from `/opt/cubecrank-static/` on the web VM
(`handle_path /static/* → root /opt/cubecrank-static`). That directory is **root-owned** — the
normal `deploy@` user cannot write it, and it is *not* part of the app's repo or normal deploy, so
a standard app deploy never touches it. Publishing a new tarball needs root on the VM.

## Edit → ship recipe (the root path)

1. **Edit + commit + push** this repo (keeps GitHub in sync; does not deploy).
2. **Build the tarball from `HEAD`** — structure MUST be a top-level `vast-ltx-worker/` dir
   containing the tracked files:
   ```bash
   rm -rf /tmp/b && mkdir -p /tmp/b/vast-ltx-worker
   git -C ~/vast-ltx-worker archive HEAD | tar -x -C /tmp/b/vast-ltx-worker
   ( cd /tmp/b && tar czf vast-bootstrap.tar.gz vast-ltx-worker )
   shasum -a 256 /tmp/b/vast-bootstrap.tar.gz     # ← the new BS_SHA
   ```
3. **Publish it** (needs root on the web VM):
   ```bash
   sudo install -o root -g root -m644 /tmp/b/vast-bootstrap.tar.gz /opt/cubecrank-static/vast-bootstrap.tar.gz
   ```
4. **Bump `BS_SHA`** in the template onstart to the new sha (Vast console → edit template, or
   `vastai`). If it doesn't match the served file, every fresh worker fails the check.
5. **Recycle the workers** (see below).

## Recycling — the resume-doesn't-refetch gotcha

**A Vast *resume* of a stopped worker restarts the OLD baked code — it does NOT re-run the onstart
or re-fetch the tarball.** Only a **fresh create** pulls the new code. So:

1. Pause the autoscaler: set the endpoint's `cold_workers` to `0` (avoids a worker booting
   mid-swap against a half-updated tarball/sha).
2. `vastai destroy instance <id>` every current `ltx-i2v` worker.
3. Restore `cold_workers` to its normal value → the autoscaler **creates fresh** workers that fetch
   the new tarball.
4. Watch one boot to `idle` and confirm `GET :8188/health` → 200 before trusting the fleet.

## Path C — ship a fix with NO root on the VM

Because the tarball sha-check *passes on the unchanged old tarball*, you can ship a code fix without
touching `/opt/cubecrank-static` at all — by making a **new Vast template whose onstart patches the
changed files in place after extracting the old tarball**:

1. Take the current `/root/onstart.sh` (grab it from any worker), and **after** the `tar xzf …` and
   **before** the `cp -r … /workspace/vast-pyworker`, `cat >` the fixed `model_server.py` / `worker.py`
   into `/workspace/vast-ltx-worker/` (guard each with a `sha256sum -c` + an `echo "[ltxfix] … OK"`
   so a bad paste is obvious in the boot log). Leave `BS_URL`/`BS_SHA` pointing at the OLD tarball.
2. `vastai create template … --onstart-cmd "$(cat onstart.sh)" --raw` (same image / disk / env /
   search params as the live template) → parse the new `hash_id` from the output.
3. `vastai update workergroup <wg> --endpoint_id <ep> --template_hash <NEW> …` — **re-pass every
   knob** (search params, cold_mult, min_load, target_util, cold_workers) or they clobber to
   defaults. Pause the endpoint first (`--cold_workers 0`).
4. Destroy the workers → fresh ones boot the new template. **Rollback** = repoint the workergroup
   at the previous template hash + destroy.

> `vastai update template` on the live hash is server-side broken (400 "Invalid Creator ID") — make
> a **new** template rather than editing in place.

This is exactly how the 2026-07-12 cold-start fix (`model_server.py` health-gated `PIPELINE_READY`)
went live without VM root.
