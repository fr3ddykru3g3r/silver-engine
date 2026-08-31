# IRIS safety publication

The local publication staging tree is prepared at:

`/private/tmp/silver-engine-repo`

The staged branch is `codex/iris-base-report-2026-08-31` and its current local
commit is recorded in `IRIS_ARCHIVE_AUDIT_2026-09-01.json`. The branch contains
the report, notebooks, code, sanitized manifests, figures, tables, and OBJ
meshes. The large raw-data archive is intentionally kept out of Git history.

After authenticating with an account authorized to publish to
`fr3ddykru3g3r/silver-engine`, run:

```bash
gh auth login -h github.com
git -C /private/tmp/silver-engine-repo push -u origin codex/iris-base-report-2026-08-31
gh release create iris-2026-08-31 \
  --repo fr3ddykru3g3r/silver-engine \
  --title "IRIS full safety archive" \
  --notes "Use the SHA-256 recorded in IRIS_ARCHIVE_AUDIT_2026-09-01.json."
gh release upload iris-2026-08-31 \
  '/Users/Kyrosah/Documents/ChatGPT/La Liga 38-0/IRIS_Colab_FULL_WITH_ACQUIRED_DATA_2026-08-31.zip' \
  --repo fr3ddykru3g3r/silver-engine --clobber
```

Verify both the branch and release asset independently before any cleanup:

```bash
gh api repos/fr3ddykru3g3r/silver-engine/branches/codex/iris-base-report-2026-08-31
gh release view iris-2026-08-31 --repo fr3ddykru3g3r/silver-engine
```

The local IRIS data must be retained until the branch, release, asset name,
asset size, and SHA-256 value are all verified remotely. No deletion command
is part of this recipe.
