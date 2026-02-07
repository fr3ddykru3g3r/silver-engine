# Release Guide

This project ships installable desktop builds via GitHub Releases.

## Create a release
1. Bump the `version` in `package.json`.
2. Commit the change.
3. Tag the commit with `vX.Y.Z` (for example, `v0.2.0`).
4. Push the tag to GitHub.

The `Release` workflow in `.github/workflows/release.yml` will:
- Build the renderer.
- Create platform installers (macOS DMG, Windows NSIS, Linux AppImage).
- Publish the artifacts to the GitHub Release associated with the tag.
