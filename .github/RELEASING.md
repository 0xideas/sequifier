# Publishing release notes

GitHub Releases is the public release-notes channel for Sequifier. Use the
existing `vX.X.X.X` tag for each version; the docs workflow also builds from
these tags.

For each release:

1. Confirm the tag points to the intended commit and matches the version in
   `pyproject.toml` and `docs/source/conf.py`.
2. Confirm that the corresponding package version is available on PyPI.
3. On GitHub, open **Releases → Draft a new release** and select the existing
   tag. Select the previous published tag as the comparison baseline.
4. If `release-notes/<tag>.md` exists, paste it into the release description;
   otherwise write a short summary of user-visible features, fixes, and
   breaking changes or migration steps. Click **Generate release notes** to
   add the PR list and full changelog link below the summary, then review them.
5. Publish the release. Check that its tag, package version, and documentation
   version agree.

Release notes should describe what users can do or need to change, rather than
repeat commit messages. If a version has no breaking changes, say so explicitly.
