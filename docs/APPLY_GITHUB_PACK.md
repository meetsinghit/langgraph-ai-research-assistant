# Applying this GitHub portfolio update pack

This ZIP is a **documentation/assets overlay**, not a replacement source repository. No application source, tests, Git metadata, local databases or credentials are included.

## 1. Review before applying

1. Keep the GitHub repository **Private** until your public-publication audit and any required IP/ownership checks are complete.
2. Review the included `README.md`, `docs/technical-handbook.html`, and images for the content you want to disclose.
3. Make sure Git is clean in the local project before applying:

   ```powershell
   cd "$env:USERPROFILE\Downloads\OpenCode\AI_POC"
   git status
   ```

   **Tip:** Change the relative project directory if your clone is located elsewhere. The pack does not need this user-specific path in any of its public-facing documentation.

## 2. Apply the ZIP on Windows

If `ai-research-assistant-github-pack.zip` is in Downloads:

```powershell
cd "$env:USERPROFILE\Downloads\OpenCode\AI_POC"
Copy-Item README.md "$env:TEMP\AI_Research_Assistant_README_before_portfolio.md"
Expand-Archive -Path "$env:USERPROFILE\Downloads\ai-research-assistant-github-pack.zip" -DestinationPath . -Force
```

The ZIP contains only README + `docs/` paths. It will **overwrite README.md and docs/SESSION_HANDOFF.md** with the proposed portfolio copy, and add the handbook, screenshots, diagrams and supporting documentation. It does not touch `agent/`, `app/`, `frontend/`, `tests/` or `.env`.

If you prefer to review first, extract the ZIP to a temporary directory and inspect the individual files before copying them into the project.

## 3. Inspect the diff

```powershell
git status
git diff -- README.md docs/SESSION_HANDOFF.md
```

Review the new `docs/assets/` images and `docs/*.md` files in your editor as well. `git diff` will not show *untracked* newly added assets until you stage them.

## 4. Commit and push (only after review)

```powershell
git add README.md docs/
git diff --cached --stat
git diff --cached --check
git commit -m "docs: publish engineering handbook and GitHub portfolio showcase"
git push origin main
```

This pushes only to the **existing GitHub repository**, respecting its current visibility. Changing visibility is a separate action in GitHub settings.

## 5. GitHub About and social preview

1. Open the repository's page on GitHub.
2. Edit the **About** description and **Topics** using `docs/GITHUB_METADATA.md`.
3. In **Settings → General → Social preview**, upload `docs/assets/social-preview.png`.
4. Verify README images and links render correctly on GitHub.
5. Do **not** set a localhost URL as the repository Website.

## 6. Open the HTML handbook

GitHub will display `.html` as source text in the file viewer. To use the interactive table of contents, theme toggle and embedded diagrams, **download `docs/technical-handbook.html` and open it locally in a browser**. It has no external script/style dependencies.

## 7. Public release is a separate decision

Read `docs/GITHUB_PUBLICATION_REVIEW.md`. The snapshot scan has limitations: the source ZIP did not contain Git history and could not independently verify secret exposure in earlier commits. Do not set the repository Public until historical scans, license/IP review and asset review are complete.
