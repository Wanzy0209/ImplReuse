# Find the hash of the commit you want to cherry pick
# (for example, abcdef12345)
git log

git fetch origin release/2.9
git checkout release/2.9
git cherry-pick abcdef12345

# Submit a PR based against 'release/2.9' either:
# via the GitHub UI
git push my-fork

# via the GitHub CLI
gh pr create --base release/2.9