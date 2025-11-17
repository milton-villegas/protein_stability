# CI/CD Guide for Protein Stability DoE Toolkit

## What is CI/CD?

**CI/CD = Continuous Integration / Continuous Deployment**

A system that automatically tests your code every time you push changes to GitHub.

---

## Why Do You Need It?

### Problem Without CI/CD

```
You change optimizer.py
        ↓
Push to GitHub
        ↓
??? (Hope nothing broke)
        ↓
3 weeks later: Bug discovered by user
        ↓
Emergency fix needed
```

### Solution With CI/CD

```
You change optimizer.py
        ↓
Push to GitHub
        ↓
GitHub automatically runs 71 tests in 2 minutes
        ↓
✅ All pass → Confident!
❌ 3 fail → Fix immediately (before anyone sees it)
```

---

## Real Benefits for This Project

### 1. **Safety for Lab Experiments**

Your code calculates:
- Buffer volumes for Opentrons robot
- Statistical analysis of experiments
- Bayesian optimization suggestions

**A bug could:**
- ❌ Waste expensive reagents
- ❌ Ruin days of experiments
- ❌ Give wrong scientific conclusions

**CI/CD ensures:**
- ✅ Volume calculations always correct
- ✅ Statistical models validated
- ✅ Bugs caught before experiments

### 2. **Multi-Platform Compatibility**

Tests run on:
- ✅ Linux (lab computers)
- ✅ Windows (most users)
- ✅ macOS (some researchers)

Catches OS-specific bugs automatically!

### 3. **Collaborative Development**

When someone contributes:
```
1. They submit pull request
2. CI/CD automatically tests their code
3. You see: ✅ 71/71 tests passed
4. Safe to merge!
```

No need to manually test everything!

### 4. **Code Quality Standards**

Enforces minimum standards:
- ✅ All tests must pass
- ✅ Coverage ≥ 40%
- ✅ No syntax errors
- ✅ Works on all Python versions (3.8-3.11)

---

## What Gets Tested Automatically?

Every push triggers:

### 1. **Unit Tests** (71+ tests)
```bash
tests/core/test_doe_analyzer.py .......... ✅
tests/core/test_optimizer.py .......... ✅
tests/core/test_plotter.py .......... ✅
... (all test files)
```

### 2. **Code Coverage** (40%+ required)
```
core/optimizer.py      60%  ✅
core/doe_analyzer.py   85%  ✅
core/plotter.py        99%  ✅
----------------------------
Overall coverage:      45%  ✅ (above 40% threshold)
```

### 3. **Platform Compatibility**
```
Python 3.8 + Ubuntu     ✅
Python 3.9 + Ubuntu     ✅
Python 3.10 + Ubuntu    ✅
Python 3.11 + Ubuntu    ✅
Python 3.11 + Windows   ✅
Python 3.11 + macOS     ✅
... (12 combinations)
```

---

## How to View Results

### On GitHub Website

1. **Go to your repository**
2. **Click "Actions" tab**
3. **See all test runs:**

```
✅ Increase test coverage from 23% to 40-45%
   Ran 2 minutes ago • All checks passed

✅ Add documentation
   Ran 1 day ago • All checks passed

❌ Refactor optimizer (WIP)
   Ran 2 days ago • 3 tests failed
```

4. **Click any run to see details:**

```
Test on Python 3.11 / ubuntu-latest
  ✅ Checkout code
  ✅ Set up Python 3.11
  ✅ Install dependencies
  ✅ Run tests (71 passed, 0 failed)
  ✅ Upload coverage reports
  ✅ Check coverage threshold (45% > 40%)
```

### On Pull Requests

Every PR shows:

```
Pull Request #15: Add new design type

Checks: ✅ All checks have passed

  ✅ Tests (71/71 passed)
  ✅ Coverage: 45%
  ✅ Python 3.8-3.11 compatible
  ✅ Works on Linux, Windows, macOS

[Merge pull request] ← Green button = safe to merge!
```

---

## Setup Instructions

### Option 1: Full CI/CD (Recommended)

The file `.github/workflows/tests.yml` is already created!

**Tests on:**
- Python 3.8, 3.9, 3.10, 3.11
- Ubuntu, Windows, macOS
- 12 total configurations

**Just push it:**
```bash
git add .github/workflows/tests.yml
git commit -m "Add CI/CD automation"
git push
```

That's it! GitHub will automatically start running tests.

### Option 2: Simple CI/CD

For a simpler setup (just Ubuntu + Python 3.11):

```bash
# Remove the full version
rm .github/workflows/tests.yml

# Rename simple version
mv .github/workflows/tests-simple.yml.example .github/workflows/tests.yml

# Commit
git add .github/workflows/tests.yml
git commit -m "Add simple CI/CD"
git push
```

---

## Understanding the Workflow File

### Key Sections

```yaml
# When to run
on:
  push:
    branches: [ main, master ]  # Run on pushes to main
  pull_request:               # Run on all pull requests

# What to test
strategy:
  matrix:
    python-version: ["3.8", "3.9", "3.10", "3.11"]
    os: [ubuntu-latest, windows-latest, macos-latest]

# Steps to execute
steps:
  - Checkout code
  - Set up Python
  - Install dependencies (pip install -r requirements.txt)
  - Run tests (pytest tests/)
  - Check coverage (must be ≥ 40%)
```

---

## Cost

**GitHub Actions is FREE for public repositories!**

- Unlimited minutes for public repos
- 2,000 minutes/month for private repos (usually enough)

Your test suite runs in ~2 minutes, so:
- Public repo: ✅ Unlimited free
- Private repo: 2,000 min ÷ 2 min = 1,000 runs/month (way more than needed)

---

## Common Use Cases

### 1. Before Merging a Feature

```bash
git checkout -b new-feature
# ... make changes ...
git push origin new-feature
```

GitHub automatically:
1. Runs all tests
2. Shows results on the branch
3. You see if it's safe to merge

### 2. Protecting Main Branch

Enable "branch protection rules":
1. Go to Settings → Branches
2. Add rule for `main`
3. Require status checks to pass
4. ✅ Now you can't merge broken code!

### 3. Tracking Code Quality Over Time

Every commit shows:
```
Commit abc123: ✅ 71 tests, 45% coverage
Commit def456: ✅ 73 tests, 47% coverage  ← Improvement!
Commit ghi789: ❌ 73 tests, 2 failed      ← Caught bug!
```

---

## Troubleshooting

### Tests Pass Locally but Fail on CI

**Common causes:**

1. **Missing dependency in requirements.txt**
   ```bash
   # Add it:
   echo "missing-package" >> requirements.txt
   ```

2. **OS-specific code**
   ```python
   # Bad (only works on Windows):
   path = "C:\\data\\file.txt"

   # Good (works everywhere):
   import os
   path = os.path.join("data", "file.txt")
   ```

3. **Hardcoded paths**
   ```python
   # Bad:
   df = pd.read_csv("/home/user/data.csv")

   # Good:
   import os
   data_path = os.path.join(os.path.dirname(__file__), "data.csv")
   df = pd.read_csv(data_path)
   ```

### Tests Are Slow

Optimize by:
1. Running only changed tests first
2. Caching dependencies
3. Parallelizing tests

### Want Faster Feedback?

Set up **pre-commit hooks** to test locally BEFORE pushing:

```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
# Now tests run automatically before each commit
```

---

## Advanced Features

### 1. **Code Coverage Badges**

Add to your README.md:

```markdown
![Coverage](https://img.shields.io/codecov/c/github/your-username/protein_stability)
```

Shows: ![Coverage](https://img.shields.io/badge/coverage-45%25-green)

### 2. **Automatic Releases**

When you tag a version:
```bash
git tag v1.0.0
git push --tags
```

CI/CD can automatically:
- Build distribution packages
- Publish to PyPI
- Create GitHub release

### 3. **Dependency Updates**

Use Dependabot to automatically:
- Check for outdated packages
- Create PRs to update them
- CI/CD tests the updates automatically

---

## Best Practices

### ✅ DO:
- Run CI/CD on all branches
- Require tests to pass before merging
- Keep tests fast (< 5 minutes)
- Fix failing tests immediately
- Monitor coverage trends

### ❌ DON'T:
- Skip tests "just this once"
- Ignore failing tests
- Commit commented-out test code
- Have flaky tests (pass/fail randomly)
- Test external services directly (use mocks)

---

## Next Steps

1. **Push the workflow file** (see Setup Instructions above)
2. **Make a small change** and push it
3. **Watch the Actions tab** to see tests run
4. **Enable branch protection** to require passing tests
5. **Add a badge** to your README showing test status

---

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [pytest CI/CD Guide](https://docs.pytest.org/en/stable/example/simple.html#github-actions)
- [Coverage.py](https://coverage.readthedocs.io/)

---

## Questions?

- **"Why did my tests fail?"** → Click the failed run in Actions tab to see error logs
- **"How do I skip CI for a commit?"** → Add `[skip ci]` to commit message
- **"Can I test with different dependencies?"** → Yes! Modify the matrix in tests.yml
- **"How much does this cost?"** → Free for public repos, 2,000 min/month for private

