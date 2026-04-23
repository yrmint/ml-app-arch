# Git Workflow & Contribution Guidelines

## Overview

This project uses a feature-branch workflow with Pull Requests to ensure code quality and maintainability.

All changes must go through a Pull Request (PR) before being merged into the main branch.

---

## Branching Strategy

### Main Branch

* `main` — stable, production-ready code
* Direct commits to `main` are **not allowed**

---

### Feature Branches

All new work must be done in separate branches:

* `feature/<short-description>`
* `bugfix/<short-description>`
* `refactor/<short-description>`

#### Examples:

* `feature/audio-preprocessing`
* `feature/model-training`
* `bugfix/mfcc-calculation`

---

## Workflow

1. make sure your branch is up to date:

   ```
    git pull origin main
   ```
2. Create a branch from `main`:

   ```
   git checkout -b feature/your-feature-name
   ```
3. Implement your changes
4. Commit using clear messages
5. Push your branch:

   ```
   git push origin feature/your-feature-name
   ```
6. Open a Pull Request in GitHub
7. Assign at least one reviewer
8. Address review comments (if any)
9. Merge after approval

---

## Pull Request Rules

Each PR must:

* Have a clear title and description
* Reference the related Issue (if applicable)
* Explain:

    * What was done
    * Why it was done
    * How to test it

### PR Size

* Keep PRs small and focused
* Avoid large, multi-feature PRs

---

## Code Review

* At least **one approval** is required before merging
* Do not approve your own PR
* Reviewers should check:

    * Code clarity and structure
    * Correctness
    * Possible edge cases

---

## Commit Message Guidelines

Use simple and consistent messages:

* `feat: add CNN model`
* `fix: correct audio normalization`
* `refactor: split preprocessing module`

---

## Synchronization

Before opening a PR, make sure your branch is up to date:

```
git pull origin main
```

Resolve any merge conflicts locally before submitting the PR.

---

## CI Requirements

All Pull Requests must pass automated checks:

* Linting (flake8)
* Tests (pytest)

PRs with failing checks must not be merged.

---

## Best Practices

* Write modular and readable code
* Add tests for new functionality
* Avoid committing large data files
* Keep branches short-lived
