# GitHub Repository Setup Instructions

This document provides step-by-step instructions for setting up branch protection rules and other GitHub configuration for the MAGNUS repository.

## Branch Protection Rules

To enforce code quality and ensure all tests pass before merging, follow these steps:

1. Navigate to your GitHub repository
2. Click on **Settings** > **Branches** in the repository navigation
3. Under "Branch protection rules", click **Add rule**
4. In "Branch name pattern", enter `master` (or `main` if you're using that as your default branch)
5. Configure the following settings:

### Required Settings
- [x] **Require a pull request before merging**
  - [x] Require approvals (set to at least 1)
  - [x] Dismiss stale pull request approvals when new commits are pushed
- [x] **Require status checks to pass before merging**
  - [x] Require branches to be up to date before merging
  - In the search box, find and select:
    - `Test` (from the rust-ci workflow)
    - `Lint` (from the rust-ci workflow)
- [x] **Require conversation resolution before merging**
- [x] **Include administrators** (recommended to ensure everyone follows the same rules)

### Optional Settings
- [x] **Restrict who can push to matching branches** (Add repository administrators and core team members)
- [ ] **Allow force pushes** (Leave unchecked for better history preservation)
- [ ] **Allow deletions** (Leave unchecked to prevent accidental branch deletion)

6. Click **Create** to save the rule

## Repository Settings

### Default Branch
1. Go to **Settings** > **Branches**
2. Under "Default branch", make sure it's set to `master` (or `main`)
3. If you need to change it, click the switch icon and follow the prompts

### Merge Button Settings
1. Go to **Settings** > **General**
2. Scroll down to "Pull Requests"
3. Configure merge button settings:
   - [x] Allow merge commits
   - [x] Allow squash merging
   - [ ] Allow rebase merging (uncheck to avoid rebase issues)
   - [x] Automatically delete head branches (for cleanup after merging)

### Actions Permissions
1. Go to **Settings** > **Actions** > **General**
2. Set "Actions permissions" to "Allow all actions and reusable workflows"
3. Under "Workflow permissions", select "Read and write permissions" 

## Additional Settings

### Repository Topics
1. Go to the main page of your repository
2. Click the ⚙ (gear) icon next to "About"
3. Add relevant topics such as:
   - rust
   - sparse-matrix
   - linear-algebra
   - high-performance-computing
   - scientific-computing

### Social Preview
1. Go to **Settings** > **General**
2. Scroll down to "Social preview"
3. Click "Edit" and upload a project logo or diagram to help identify the project in social media shares

## License File
The repository already includes an MIT license file, which is appropriate for open-source scientific computing projects.

---

After completing this setup, your repository will enforce that:
1. All pull requests require at least one review
2. All tests must pass before merging
3. CI checks must succeed
4. Branch protection applies to everyone, including administrators

This ensures code quality remains high and prevents accidental pushing of broken code to the main branch.