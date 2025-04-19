# GitHub Pages Configuration Guide

This guide will help you set up GitHub Pages for the MAGNUS project while properly working with branch protection rules.

## Setting Up GitHub Pages with Branch Protection

When you want GitHub Pages to deploy from your `master` branch but still allow development on feature branches without deployment errors, follow these steps:

### 1. Configure GitHub Pages

1. Go to your repository on GitHub
2. Click on **Settings** > **Pages** in the left sidebar
3. Under "Source", select "Deploy from a branch"
4. Select "master" branch and "/docs" folder (or root folder if you prefer)
5. Click **Save**

### 2. Configure Environment Protection Rules

To prevent the "not allowed to deploy to github-pages due to environment protection rules" error:

1. Go to **Settings** > **Environments**
2. You should see a "github-pages" environment that was automatically created
3. Click on **github-pages** to configure it
4. Under "Deployment branches and tags", select one of these options:
   - **All branches**: If you want all branches to be able to trigger deployments (but only `master` will actually deploy)
   - **Selected branches**: If you want to restrict which branches can trigger builds
      - Add both `master` and your development branches (like `hw-agnostic`)
5. Click **Save protection rules**

### 3. Create a GitHub Actions Workflow for Pages (Optional but Recommended)

Creating a custom GitHub Actions workflow gives you more control over Pages deployments:

1. Create a new file: `.github/workflows/pages.yml`
2. Use the following configuration:

```yaml
name: Deploy GitHub Pages

on:
  push:
    branches: [ master ]
    paths:
      - 'docs/**'
      - '.github/workflows/pages.yml'
  
  # Allow manual triggering
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

# Allow only one concurrent deployment, skipping runs queued between the run in-progress and latest queued
concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/master'
    steps:
      - name: Checkout
        uses: actions/checkout@v3
      
      - name: Setup Pages
        uses: actions/configure-pages@v3
      
      - name: Build with Jekyll
        uses: actions/jekyll-build-pages@v1
        with:
          source: ./docs
          destination: ./_site
      
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v2

  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v2
```

This workflow:
- Only runs on pushes to the `master` branch
- Only deploys from the `master` branch
- Restricts builds to changes in the `docs/` directory
- Properly handles environment deployment protection

### 4. Configure Branch Protection for the Workflow

If using the GitHub Actions workflow approach, update your branch protection rules:

1. Go to **Settings** > **Branches** > **Branch protection rules**
2. Edit your `master` branch protection rule
3. Under "Require status checks to pass before merging"
4. Add "build" from the pages workflow to the required checks

## Additional Tips

1. **Documentation in Feature Branches**: You can work on documentation in feature branches, but it won't be deployed to GitHub Pages until merged to `master`.

2. **Preview Documentation Changes**: To preview documentation changes before merging:
   - Use local preview (with tools like Jekyll or MkDocs)
   - Consider setting up a preview workflow for pull requests (more advanced)

3. **Custom Domain**: If using a custom domain, configure it in **Settings** > **Pages** > **Custom domain**

4. **Documentation Structure**: Keep a consistent structure in your `/docs` folder with a clear README.md or index.md as the landing page

By following these steps, you'll have a properly configured GitHub Pages setup that works with your branch protection rules and only deploys from the `master` branch.