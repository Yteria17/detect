# Contributing Guide

## Table of Contents

1. [Welcome](#welcome)
2. [Code of Conduct](#code-of-conduct)
3. [How to Contribute](#how-to-contribute)
4. [Development Setup](#development-setup)
5. [Coding Guidelines](#coding-guidelines)
6. [Pull Request Process](#pull-request-process)
7. [Issue Guidelines](#issue-guidelines)
8. [Community](#community)

---

## Welcome

Thank you for considering contributing to the Multi-Agent Disinformation Detection System! We welcome contributions from everyone, whether you're fixing a typo, reporting a bug, or implementing a new feature.

### Ways to Contribute

- 🐛 Report bugs
- 💡 Suggest features
- 📝 Improve documentation
- 🔧 Fix issues
- ✨ Add new features
- 🧪 Write tests
- 🎨 Improve UX/UI
- 🌍 Translate documentation

---

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Examples of behavior that contributes to a positive environment:**

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Examples of unacceptable behavior:**

- The use of sexualized language or imagery
- Trolling, insulting/derogatory comments, and personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team at conduct@detect-project.com. All complaints will be reviewed and investigated promptly and fairly.

---

## How to Contribute

### First-Time Contributors

If this is your first time contributing:

1. **Find an issue labeled `good first issue`**
   - Browse [good first issues](https://github.com/yourusername/detect/labels/good%20first%20issue)
   - Comment on the issue to let others know you're working on it

2. **Fork the repository**
   ```bash
   # Click "Fork" button on GitHub
   git clone https://github.com/YOUR_USERNAME/detect.git
   cd detect
   ```

3. **Create a branch**
   ```bash
   git checkout -b fix/issue-123-description
   ```

4. **Make your changes**
   - Follow our [coding guidelines](#coding-guidelines)
   - Write tests for new features
   - Update documentation

5. **Submit a pull request**
   - See [Pull Request Process](#pull-request-process)

### Experienced Contributors

For larger contributions:

1. **Discuss your idea first**
   - Open an issue to discuss your proposal
   - Get feedback from maintainers
   - Agree on implementation approach

2. **Follow the development workflow**
   - See [Development Guide](DEVELOPMENT.md)
   - Ensure all tests pass
   - Maintain code coverage above 80%

3. **Submit a well-documented PR**
   - Clear description of changes
   - Link to related issues
   - Include screenshots for UI changes

---

## Development Setup

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Git
- PostgreSQL 14+
- Redis 7+

### Setup Steps

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/detect.git
cd detect

# 2. Add upstream remote
git remote add upstream https://github.com/yourusername/detect.git

# 3. Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 5. Setup pre-commit hooks
pre-commit install

# 6. Copy environment file
cp .env.example .env.development

# 7. Start services
docker-compose up -d postgres redis weaviate

# 8. Run migrations
alembic upgrade head

# 9. Run tests
pytest

# 10. Start development server
uvicorn detect.main:app --reload
```

### Keeping Your Fork Updated

```bash
# Fetch upstream changes
git fetch upstream

# Merge upstream main into your local main
git checkout main
git merge upstream/main

# Push to your fork
git push origin main
```

---

## Coding Guidelines

### Python Style

Follow **PEP 8** with these modifications:

- Line length: **100 characters** (not 79)
- Use **double quotes** for strings
- Use **trailing commas** in multi-line structures

```python
# Good
def process_claim(
    claim: str,
    options: Dict[str, Any],
    context: Optional[Dict] = None,
) -> FactCheckResult:
    """Process a claim and return verification result."""
    pass

# Bad
def process_claim(claim: str, options: Dict[str, Any], context: Optional[Dict] = None) -> FactCheckResult:
    pass
```

### Type Hints

Always include type hints:

```python
# Good
def calculate_confidence(verdicts: List[str]) -> float:
    pass

# Bad
def calculate_confidence(verdicts):
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def verify_claim(claim: str, evidence: List[Document]) -> Verdict:
    """
    Verify claim against provided evidence.

    Args:
        claim: The claim text to verify
        evidence: List of evidence documents

    Returns:
        Verdict object with result and confidence

    Raises:
        ValueError: If claim is empty
        VerificationError: If verification fails

    Examples:
        >>> verify_claim("Paris is capital of France", evidence)
        Verdict(verdict='SUPPORTED', confidence=0.95)
    """
    pass
```

### Testing Requirements

- **Unit tests** for all new functions/methods
- **Integration tests** for new features
- **Minimum 80% code coverage**
- Tests must pass before PR approval

```python
# Example test
def test_claim_verification():
    """Test claim verification with mock evidence"""
    claim = "Paris is the capital of France"
    evidence = [create_mock_evidence()]

    result = verify_claim(claim, evidence)

    assert result.verdict == "SUPPORTED"
    assert result.confidence > 0.8
```

### Commit Messages

Follow **Conventional Commits**:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

**Examples:**

```
feat(fact-checker): add graph-based reasoning

- Implement GraphFC algorithm for complex claims
- Add triplet extraction and verification
- Update tests and documentation

Closes #123

fix(api): handle timeout errors gracefully

Previously, LLM timeouts crashed the entire pipeline.
Now we retry with exponential backoff.

Fixes #456
```

---

## Pull Request Process

### Before Submitting

1. **Run all checks locally**
   ```bash
   # Format code
   make format

   # Run linters
   make lint

   # Run tests
   pytest tests/ --cov=detect

   # Build docs
   cd docs && make html
   ```

2. **Update documentation**
   - Update relevant `.md` files
   - Add docstrings to new functions
   - Update API documentation if endpoints changed

3. **Update CHANGELOG.md**
   ```markdown
   ## [Unreleased]

   ### Added
   - Graph-based reasoning for complex claims (#123)

   ### Fixed
   - Timeout handling in LLM service (#456)
   ```

### PR Template

When creating a PR, use this template:

```markdown
## Description

Brief description of changes

Fixes #(issue number)

## Type of Change

- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing

Describe the tests you ran:

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing performed

## Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Code is commented (particularly complex areas)
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Tests added that prove fix/feature works
- [ ] Dependent changes merged
- [ ] CHANGELOG.md updated

## Screenshots (if applicable)

Add screenshots for UI changes

## Additional Notes

Any additional information
```

### Review Process

1. **Automated checks**
   - CI pipeline must pass (tests, linting, security scan)
   - Code coverage must not decrease

2. **Code review**
   - At least one maintainer approval required
   - Address all review comments
   - Re-request review after changes

3. **Merge**
   - Squash commits for feature PRs
   - Rebase commits for bug fixes
   - Maintainer will merge once approved

---

## Issue Guidelines

### Bug Reports

Use the bug report template:

```markdown
**Describe the bug**
A clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Click on '...'
3. See error

**Expected behavior**
What you expected to happen

**Screenshots**
If applicable, add screenshots

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10.12]
- Detect version: [e.g., 1.0.0]

**Additional context**
Any other relevant information
```

### Feature Requests

Use the feature request template:

```markdown
**Is your feature request related to a problem?**
Clear description of the problem

**Describe the solution you'd like**
Clear description of desired solution

**Describe alternatives you've considered**
Other approaches considered

**Additional context**
Mockups, examples, or any other relevant information
```

### Issue Labels

| Label | Description |
|-------|-------------|
| `bug` | Something isn't working |
| `enhancement` | New feature or request |
| `documentation` | Documentation improvements |
| `good first issue` | Good for newcomers |
| `help wanted` | Extra attention needed |
| `question` | Further information requested |
| `wontfix` | This will not be worked on |
| `duplicate` | Duplicate of existing issue |
| `priority: high` | High priority |
| `priority: low` | Low priority |

---

## Community

### Communication Channels

- **GitHub Discussions**: [General discussions](https://github.com/yourusername/detect/discussions)
- **Discord**: [Join our server](https://discord.gg/detect)
- **Twitter**: [@detect_project](https://twitter.com/detect_project)
- **Email**: community@detect-project.com

### Weekly Office Hours

Join our maintainers for office hours:
- **When**: Every Tuesday, 3-4 PM UTC
- **Where**: Discord voice channel
- **What**: Q&A, discuss contributions, get help

### Recognition

Contributors are recognized in:

- **README.md**: Listed as contributors
- **CONTRIBUTORS.md**: Detailed contributions
- **Release Notes**: Credited for features/fixes
- **Monthly Newsletter**: Featured contributors

### Becoming a Maintainer

Active contributors may be invited to become maintainers. Maintainers have:

- **Write access** to the repository
- **Voting rights** on project decisions
- **Responsibility** to review PRs and help contributors

Criteria:
- 10+ merged PRs
- Consistent quality contributions
- Active in community support
- Understanding of project architecture

---

## Financial Support

### Sponsorship

Support the project through:

- **GitHub Sponsors**: [Sponsor us](https://github.com/sponsors/yourusername)
- **Open Collective**: [Contribute](https://opencollective.com/detect)

Funds are used for:
- Infrastructure costs
- Contributor incentives
- Conference attendance
- Swag for contributors

### Bounty Program

We offer bounties for:
- Critical bug fixes: $50-$200
- Priority features: $100-$500
- Security vulnerabilities: $200-$2000

See [Bounty Program](https://detect-project.com/bounty) for details.

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## Questions?

Don't hesitate to ask! We're here to help:

- Open a [GitHub Discussion](https://github.com/yourusername/detect/discussions)
- Join our [Discord](https://discord.gg/detect)
- Email us at community@detect-project.com

**Thank you for contributing to making the digital information ecosystem safer!** 🛡️

---

**Last Updated**: 2025-01-15
