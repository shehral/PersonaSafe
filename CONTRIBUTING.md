# Contributing to PersonaSafe

Thank you for your interest in contributing to PersonaSafe! This document provides guidelines and instructions for contributing.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing Guidelines](#testing-guidelines)
6. [Documentation](#documentation)
7. [Pull Request Process](#pull-request-process)

---

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- Hugging Face account with access to Gemma 3 models (accept license)

### Setup Development Environment

```bash
# 1) Clone
git clone https://github.com/shehral/PersonaSafe.git
cd PersonaSafe

# 2) Create and activate venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run tests
pytest -q

# 5) Launch dashboard (optional)
streamlit run examples/dashboard/app.py
```

---

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

**Branch Naming:**
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions/fixes

### 2. Make Changes

- Write code following our standards
- Add tests for new features
- Update documentation
- Keep commits atomic and well-described

### 3. Test Your Changes

```bash
# Run all tests
pytest -v

# Run specific test
pytest tests/core/test_persona_extractor.py -v

# Check coverage
pytest --cov=personasafe tests/

# Lint code (optional if installed)
black personasafe/ tests/
flake8 personasafe/ tests/
mypy personasafe/
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: Add new screening algorithm

- Implement multi-trait screening
- Add tests for edge cases
- Update documentation"
```

**Commit Message Format (Conventional):**
```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, etc.)
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

CI will run tests and coverage on Linux/macOS with Python 3.10–3.12.

---

## Coding Standards

### Python Style

We follow **PEP 8** with these specifics:

```python
# Good
def compute_persona_vector(
    positive_prompts: List[str],
    negative_prompts: List[str],
    trait_name: str
) -> torch.Tensor:
    """
    Computes a persona vector using contrastive prompts.

    Args:
        positive_prompts: Prompts exhibiting the target trait
        negative_prompts: Prompts exhibiting the opposite trait
        trait_name: Name of the trait for caching

    Returns:
        Normalized persona vector
    """
    # Implementation
    pass
```

**Guidelines:**
- Maximum line length: 100 characters
- Use type hints for all functions
- Write docstrings (Google style)
- Use descriptive variable names
- Keep functions under 50 lines when possible

### Code Organization

```python
# Imports order
import os  # Standard library
import sys

import torch  # Third-party
import numpy as np

from personasafe.core import PersonaExtractor  # Local
```

### Formatting Tools

```bash
# Auto-format code
black personasafe/ tests/ scripts/

# Sort imports
isort personasafe/ tests/ scripts/

# Check style
flake8 personasafe/ tests/

# Type checking
mypy personasafe/
```

---

## Testing Guidelines

### Test Structure

```python
# tests/module/test_feature.py
import pytest
from personasafe.module import Feature

def test_feature_basic_functionality():
    """Test basic feature operation."""
    feature = Feature()
    result = feature.process("input")
    assert result == "expected"

def test_feature_edge_cases():
    """Test edge cases and error handling."""
    feature = Feature()
    with pytest.raises(ValueError):
        feature.process(None)
```

### Test Coverage Requirements

- **Minimum coverage:** 80%
- All new features must include tests
- Test both happy path and error cases
- Use mocking to avoid external dependencies

### Running Tests

```bash
# All tests
pytest -v

# With coverage
pytest --cov=personasafe --cov-report=html tests/

# Specific marker
pytest -m "not slow" -v

# Stop on first failure
pytest -x
```

---

## Documentation

### Code Documentation

```python
def my_function(param1: str, param2: int) -> bool:
    """
    Brief description of function.

    Longer description if needed. Explain complex logic,
    algorithms, or important considerations.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is empty
        RuntimeError: When operation fails

    Example:
        >>> result = my_function("test", 42)
        >>> print(result)
        True
    """
    pass
```

### Documentation Files

Update these when making changes:

- `docs/API_REFERENCE.md` - For API changes
- `docs/TUTORIAL.md` - For usage changes
- `CHANGELOG.md` - For version changes
- `README.md` - For major features

---

## Pull Request Process

### Before Submitting

- [ ] All tests pass locally
- [ ] Code is formatted (black, isort)
- [ ] No linting errors (flake8, mypy)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] Commits are clean and descriptive

### PR Description Template

```markdown
## Description
Brief description of changes

## Motivation and Context
Why is this change needed?

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How has this been tested?

## Checklist
- [ ] Tests pass
- [ ] Code formatted
- [ ] Documentation updated
- [ ] No breaking changes (or documented if yes)
```

### Review Process

1. **Automated Checks:** CI/CD runs tests and linting
2. **Code Review:** Maintainer reviews code
3. **Feedback:** Address review comments
4. **Approval:** Maintainer approves
5. **Merge:** PR is merged into main

---

## Areas for Contribution

### High Priority

- [ ] Integration tests
- [ ] Additional persona traits
- [ ] Performance optimizations
- [ ] Dashboard enhancements
- [ ] HPC batch scripts

### Medium Priority

- [ ] Visualization improvements
- [ ] Additional model support
- [ ] Monitoring features
- [ ] Export/import utilities

### Documentation

- [ ] Tutorials and examples
- [ ] Troubleshooting guides
- [ ] Case studies

---

## Getting Help

- **Questions:** Open a GitHub Discussion
- **Bugs:** Open a GitHub Issue
- **Chat:** Join our community (link TBD)
- **Email:** contact@personasafe.dev (when available)

---

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Given credit in publications (if applicable)

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to PersonaSafe!** 🎉
