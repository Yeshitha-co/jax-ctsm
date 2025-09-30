# Contributing to JAX-CTSM

Thank you for your interest in contributing to JAX-CTSM! This document provides guidelines for contributing to the project.

## Development Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/jax-ctsm.git
cd jax-ctsm
```

### 2. Create a development environment

```bash
conda create -n jax-ctsm-dev python=3.10
conda activate jax-ctsm-dev
pip install -e ".[dev]"
```

### 3. Install pre-commit hooks

```bash
pre-commit install
```

## Code Standards

### Python Style

- Follow **PEP 8** guidelines
- Use **Black** for code formatting (line length: 100)
- Use **Ruff** for linting
- Use **type hints** for all functions
- Write **docstrings** for all public functions (Google style)

### JAX-Specific Guidelines

1. **Pure Functions**: All physical process functions should be pure (no side effects)
2. **Immutable Data**: Use NamedTuples or frozen dataclasses for state
3. **Vectorization**: Design for `vmap` - avoid Python loops over data
4. **JIT Compatibility**: Ensure functions work with `jax.jit`
5. **Array Shape**: Always document array shapes in docstrings `[n_patches, ...]`

### Example Function Template

```python
def my_process(
    state: PatchState,
    params: MyParams,
) -> MyOutput:
    """Short description.
    
    Longer description explaining the physical process and equations.
    
    Args:
        state: Patch state containing inputs
        params: Process-specific parameters
        
    Returns:
        Output containing calculated fluxes/states
        
    Note:
        Important implementation details or caveats.
        
    Example:
        >>> state = create_patch_state(...)
        >>> output = my_process(state, params)
    """
    # Implementation
    ...
```

## Testing

### Writing Tests

- Place tests in `tests/` mirroring `src/` structure
- Use `pytest` for all tests
- Aim for >90% code coverage
- Include both unit and integration tests

### Test Categories

1. **Unit Tests**: Test individual functions in isolation
2. **Integration Tests**: Test full process chains
3. **Validation Tests**: Compare against Fortran CTSM output
4. **Physical Tests**: Verify physical behavior (e.g., conservation)

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src/jax_ctsm --cov-report=html

# Specific test file
pytest tests/physics/test_maintenance_respiration.py -v

# Specific test
pytest tests/physics/test_maintenance_respiration.py::TestLeafMR::test_basic -v
```

## Adding a New Physical Process

### 1. Study the Fortran Implementation

```bash
# Find the Fortran module
cd CTSM/src/biogeochem
grep -r "subroutine MyProcess"
```

### 2. Design the Data Structures

- Identify inputs (state variables)
- Identify outputs (fluxes)
- Design PyTree-compatible structures

### 3. Implement the Process

```python
# src/jax_ctsm/physics/my_process.py

from jax_ctsm.core.hierarchy import PatchState
from jax_ctsm.params.my_params import MyParams

def my_process(state: PatchState, params: MyParams) -> MyOutput:
    """Implementation"""
    ...
```

### 4. Write Comprehensive Tests

```python
# tests/physics/test_my_process.py

class TestMyProcess:
    def test_basic_calculation(self):
        """Test basic functionality"""
        ...
    
    def test_edge_cases(self):
        """Test boundary conditions"""
        ...
    
    def test_vs_fortran(self):
        """Compare with CTSM Fortran output"""
        ...
```

### 5. Add Documentation

- Update `README.md` with status
- Add example in `examples/`
- Document equations and references

### 6. Submit Pull Request

- Create feature branch: `git checkout -b feature/my-process`
- Commit with clear messages
- Push and create PR
- Request review

## Documentation

### Docstring Style (Google Format)

```python
def function(arg1: type1, arg2: type2) -> return_type:
    """One-line summary.
    
    Extended description with more details. Explain the physical
    meaning, equations, and any important implementation notes.
    
    Args:
        arg1: Description of arg1 with units [unit] and shape [n, m]
        arg2: Description of arg2
        
    Returns:
        Description of return value with shape
        
    Raises:
        ValueError: When this error occurs
        
    Note:
        Additional implementation notes or caveats.
        
    References:
        - Citation 1
        - Citation 2
    """
```

### Type Hints

Always use type hints:

```python
from typing import Tuple, Optional
import jax.numpy as jnp

def process(
    values: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    ...
```

## Spatial Hierarchy

All processes should respect CTSM's spatial hierarchy:

```
Gridcell (1°×1° grid cell)
  └── Landunit (land cover type: veg, crop, urban, lake, glacier)
      └── Column (soil/snow column)
          └── Patch (plant functional type - PFT)
```

### Aggregation Functions

Use provided utilities:

```python
from jax_ctsm.core.hierarchy import patch_to_column, column_to_patch

# Aggregate patch → column
column_values = patch_to_column(
    patch_values, 
    patch_weights, 
    column_indices, 
    n_columns
)

# Broadcast column → patch
patch_values = column_to_patch(column_values, column_indices)
```

## Performance Guidelines

### DO

✅ Use JAX arrays (`jnp.array`) not NumPy
✅ Vectorize operations over patches
✅ Use `@jax.jit` for hot paths
✅ Profile before optimizing

### DON'T

❌ Use Python loops over data dimensions
❌ Use global state or side effects
❌ Modify arrays in-place
❌ Use dynamic shapes inside JIT

### Example Vectorization

```python
# BAD: Python loop
for p in range(n_patches):
    result[p] = compute(values[p])

# GOOD: Vectorized
result = jax.vmap(compute)(values)
```

## Git Workflow

### Branch Naming

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation
- `test/description` - Test improvements

### Commit Messages

Follow conventional commits:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

Example:
```
feat(physics): add phenology module

Implement seasonal deciduous phenology based on CTSM's CNPhenologyMod.
Includes onset/offset triggers and leaf area dynamics.

Closes #42
```

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Join our Slack/Discord (link TBD)

## Code of Conduct

Be respectful, inclusive, and professional. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).
