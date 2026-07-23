---
name: Bug report
about: Report incorrect results, a crash, or unexpected behavior
title: ""
labels: bug
assignees: ""
---

**Describe the bug**
A clear description of what went wrong.

**To reproduce**
A minimal, self-contained example (Python or Rust) that triggers the issue.
Please include the input shapes and any non-default parameters.

```python
import numpy as np
import rustcpd as cpd
# ...
```

**Expected behavior**
What you expected to happen instead.

**Environment**
- rustcpd version:
- Installed via: [ ] PyPI wheel  [ ] built from source  [ ] Rust crate
- OS and architecture:
- Python version (if applicable):

**Additional context**
Anything else that might help — cloud sizes, whether it's deterministic,
whether `normalize`/`low_rank`/`single_precision` change the outcome.
