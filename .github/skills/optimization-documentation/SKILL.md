# Optimization Documentation Skill

## Purpose
Generate standardized documentation for optimization changes, results tracking, and implementation checklists.

## Available Patterns

### 1. Change Template
Generate documentation for a single optimization change.

**Input**: Bottleneck description, location, current code, optimized code
**Output**: Formatted markdown following the standard change template

**Template**:
```markdown
## Change #X: [Concise Title]

**Issue**: Description of the bottleneck

**Location**: File path, line numbers

**Current Code**:
```python
[relevant snippet]
```

**Optimized Code**:
```python
[new optimized version, ≤20 lines]
```

**Expected Impact**: Speed improvement estimate

**Test Plan**: How to validate this change
```

### 2. Results Tracking
Document test results for optimization changes.

**Input**: Baseline metrics, change results, analysis
**Output**: Formatted markdown for results.md or mac-results.md

**Standardized Template**:
```markdown
## Change #X: [Concise Title]

**Status**: ✅ COMPLETED / ⚠️ REVERTED / ⚠️ INCONCLUSIVE

**Issue**: Description of the bottleneck

**Location**: File path, line numbers

**Current Code**:
```python
[relevant snippet]
```

**Optimized Code**:
```python
[new optimized version, ≤20 lines]
```

**Test Results**:
- Training: X.XXs/it → Y.YYs/it (Z% change)
- Samples: A.AAs/it → B.BBs/it (C% change)

**Analysis**: [detailed analysis of results]

**Verdict**: ✅ Keep / ⚠️ Revert / ⚠️ Monitor
```

### 3. Implementation Checklist
Generate checklist for tracking optimization progress.

**Input**: List of changes with status
**Output**: Markdown checklist following implementation-checklist.md format

## Usage

Invoke this skill when you need to:
- Document a new optimization change
- Record test results after validation
- Generate progress checklists

## Reference

See also: `.github/optimization-workflow.md` for detailed protocols and decision rules.
