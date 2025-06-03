# Intuitive Vector

An intuitive and versatile vector class (`Vec`) for numerical operations in Python. This library provides a `Vec` object that supports various arithmetic operations, norm calculations, distance metrics, and more, with a focus on clear and type-hinted code.

## Features

*   **Flexible Initialization**: Create vectors from numbers, sequences (lists, tuples), dictionaries (sparse vectors), or even other `Vec` objects.
*   **Dimension Control**: Explicitly set vector dimensions during initialization, with automatic padding (with zeros) or truncation.
*   **Comprehensive Type Hinting**: Enhances code clarity, enables static analysis, and improves developer experience.
*   **Rich Arithmetic Operations**:
    *   Element-wise: `+`, `-`, `*` (Hadamard product), `/` (true division), `//` (floor division), `%` (modulo), `^` (XOR for integer components).
    *   Scalar operations: `vector + scalar`, `scalar * vector`, etc.
*   **Vector-Specific Methods**:
    *   `norm()`: Euclidean (L2) norm, with an option for squared norm.
    *   `l1_norm()`: Manhattan (L1) norm.
    *   `hw()` or `wt()`: Hamming weight (count of non-zero elements), with optional index subset.
    *   `inner()`: Dot product (inner product).
    *   `abs()`: Returns a new vector with the absolute value of each element.
    *   `slice()`: Creates a new vector from elements at specified indices.
    *   `permute()`: Creates a new vector by reordering elements based on specified indices.
*   **Distance Metrics**:
    *   `euclid_dist()`: Euclidean distance between two vectors.
    *   `l1_dist()`: L1 (Manhattan) distance between two vectors.
    *   `hw_dist()`: Hamming distance between two vectors (assumes binary context after modulo 2).
*   **Comparison Operations**: `==`, `!=`, `<`, `<=`, `>`, `>=` (element-wise for ordering, requires same dimensions).
*   **In-place Modification**:
    *   `set_values()`: Versatile method to append, extend, or set values at specific indices.
    *   `<<` (pad to index): Extends the vector to ensure a given index exists, padding with zeros.
    *   `>>` (take last N): Truncates the vector to keep only the last N elements.
*   **Standard Pythonic Interface**:
    *   Supports `len()`, iteration, and `__getitem__` (for indexing and slicing).
    *   Clear `__repr__` for easy debugging.
*   **Customizable Printing**: `print()` method with options for grouping elements and custom separators.
*   **Robust Error Handling**: Raises appropriate `ValueError`, `TypeError`, `IndexError`, or `ZeroDivisionError` for invalid operations.

## Installation

Once published to PyPI, you will be able to install `intuitive-vector` using pip:
