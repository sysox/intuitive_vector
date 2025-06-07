# C:/Users/user/PycharmProjects/intuitive_vector/src/intuitive_vector/__init__.py

# This file makes 'intuitive_vector' a package.

# You can expose key components from your modules at the package level
# for easier imports. For example, to allow users to do:
# from intuitive_vector import Vec, Number
# instead of:
# from intuitive_vector.vector import Vec, Number

from .vector import Vec, Number

# It's also a common practice to define the package version here.
__version__ = "0.1.0"  # Or your desired version

# The __all__ variable defines the public API of the package when
# 'from intuitive_vector import *' is used.
__all__ = [
    "Vec",
    "Number",
    "__version__",
]