import math
import collections.abc # For isinstance checks, good practice
from typing import List, Sequence, Mapping, Tuple, Union, Any, Iterator, Optional # Added more specific types

# Type alias for numeric types used in vector components
Number = Union[int, float]

def extend(values: List[Number], idx: int) -> None:
    """Extends the list 'values' with zeros if 'idx' is outside its current bounds."""
    if idx >= len(values):
        # Using 0.0 to be consistent as Number can be float
        values.extend([0.0] * (idx + 1 - len(values)))

class Vec(object):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Create a vector. Examples:
        v = Vec(1, 2) # -> [1.0, 2.0]
        v = Vec([1, 2, 3]) # -> [1.0, 2.0, 3.0]
        v = Vec(Vec(1,2), 3) # -> [1.0, 2.0, 3.0]
        v = Vec({0: 1, 2: 5}, dim=4) # -> [1.0, 0.0, 5.0, 0.0]
        v = Vec(([0, 2], 5), dim=3) # -> [5.0, 0.0, 5.0] (indices [0,2] get value 5)

        Accepted kwargs for dimension: 'dim', 'size', 'length', 'len'.
        The final dimension is the maximum of elements provided and any 'dim' kwarg.
        If 'dim' kwarg is smaller than elements provided, values are truncated.
        """
        self.values: List[Number] = []

        # Consolidate initialization logic here rather than calling self.set_values
        # which can be confusing during __init__
        processed_args = []
        for arg in args:
            if isinstance(arg, Vec):
                processed_args.extend(arg.values)
            elif isinstance(arg, collections.abc.Mapping): # e.g., {index: value, ...}
                # This needs careful handling to determine max index for pre-allocation
                # or dynamic extension. For simplicity, let's build a temporary list.
                temp_dict_values: List[Number] = []
                if arg: # Check if mapping is not empty
                    max_idx = -1
                    # Ensure keys are integers before finding max
                    int_keys = [k for k in arg.keys() if isinstance(k, int)]
                    if int_keys:
                        max_idx = max(int_keys)

                    if max_idx != -1:
                        extend(temp_dict_values, max_idx) # Pre-extend temp list
                    for k, v_item in arg.items():
                        if not isinstance(k, int) or k < 0:
                            raise TypeError(f"Dictionary keys for vector init must be non-negative integers, got {k}")
                        if not isinstance(v_item, (int, float)):
                            raise TypeError(f"Dictionary values for vector init must be numbers, got {type(v_item)}")
                        temp_dict_values[k] = float(v_item)
                processed_args.extend(temp_dict_values)
            elif isinstance(arg, tuple) and len(arg) == 2 and \
                 isinstance(arg[0], collections.abc.Sequence) and not isinstance(arg[0], str) and \
                 isinstance(arg[1], (int, float)): # Heuristic for pair: (indices, value)
                indices, value_item = arg
                if not all(isinstance(i, int) and i >= 0 for i in indices):
                    raise TypeError("Indices in pair must be non-negative integers.")
                # Similar to dict, build temporary list based on max index
                temp_pair_values: List[Number] = []
                if indices: # Check if indices sequence is not empty
                    max_idx = max(indices)
                    extend(temp_pair_values, max_idx)
                    for idx_item in indices:
                        temp_pair_values[idx_item] = float(value_item)
                processed_args.extend(temp_pair_values)
            elif isinstance(arg, collections.abc.Sequence) and not isinstance(arg, str): # e.g., list, tuple of numbers
                if not all(isinstance(x, (int, float)) for x in arg):
                    raise TypeError("Sequence elements for vector init must be numbers.")
                processed_args.extend(float(x) for x in arg)
            elif isinstance(arg, (int, float)): # Single number
                processed_args.append(float(arg))
            else:
                raise TypeError(f'Unsupported type {type(arg)} for vector initialization')

        self.values = processed_args # Assign the fully processed list

        # Determine dimension based on initialized values and kwargs
        current_len = len(self.values)
        dim_from_kwargs = 0
        for key in ['dim', 'size', 'length', 'len']:
            if key in kwargs:
                val = kwargs[key]
                if not isinstance(val, int):
                    raise TypeError(f"Dimension keyword '{key}' must be an integer, got {type(val)}")
                if val < 0:
                    raise ValueError(f"Dimension keyword '{key}' must be non-negative, got {val}")
                dim_from_kwargs = max(dim_from_kwargs, val)

        self.dim: int = max(current_len, dim_from_kwargs)

        if self.dim > current_len:
            extend(self.values, self.dim - 1) # Pad with 0.0
        elif self.dim < current_len: # Truncate if explicit dim is smaller
            self.values = self.values[:self.dim]


    def set_values(self,
                   val: Optional[Number] = None,
                   seq: Optional[Sequence[Number]] = None,
                   d: Optional[Mapping[int, Number]] = None,
                   p: Optional[Tuple[Sequence[int], Number]] = None) -> 'Vec':
        """
        Sets or extends vector values. Modifies the vector in-place.
        - val: Appends a single number.
        - seq: Extends with a sequence of numbers.
        - d (dict): Sets values at specified indices. {index: value}.
        - p (pair): Sets a single value at multiple specified indices. (indices, value).
        Updates self.dim to reflect the new length of self.values.
        """
        initial_len = len(self.values)
        max_idx_accessed = initial_len -1

        if val is not None:
            if not isinstance(val, (int, float)):
                raise TypeError(f"Value for 'val' must be a number, got {type(val)}")
            self.values.append(float(val))
        if seq is not None:
            if not (isinstance(seq, collections.abc.Sequence) and not isinstance(seq, str) and
                    all(isinstance(x, (int, float)) for x in seq)):
                raise TypeError("Value for 'seq' must be a sequence of numbers.")
            self.values.extend(float(x) for x in seq)
        if d is not None:
            if not isinstance(d, collections.abc.Mapping):
                raise TypeError("Value for 'd' must be a mapping.")
            for k, v_item in d.items():
                if not isinstance(k, int) or k < 0:
                    raise TypeError(f"Dictionary keys for setting values must be non-negative integers, got {k}")
                if not isinstance(v_item, (int, float)):
                    raise TypeError(f"Dictionary values for setting values must be numbers, got {type(v_item)}")
                extend(self.values, k)
                self.values[k] = float(v_item)
        if p is not None:
            if not (isinstance(p, tuple) and len(p) == 2 and
                    isinstance(p[0], collections.abc.Sequence) and not isinstance(p[0], str) and
                    all(isinstance(i, int) and i >= 0 for i in p[0]) and
                    isinstance(p[1], (int, float))):
                 raise TypeError("Value for 'p' must be a tuple of (sequence of non-negative ints, number).")
            indices, value_item = p
            if indices: # Check if indices sequence is not empty
                current_max_pair_idx = max(indices)
                extend(self.values, current_max_pair_idx)
            for idx_item in indices:
                self.values[idx_item] = float(value_item)

        self.dim = len(self.values) # Update dimension after modifications
        return self

    def norm(self, sqrt: bool = True) -> float:
        """ Returns the norm (length, magnitude) of the vector. """
        # sum of x*x can be float even if x are ints, math.sqrt returns float
        tmp: float = sum(x * x for x in self.values) # Iterate over self.values
        if sqrt:
            return math.sqrt(tmp)
        else:
            return tmp

    def hw(self, idxs: Optional[Sequence[int]] = None) -> int:
        """ Computes Hamming weight (number of non-zero elements). """
        target_values: Sequence[Number]
        if idxs is not None:
            if not (isinstance(idxs, collections.abc.Sequence) and not isinstance(idxs, str) and
                    all(isinstance(i, int) and 0 <= i < self.dim for i in idxs)):
                raise IndexError("Invalid indices for hw: must be a sequence of valid integers.")
            target_values = [self.values[i] for i in idxs]
        else:
            target_values = self.values
        return sum(1 for x in target_values if x != 0)

    def slice(self, idxs: Sequence[int]) -> 'Vec':
        """ Creates a new vector from a slice of elements at specified indices. """
        if not (isinstance(idxs, collections.abc.Sequence) and not isinstance(idxs, str) and
                all(isinstance(i, int) and 0 <= i < self.dim for i in idxs)):
            raise IndexError("All indices for slice must be valid integers within vector bounds.")
        return self.__class__([self.values[i] for i in idxs], dim=len(idxs))

    def permute(self, idxs: Sequence[int]) -> 'Vec':
        """ Creates a new vector by permuting elements according to specified indices. """
        if not (isinstance(idxs, collections.abc.Sequence) and not isinstance(idxs, str) and
                all(isinstance(i, int) and 0 <= i < self.dim for i in idxs)):
            raise IndexError("All indices for permute must be valid integers within vector bounds.")
        # A true permutation of length N would have len(idxs) == N and all unique indices from 0 to N-1
        # This implementation allows selecting elements, which is more like a generalized slice/reorder.
        # If a strict permutation is intended, more checks are needed (e.g., len(set(idxs)) == self.dim).
        return self.__class__([self.values[i] for i in idxs], dim=len(idxs))


    def wt(self, idxs: Optional[Sequence[int]] = None) -> int:
        """ Alias for Hamming weight (hw). """
        return self.hw(idxs)

    def _check_operand_compatibility(self, other: Any, operation_name: str, check_dim: bool = True) -> None:
        """Helper to check type and dimension for binary operations."""
        if not isinstance(other, Vec):
            # For operations strictly between Vec instances
            if not isinstance(other, (int, float)) and operation_name not in ["XOR with scalar", "scalar multiplication", "scalar division", "scalar modulo", "scalar addition", "scalar subtraction"]:
                 raise TypeError(f"{operation_name} requires a Vec instance or a scalar, got {type(other)}")
        if check_dim and isinstance(other, Vec) and self.dim != other.dim:
            raise ValueError(f"Vectors must have the same dimension for {operation_name}. Self: {self.dim}, Other: {other.dim}")


    def hw_dist(self, other: 'Vec') -> int:
        """ Computes Hamming distance to another vector. """
        self._check_operand_compatibility(other, "Hamming distance")
        # Assuming components become 0 or 1 after % 2 for typical hw_dist.
        diff_mod_2 = (self - other) % 2
        return diff_mod_2.wt()


    def euclid_dist(self, other: 'Vec') -> float:
        """ Computes Euclidean distance to another vector. """
        self._check_operand_compatibility(other, "Euclidean distance")
        return (self - other).norm()

    def l1_norm(self) -> Number:
        """ Computes L1 norm (Manhattan norm) of the vector. """
        return sum(abs(x) for x in self.values)

    def l1_dist(self, other: 'Vec') -> Number:
        """ Computes L1 distance (Manhattan distance) to another vector. """
        self._check_operand_compatibility(other, "L1 distance")
        return sum(abs(a - b) for a, b in zip(self.values, other.values))

    def inner(self, other: 'Vec') -> Number:
        """ Returns the dot product (inner product) of self and another vector. """
        # Corrected check: raise if NOT a Vec instance
        if not isinstance(other, Vec):
            raise TypeError('The dot product requires another Vec instance.')
        self._check_operand_compatibility(other, "inner product")
        return sum(a * b for a, b in zip(self.values, other.values))

    def __xor__(self, other: Union['Vec', int]) -> 'Vec':
        """ Returns the element-wise XOR of self and other. """
        xored_values: List[Number] # Use list for mutability, then tuple for constructor
        if isinstance(other, Vec):
            self._check_operand_compatibility(other, "element-wise XOR")
            xored_values = [int(a) ^ int(b) for a, b in zip(self.values, other.values)]
        elif isinstance(other, int):
            xored_values = [int(a) ^ other for a in self.values]
        else:
            raise TypeError(f"XOR with type {type(other)} not supported (must be Vec or int).")
        return self.__class__(xored_values, dim=self.dim)

    def __mul__(self, other: Union['Vec', Number]) -> 'Vec':
        """
        Element-wise multiplication if multiplied by another Vector (Hadamard product).
        Scalar multiplication if multiplied by an int or float.
        """
        product_values: List[Number]
        if isinstance(other, Vec):
            self._check_operand_compatibility(other, "element-wise multiplication")
            product_values = [a * b for a, b in zip(self.values, other.values)]
        elif isinstance(other, (int, float)):
            product_values = [a * other for a in self.values]
        else:
            raise TypeError(f"Multiplication with type {type(other)} not supported.")
        return self.__class__(product_values, dim=self.dim)

    def __rmul__(self, other: Number) -> 'Vec':
        """ Called for scalar * self. """
        if not isinstance(other, (int, float)): # Should be Number
            raise TypeError(f"Scalar multiplication requires a number, got {type(other)}")
        return self.__mul__(other)

    def __truediv__(self, other: Union['Vec', Number]) -> 'Vec':
        """ Element-wise true division or scalar division. Returns floats. """
        divided_values: List[float]
        if isinstance(other, Vec):
            self._check_operand_compatibility(other, "element-wise division")
            if any(b == 0 for b in other.values):
                raise ZeroDivisionError("Vector element-wise division by zero.")
            divided_values = [a / b for a, b in zip(self.values, other.values)]
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Scalar division by zero.")
            divided_values = [a / other for a in self.values]
        else:
            raise TypeError(f"Division with type {type(other)} not supported.")
        return self.__class__(divided_values, dim=self.dim)

    def __rtruediv__(self, other: Number) -> 'Vec':
        """ Called for scalar / self (element-wise). """
        if not isinstance(other, (int, float)):
            raise TypeError(f"Left operand for division must be a number, got {type(other)}")
        if any(val == 0 for val in self.values):
            raise ZeroDivisionError("Element-wise division by zero in vector (scalar / vector).")
        divided_values = [other / val for val in self.values]
        return self.__class__(divided_values, dim=self.dim)


    def __floordiv__(self, other: Union['Vec', int]) -> 'Vec': # Typically int context
        """ Element-wise floor division or scalar floor division. """
        divided_values: List[Number]
        if isinstance(other, Vec):
            self._check_operand_compatibility(other, "element-wise floor division")
            if any(b == 0 for b in other.values):
                raise ZeroDivisionError("Vector element-wise floor division by zero.")
            # Ensure operands are suitable for floor division (typically int or float that results in int-like)
            divided_values = [a // b for a, b in zip(self.values, other.values)]
        elif isinstance(other, int): # Floor division with float scalar is less common, stick to int
            if other == 0:
                raise ZeroDivisionError("Scalar floor division by zero.")
            divided_values = [a // other for a in self.values]
        else:
            raise TypeError(f"Floor division with type {type(other)} not supported (must be Vec or int).")
        return self.__class__(divided_values, dim=self.dim)

    def __rfloordiv__(self, other: int) -> 'Vec': # Typically int context
        """ Called for integer_scalar // self (element-wise). """
        if not isinstance(other, int):
            raise TypeError(f"Left operand for floor division must be an integer, got {type(other)}")
        if any(val == 0 for val in self.values):
            raise ZeroDivisionError("Element-wise floor division by zero in vector (scalar // vector).")
        divided_values = [other // val for val in self.values]
        return self.__class__(divided_values, dim=self.dim)


    def __add__(self, other: Union['Vec', Number]) -> 'Vec':
        """ Element-wise addition or scalar addition. """
        added_values: List[Number]
        if isinstance(other, Vec):
            self._check_operand_compatibility(other, "addition")
            added_values = [a + b for a, b in zip(self.values, other.values)]
        elif isinstance(other, (int, float)):
            added_values = [a + other for a in self.values]
        else:
            raise TypeError(f"Addition with type {type(other)} not supported.")
        return self.__class__(added_values, dim=self.dim)

    def __radd__(self, other: Number) -> 'Vec':
        """ Called for scalar + self. """
        if not isinstance(other, (int, float)):
            raise TypeError(f"Scalar addition requires a number, got {type(other)}")
        return self.__add__(other)

    def __sub__(self, other: Union['Vec', Number]) -> 'Vec':
        """ Element-wise subtraction or scalar subtraction. """
        subtracted_values: List[Number]
        if isinstance(other, Vec):
            self._check_operand_compatibility(other, "subtraction")
            subtracted_values = [a - b for a, b in zip(self.values, other.values)]
        elif isinstance(other, (int, float)):
            subtracted_values = [a - other for a in self.values]
        else:
            raise TypeError(f"Subtraction with type {type(other)} not supported.")
        return self.__class__(subtracted_values, dim=self.dim)

    def __rsub__(self, other: Number) -> 'Vec':
        """ Called for scalar - self. """
        if not isinstance(other, (int, float)):
            raise TypeError(f"Left operand for subtraction must be a number, got {type(other)}")
        subtracted_values = [other - val for val in self.values]
        return self.__class__(subtracted_values, dim=self.dim)

    def __iter__(self) -> Iterator[Number]:
        return iter(self.values)

    def __len__(self) -> int:
        return self.dim # Should be consistent with len(self.values)

    def __getitem__(self, key: Union[int, slice]) -> Union[Number, List[Number]]:
        if isinstance(key, int):
            # Handle negative indices correctly relative to self.dim
            if key < -self.dim or key >= self.dim :
                 raise IndexError("Vector index out of range")
            return self.values[key] # Python list handles negative indexing from -len(list)
        elif isinstance(key, slice):
            # Slicing a Vec should probably return a new Vec, not a list
            # For now, matches original behavior of returning list slice
            return self.values[key]
        else:
            raise TypeError("Vector indices must be integers or slices.")

    def __repr__(self) -> str:
        # A more standard representation
        return f"{self.__class__.__name__}({self.values})"


    def __mod__(self, other: Union['Vec', Number]) -> 'Vec':
        """ Element-wise modulo or scalar modulo. """
        remainder_values: List[Number]
        if isinstance(other, Vec):
            self._check_operand_compatibility(other, "modulo")
            if any(b == 0 for b in other.values): # Check for division by zero
                raise ZeroDivisionError("Vector element-wise modulo by zero.")
            remainder_values = [a % b for a, b in zip(self.values, other.values)]
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Scalar modulo by zero.")
            if isinstance(other, float): # Use math.fmod for floats for C-like behavior
                remainder_values = [math.fmod(a, other) for a in self.values]
            else: # int
                remainder_values = [a % other for a in self.values]
        else:
            raise TypeError(f"Modulo with type {type(other)} not supported.")
        return self.__class__(remainder_values, dim=self.dim)

    def __rmod__(self, other: Number) -> 'Vec':
        """ Called for scalar % self. """
        if not isinstance(other, (int, float)):
            raise TypeError(f"Left operand for modulo must be a number, got {type(other)}")
        if any(val == 0 for val in self.values):
            raise ZeroDivisionError("Element-wise modulo by zero in vector (scalar % vector).")

        if isinstance(other, float):
            remainder_values = [math.fmod(other, val) for val in self.values]
        else: # int
            remainder_values = [other % val for val in self.values]
        return self.__class__(remainder_values, dim=self.dim)


    def abs(self) -> 'Vec':
        """ Returns a new vector with the absolute value of each element. """
        return self.__class__([abs(a) for a in self.values], dim=self.dim)

    def _compare_elementwise(self, other: Any, op_func) -> bool:
        if not isinstance(other, Vec):
            return NotImplemented # Allow other types to define comparison
        if self.dim != other.dim:
            # Or raise ValueError("Vectors must have the same dimension for element-wise comparison.")
            # For __eq__ and __ne__, different dims means not equal.
            # For <, <=, >, >=, it's often an error or undefined.
            # Current code implies it's False if dims differ, which might be okay for some definitions.
            # Let's make it stricter for ordering comparisons.
            if op_func.__name__ not in ['<lambda>_eq', '<lambda>_ne']: # Hacky way to check if it's for eq/ne
                 raise ValueError("Vectors must have the same dimension for ordering comparisons.")
            return False
        return all(op_func(a, b) for a, b in zip(self.values, other.values))

    def __le__(self, other: 'Vec') -> bool: return self._compare_elementwise(other, lambda a,b: a <= b)
    def __lt__(self, other: 'Vec') -> bool: return self._compare_elementwise(other, lambda a,b: a < b)
    def __gt__(self, other: 'Vec') -> bool: return self._compare_elementwise(other, lambda a,b: a > b)
    def __ge__(self, other: 'Vec') -> bool: return self._compare_elementwise(other, lambda a,b: a >= b)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vec):
            return False # Or NotImplemented
        if self.dim != other.dim:
            return False
        # Element-wise equality, considering potential float precision issues if relevant
        # For exact float comparison, math.isclose might be needed in some contexts.
        # Here, direct list comparison is fine as per original.
        return self.values == other.values

    def __ne__(self, other: object) -> bool:
        # Best practice: define __ne__ in terms of __eq__
        eq_result = self.__eq__(other)
        return not eq_result if eq_result is not NotImplemented else NotImplemented


    def __lshift__(self, other: int) -> 'Vec':
        """
        Extends the vector to ensure it has at least 'other + 1' elements (0-indexed),
        padding with zeros if necessary. Modifies the vector in-place and returns self.
        This is an unusual overload for <<.
        """
        if not isinstance(other, int) or other < 0:
            raise ValueError("Shift amount for __lshift__ (pad to index) must be a non-negative integer.")
        extend(self.values, other) # 'other' is treated as the target maximum index
        self.dim = len(self.values)
        return self

    def __ilshift__(self, other: int) -> 'Vec':
        return self.__lshift__(other)

    def __rshift__(self, other: int) -> 'Vec':
        """
        Keeps the last 'other' elements of the vector, effectively truncating from the beginning.
        Modifies the vector in-place and returns self.
        This is an unusual overload for >>.
        """
        if not isinstance(other, int) or other < 0:
            raise ValueError("Shift amount for __rshift__ (take last N) must be a non-negative integer.")
        if other > self.dim:
            # Taking more than available means taking all, which means no change if other >= self.dim
            # Or, if it means "result has length 'other' padded from left", that's different.
            # Current: self.values[-other:] will correctly take all if other >= len
            pass
        self.values = self.values[-other:]
        self.dim = len(self.values)
        return self

    def __irshift__(self, other: int) -> 'Vec':
        return self.__rshift__(other)

    def print(self, grouping: int = 1, sep: str = ',') -> None:
        """ Prints the vector's values, with optional grouping and separator. """
        if not isinstance(grouping, int) or grouping <= 0:
            raise ValueError("Grouping for print must be a positive integer.")
        if not isinstance(sep, str):
            raise TypeError("Separator for print must be a string.")

        # Create string representations of numbers first
        str_values = [str(v) for v in self.values]

        if not str_values and self.dim > 0: # Handle empty vector with a dimension
            str_values = [str(0.0)] * self.dim # Or based on default fill value

        grouped_strs = []
        for i in range(0, len(str_values), grouping):
            grouped_strs.append("".join(str_values[i:i + grouping]))
        print(sep.join(grouped_strs))


if __name__ == "__main__":
    print("--- Vector Initialization ---")
    v_empty = Vec(dim=3)
    print(f"Empty vector with dim 3: {v_empty!r}, Length: {len(v_empty)}") # Vec([0.0, 0.0, 0.0])

    v1 = Vec(1, 2.5, 3)
    print(f"v1 (from numbers): {v1!r}, Length: {len(v1)}") # Vec([1.0, 2.5, 3.0])

    v2 = Vec([4, 5, 6])
    print(f"v2 (from list): {v2!r}") # Vec([4.0, 5.0, 6.0])

    v_from_v = Vec(v1, 7, 8)
    print(f"v_from_v (from v1 and numbers): {v_from_v!r}") # Vec([1.0, 2.5, 3.0, 7.0, 8.0])

    v_dict = Vec({0: 10, 2: 30, 1: 20}, dim=4) # Specify dim
    print(f"v_dict (from dict with dim): {v_dict!r}") # Vec([10.0, 20.0, 30.0, 0.0])

    v_pair = Vec(([0, 3], 5), 99, dim=5) # ([indices], value)
    print(f"v_pair (from pair and number with dim): {v_pair!r}") # Vec([5.0, 99.0, 0.0, 5.0, 0.0])

    v_trunc = Vec(1,2,3,4,5, dim=3)
    print(f"v_trunc (truncated by dim): {v_trunc!r}") # Vec([1.0, 2.0, 3.0])

    print("\n--- Accessing Elements ---")
    print(f"v1[0]: {v1[0]}")
    print(f"v1[1]: {v1[1]}")
    # print(f"v1[3]: {v1[3]}") # IndexError
    print(f"v1[-1]: {v1[-1]}") # Last element
    print(f"v2[0:2]: {v2[0:2]}") # Slicing returns a list: [4.0, 5.0]

    print("\n--- set_values Method ---")
    v_set = Vec(1, 2)
    print(f"Original v_set: {v_set!r}")
    v_set.set_values(val=3)
    print(f"After set_values(val=3): {v_set!r}")
    v_set.set_values(seq=[4, 5])
    print(f"After set_values(seq=[4,5]): {v_set!r}")
    v_set.set_values(d={0: 100, 5: 500}) # Extends and sets
    print(f"After set_values(d={{0:100, 5:500}}): {v_set!r}, Dim: {v_set.dim}")
    v_set.set_values(p=([1, 6], 777)) # (indices, value)
    print(f"After set_values(p=([1,6], 777)): {v_set!r}, Dim: {v_set.dim}")


    print("\n--- Arithmetic Operations ---")
    v_a = Vec(1, 2, 3)
    v_b = Vec(4, 5, 6)
    print(f"v_a: {v_a!r}")
    print(f"v_b: {v_b!r}")

    print(f"v_a + v_b: {v_a + v_b!r}")
    print(f"v_a + 10: {v_a + 10!r}")
    print(f"10 + v_a: {10 + v_a!r}")

    print(f"v_a - v_b: {v_a - v_b!r}")
    print(f"v_a - 1: {v_a - 1!r}")
    print(f"10 - v_a: {10 - v_a!r}") # Vec([9.0, 8.0, 7.0])

    print(f"v_a * v_b (Hadamard): {v_a * v_b!r}")
    print(f"v_a * 2: {v_a * 2!r}")
    print(f"2 * v_a: {2 * v_a!r}")

    v_c = Vec(2, 4, 6)
    print(f"v_c: {v_c!r}")
    print(f"v_c / 2: {v_c / 2!r}")
    print(f"v_b / v_a : {v_b / v_a!r}") # Vec([4.0, 2.5, 2.0])
    print(f"12 / v_c: {12 / v_c!r}") # Vec([6.0, 3.0, 2.0])
    # print(f"v_a / Vec(1,0,1): ") # ZeroDivisionError

    print(f"v_c // 2: {v_c // 2!r}")
    print(f"v_b // v_a: {v_b // v_a!r}") # Vec([4.0, 2.0, 2.0])
    print(f"13 // v_a: {13 // v_a!r}") # Vec([13.0, 6.0, 4.0])

    v_mod1 = Vec(5, 6, 7)
    v_mod2 = Vec(2, 3, 4)
    print(f"v_mod1 % v_mod2: {v_mod1 % v_mod2!r}") # Vec([1.0, 0.0, 3.0])
    print(f"v_mod1 % 3: {v_mod1 % 3!r}") # Vec([2.0, 0.0, 1.0])
    print(f"7 % Vec(2,3,4): {7 % Vec(2,3,4)!r}") # Vec([1.0, 1.0, 3.0])

    v_xor1 = Vec(1, 0, 1, 0) # Typically binary for XOR
    v_xor2 = Vec(1, 1, 0, 0)
    print(f"v_xor1 ^ v_xor2: {v_xor1 ^ v_xor2!r}") # Vec([0.0, 1.0, 1.0, 0.0])
    print(f"v_xor1 ^ 1: {v_xor1 ^ 1!r}") # Vec([0.0, 1.0, 0.0, 1.0])

    print("\n--- Vector Specific Methods ---")
    v_norm_test = Vec(3, 4)
    print(f"v_norm_test: {v_norm_test!r}")
    print(f"Norm of v_norm_test: {v_norm_test.norm()}") # 5.0
    print(f"Norm squared of v_norm_test: {v_norm_test.norm(sqrt=False)}") # 25.0

    v_hw_test = Vec(1, 0, 7, 0, -2, 0.0)
    print(f"v_hw_test: {v_hw_test!r}")
    print(f"Hamming weight (hw) of v_hw_test: {v_hw_test.hw()}") # 3
    print(f"Hamming weight (wt) of v_hw_test: {v_hw_test.wt()}") # 3
    print(f"Hamming weight of v_hw_test at indices [0, 2, 4]: {v_hw_test.hw(idxs=[0, 2, 4])}") # 3

    v_slice_perm = Vec(10, 20, 30, 40, 50)
    print(f"v_slice_perm: {v_slice_perm!r}")
    print(f"Slice v_slice_perm at [0, 2, 4]: {v_slice_perm.slice([0, 2, 4])!r}") # Vec([10.0, 30.0, 50.0])
    print(f"Permute v_slice_perm with [4, 3, 0, 1, 2]: {v_slice_perm.permute([4, 3, 0, 1, 2])!r}") # Vec([50.0, 40.0, 10.0, 20.0, 30.0])

    v_dist1 = Vec(1, 1, 0, 0)
    v_dist2 = Vec(1, 0, 1, 0)
    print(f"v_dist1: {v_dist1!r}")
    print(f"v_dist2: {v_dist2!r}")
    print(f"Hamming distance between v_dist1 and v_dist2: {v_dist1.hw_dist(v_dist2)}") # 2

    v_euc1 = Vec(1, 2)
    v_euc2 = Vec(4, 6)
    print(f"v_euc1: {v_euc1!r}")
    print(f"v_euc2: {v_euc2!r}")
    print(f"Euclidean distance between v_euc1 and v_euc2: {v_euc1.euclid_dist(v_euc2)}") # 5.0

    v_l1 = Vec(-1, 2, -3)
    print(f"v_l1: {v_l1!r}")
    print(f"L1 norm of v_l1: {v_l1.l1_norm()}") # 6.0
    v_l1_other = Vec(1,1,1)
    print(f"L1 distance between v_l1 and {v_l1_other!r}: {v_l1.l1_dist(v_l1_other)}") # 2+1+4 = 7.0

    v_dot1 = Vec(1, 2, 3)
    v_dot2 = Vec(4, 5, 6)
    print(f"Inner product of {v_dot1!r} and {v_dot2!r}: {v_dot1.inner(v_dot2)}") # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32.0

    print(f"Absolute value of {v_l1!r}: {v_l1.abs()!r}") # Vec([1.0, 2.0, 3.0])

    print("\n--- Comparison Operations ---")
    comp1 = Vec(1, 2, 3)
    comp2 = Vec(1, 2, 3)
    comp3 = Vec(1, 2, 4)
    comp4 = Vec(0, 3, 2)
    print(f"comp1: {comp1!r}, comp2: {comp2!r}, comp3: {comp3!r}, comp4: {comp4!r}")
    print(f"comp1 == comp2: {comp1 == comp2}") # True
    print(f"comp1 == comp3: {comp1 == comp3}") # False
    print(f"comp1 != comp3: {comp1 != comp3}") # True
    print(f"comp1 < comp3: {comp1 < comp3}")   # True (element-wise all true)
    print(f"comp1 <= comp2: {comp1 <= comp2}") # True
    print(f"comp3 > comp1: {comp3 > comp1}")   # True
    # print(f"comp1 < Vec(1,2): ") # ValueError (different dimensions for ordering)
    print(f"Vec(1,2) == Vec(1,2,0): {Vec(1,2) == Vec(1,2,0)}") # False (different dimensions)


    print("\n--- Shift-like Operations (Padding/Truncating) ---")
    v_shift = Vec(1, 2, 3)
    print(f"Original v_shift: {v_shift!r}")
    v_shift << 4 # Pad to ensure index 4 exists (i.e., length 5)
    print(f"After v_shift << 4: {v_shift!r}, Dim: {v_shift.dim}") # Vec([1.0, 2.0, 3.0, 0.0, 0.0])

    v_shift2 = Vec(10,20,30,40,50)
    print(f"Original v_shift2: {v_shift2!r}")
    v_shift2 >> 3 # Keep last 3 elements
    print(f"After v_shift2 >> 3: {v_shift2!r}, Dim: {v_shift2.dim}") # Vec([30.0, 40.0, 50.0])
    v_shift2 >> 5 # Keep last 5 (more than available, keeps all remaining)
    print(f"After v_shift2 >> 5: {v_shift2!r}, Dim: {v_shift2.dim}") # Vec([30.0, 40.0, 50.0])


    print("\n--- Custom Print Method ---")
    v_print_test = Vec(1,2,3,4,5,6,7,8,9)
    print("v_print_test.print(grouping=3, sep='-'):")
    v_print_test.print(grouping=3, sep='-') # 123-456-789
    print("v_print_test.print(grouping=1, sep=' '):")
    v_print_test.print(grouping=1, sep=' ') # 1 2 3 4 5 6 7 8 9
    print("v_print_test.print():")
    v_print_test.print() # 1,2,3,4,5,6,7,8,9

    print("\n--- Edge Cases / Error Handling (Illustrative) ---")
    try:
        Vec(1, 2) + Vec(1, 2, 3)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    try:
        Vec(1,0,1) / Vec(1,0,1) # This will cause error due to 0/0 if not handled, but /0 is the direct error
    except ZeroDivisionError as e:
         print(f"Caught expected error for division: {e}")

    try:
        Vec(1,2,3)[5]
    except IndexError as e:
        print(f"Caught expected error for getitem: {e}")

    try:
        Vec("a", "b")
    except TypeError as e:
        print(f"Caught expected error for init with strings: {e}")