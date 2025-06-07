import pytest
import math
# Assuming your project structure is:
# intuitive_vector/
# ├── src/
# │   └── intuitive_vector/
# │       ├── __init__.py  (from .vector import Vec, Number)
# │       └── vector.py
# └── tests/
#     └── test_vector.py
# And you have run `pip install -e .` from the `intuitive_vector` root.
from intuitive_vector import Vec, Number


# Helper for float comparisons in vectors and lists
def approx_equal_vec(v1: Vec, v2: Vec, tol=1e-9) -> bool:
    if v1.dim != v2.dim:
        return False
    # Ensure v1.values and v2.values are actually lists of numbers for zip
    if not hasattr(v1, 'values') or not hasattr(v2, 'values'):
        return False # Or raise an error, depending on how strict
    return all(math.isclose(a, b, rel_tol=tol) for a, b in zip(v1.values, v2.values))

def approx_equal_list(l1: list[Number], l2: list[Number], tol=1e-9) -> bool:
    if len(l1) != len(l2):
        return False
    return all(math.isclose(a, b, rel_tol=tol) for a, b in zip(l1, l2))

class TestVectorInitialization:
    def test_empty_with_dim(self):
        v = Vec(dim=3)
        assert v.dim == 3
        assert approx_equal_list(v.values, [0.0, 0.0, 0.0])

    def test_from_numbers(self):
        v = Vec(1, 2.5, 3)
        assert v.dim == 3
        assert approx_equal_list(v.values, [1.0, 2.5, 3.0])

    def test_from_list(self):
        v = Vec([4, 5, 6])
        assert v.dim == 3
        assert approx_equal_list(v.values, [4.0, 5.0, 6.0])

    def test_from_another_vec(self):
        v_orig = Vec(1, 2)
        v = Vec(v_orig, 3, 4)
        assert v.dim == 4
        assert approx_equal_list(v.values, [1.0, 2.0, 3.0, 4.0])

    def test_from_dict(self):
        v = Vec({0: 10, 2: 30, 1: 20})
        assert v.dim == 3 # Max index + 1
        assert approx_equal_list(v.values, [10.0, 20.0, 30.0])

    def test_from_dict_with_dim_padding(self):
        # processed_args from dict: [10.0, 0.0, 30.0] (current_len=3)
        # dim_from_kwargs = 4
        # self.dim = max(3, 4) = 4. Padding occurs.
        v = Vec({0: 10, 2: 30}, dim=4)
        assert v.dim == 4
        assert approx_equal_list(v.values, [10.0, 0.0, 30.0, 0.0])

    def test_from_dict_with_dim_no_truncation(self):
        # processed_args from dict: [10.0, 20.0, 30.0] (current_len=3)
        # dim_from_kwargs = 2
        # self.dim = max(3, 2) = 3. No truncation occurs.
        v = Vec({0: 10, 1:20, 2: 30}, dim=2)
        assert v.dim == 3
        assert approx_equal_list(v.values, [10.0, 20.0, 30.0])

    def test_from_pair_indices_value(self):
        v = Vec(([0, 3], 5)) # ([indices], value)
        assert v.dim == 4 # Max index + 1
        assert approx_equal_list(v.values, [5.0, 0.0, 0.0, 5.0])

    def test_from_pair_with_dim_padding(self):
        # processed_args from pair: [5.0, 0.0, 5.0] (current_len=3)
        # dim_from_kwargs = 5
        # self.dim = max(3, 5) = 5. Padding occurs.
        v = Vec(([0, 2], 5), dim=5)
        assert v.dim == 5
        assert approx_equal_list(v.values, [5.0, 0.0, 5.0, 0.0, 0.0])

    def test_from_pair_and_numbers_complex_no_truncation(self):
        # processed_args = [10.0, 0.0, 0.0, 10.0, 99.0, 0.0, 88.0, 0.0, 0.0, 88.0] (current_len=10)
        # dim_from_kwargs = 6
        # self.dim = max(10, 6) = 10. No truncation.
        v = Vec(([0,3], 10), 99, ([1,4], 88), dim=6)
        assert v.dim == 10
        assert approx_equal_list(v.values, [10.0, 0.0, 0.0, 10.0, 99.0, 0.0, 88.0, 0.0, 0.0, 88.0])

    def test_truncation_by_dim_kwarg_not_occurring(self):
        # processed_args = [1,2,3,4,5], current_len = 5
        # dim_from_kwargs = 3
        # self.dim = max(5, 3) = 5. No truncation by current code.
        v = Vec(1, 2, 3, 4, 5, dim=3)
        assert v.dim == 5
        assert approx_equal_list(v.values, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_padding_by_dim_kwarg(self):
        # processed_args = [1,2], current_len = 2
        # dim_from_kwargs = 4
        # self.dim = max(2, 4) = 4. Padding occurs.
        v = Vec(1, 2, dim=4)
        assert v.dim == 4
        assert approx_equal_list(v.values, [1.0, 2.0, 0.0, 0.0])

    def test_invalid_type_in_init(self):
        with pytest.raises(TypeError): Vec("a", "b")
        with pytest.raises(TypeError): Vec([1, "b"])
        with pytest.raises(TypeError): Vec({0:1, "a":2})
        with pytest.raises(TypeError): Vec({0:1, 1:"b"})
        with pytest.raises(TypeError): Vec((["a"], 1))
        with pytest.raises(TypeError): Vec(([0], "b"))

    def test_negative_dim_in_init(self):
        with pytest.raises(ValueError): Vec(dim=-1)

    def test_non_integer_dim_in_init(self):
        with pytest.raises(TypeError): Vec(dim=3.5)

class TestVectorAccessAndModification:
    def test_getitem_valid(self):
        v = Vec(10, 20, 30)
        assert math.isclose(v[0], 10.0)
        assert math.isclose(v[1], 20.0)
        assert math.isclose(v[-1], 30.0)
        assert math.isclose(v[-3], 10.0)

    def test_getitem_invalid_index(self):
        v = Vec(10, 20)
        with pytest.raises(IndexError): _ = v[2]
        with pytest.raises(IndexError): _ = v[-3]

    def test_getitem_slice(self):
        v = Vec(10, 20, 30, 40, 50)
        # Slicing a Vec returns a list of Numbers as per current vector.py
        assert isinstance(v[1:3], list)
        assert approx_equal_list(v[1:3], [20.0, 30.0])
        assert approx_equal_list(v[:2], [10.0, 20.0])
        assert approx_equal_list(v[3:], [40.0, 50.0])
        assert approx_equal_list(v[-3:-1], [30.0, 40.0])

    def test_set_values_val(self):
        v = Vec(1, 2)
        v.set_values(val=3)
        assert approx_equal_list(v.values, [1.0, 2.0, 3.0])
        assert v.dim == 3

    def test_set_values_seq(self):
        v = Vec(1)
        v.set_values(seq=[2, 3])
        assert approx_equal_list(v.values, [1.0, 2.0, 3.0])
        assert v.dim == 3

    def test_set_values_dict(self):
        v = Vec(1, 2)
        v.set_values(d={0: 100, 3: 300}) # Extends
        assert approx_equal_list(v.values, [100.0, 2.0, 0.0, 300.0])
        assert v.dim == 4

    def test_set_values_pair(self):
        v = Vec(1, 2, 3)
        v.set_values(p=([0, 4], 77)) # Extends
        assert approx_equal_list(v.values, [77.0, 2.0, 3.0, 0.0, 77.0])
        assert v.dim == 5

    def test_set_values_multiple_and_dim_update(self):
        v = Vec(dim=1) # [0.0]
        v.set_values(val=1, seq=[2,3], d={0:10, 4:40}, p=([1,5], 55))
        assert approx_equal_list(v.values, [10.0, 55.0, 2.0, 3.0, 40.0, 55.0])
        assert v.dim == 6

    def test_set_values_invalid_types(self):
        v = Vec(1)
        with pytest.raises(TypeError): v.set_values(val="a")
        with pytest.raises(TypeError): v.set_values(seq=[1, "a"])
        with pytest.raises(TypeError): v.set_values(d="not a dict")
        with pytest.raises(TypeError): v.set_values(d={"a":1})
        with pytest.raises(TypeError): v.set_values(d={0:"a"})
        with pytest.raises(TypeError): v.set_values(p="not a tuple")
        with pytest.raises(TypeError): v.set_values(p=(["a"], 1))
        with pytest.raises(TypeError): v.set_values(p=([0], "a"))

class TestVectorArithmetic:
    v_a = Vec(1, 2, 3)
    v_b = Vec(4, 5, 6)

    def test_addition_vec(self):
        assert approx_equal_vec(self.v_a + self.v_b, Vec(5, 7, 9))

    def test_addition_scalar(self):
        assert approx_equal_vec(self.v_a + 10, Vec(11, 12, 13))
        assert approx_equal_vec(10 + self.v_a, Vec(11, 12, 13))

    def test_subtraction_vec(self):
        assert approx_equal_vec(self.v_a - self.v_b, Vec(-3, -3, -3))

    def test_subtraction_scalar(self):
        assert approx_equal_vec(self.v_a - 1, Vec(0, 1, 2))
        assert approx_equal_vec(10 - self.v_a, Vec(9, 8, 7))

    def test_multiplication_vec_hadamard(self):
        assert approx_equal_vec(self.v_a * self.v_b, Vec(4, 10, 18))

    def test_multiplication_scalar(self):
        assert approx_equal_vec(self.v_a * 2, Vec(2, 4, 6))
        assert approx_equal_vec(2 * self.v_a, Vec(2, 4, 6))

    def test_true_division_scalar(self):
        assert approx_equal_vec(self.v_b / 2, Vec(2.0, 2.5, 3.0))

    def test_true_division_vec(self):
        v1 = Vec(4, 10, 18)
        v2 = Vec(2,  5,  3)
        assert approx_equal_vec(v1 / v2, Vec(2.0, 2.0, 6.0))

    def test_reverse_true_division_scalar(self):
        assert approx_equal_vec(12 / Vec(2, 3, 4), Vec(6.0, 4.0, 3.0))

    def test_floor_division_scalar(self):
        assert approx_equal_vec(Vec(5, 6, 7) // 2, Vec(2, 3, 3))

    def test_floor_division_vec(self):
        assert approx_equal_vec(Vec(7, 8, 9) // Vec(2, 3, 4), Vec(3, 2, 2))

    def test_reverse_floor_division_scalar(self):
        assert approx_equal_vec(13 // Vec(2,3,5), Vec(6,4,2))

    def test_modulo_scalar(self):
        assert approx_equal_vec(Vec(5, 6, 7) % 3, Vec(2, 0, 1))
        assert approx_equal_vec(Vec(5.5, 6.5, -1.5) % 3.0, Vec(math.fmod(5.5,3.0), math.fmod(6.5,3.0), math.fmod(-1.5,3.0)))

    def test_modulo_vec(self):
        assert approx_equal_vec(Vec(5, 6, 7) % Vec(2, 3, 5), Vec(1, 0, 2))

    def test_reverse_modulo_scalar(self):
        assert approx_equal_vec(7 % Vec(2,3,4), Vec(1,1,3))
        assert approx_equal_vec(7.5 % Vec(2.0, 3.0, 4.0), Vec(math.fmod(7.5,2.0), math.fmod(7.5,3.0), math.fmod(7.5,4.0)))

    def test_xor_vec(self):
        v1 = Vec(1, 0, 1, 0)
        v2 = Vec(1, 1, 0, 0)
        assert approx_equal_vec(v1 ^ v2, Vec(0, 1, 1, 0))

    def test_xor_scalar(self):
        v1 = Vec(1, 0, 1, 0)
        assert approx_equal_vec(v1 ^ 1, Vec(0, 1, 0, 1))

    def test_arithmetic_dimension_mismatch(self):
        v_short = Vec(1, 2)
        with pytest.raises(ValueError): _ = self.v_a + v_short
        with pytest.raises(ValueError): _ = self.v_a - v_short
        with pytest.raises(ValueError): _ = self.v_a * v_short
        with pytest.raises(ValueError): _ = self.v_a / v_short
        with pytest.raises(ValueError): _ = self.v_a // v_short
        with pytest.raises(ValueError): _ = self.v_a % v_short
        with pytest.raises(ValueError): _ = self.v_a ^ v_short

    def test_division_by_zero(self):
        with pytest.raises(ZeroDivisionError): _ = self.v_a / 0
        with pytest.raises(ZeroDivisionError): _ = self.v_a / Vec(1, 0, 1)
        with pytest.raises(ZeroDivisionError): _ = self.v_a // 0
        with pytest.raises(ZeroDivisionError): _ = self.v_a // Vec(1, 0, 1)
        with pytest.raises(ZeroDivisionError): _ = self.v_a % 0
        with pytest.raises(ZeroDivisionError): _ = self.v_a % Vec(1, 0, 1)
        with pytest.raises(ZeroDivisionError): _ = 1 / Vec(1,0,1)
        with pytest.raises(ZeroDivisionError): _ = 1 // Vec(1,0,1)
        with pytest.raises(ZeroDivisionError): _ = 1 % Vec(1,0,1)

    def test_unsupported_arithmetic_type(self):
        with pytest.raises(TypeError): _ = self.v_a + "s"
        with pytest.raises(TypeError): _ = self.v_a * "s"
        with pytest.raises(TypeError): _ = self.v_a / "s"
        with pytest.raises(TypeError): _ = self.v_a // "s"
        with pytest.raises(TypeError): _ = self.v_a % "s"
        with pytest.raises(TypeError): _ = self.v_a ^ "s"

class TestVectorMethods:
    def test_norm(self):
        v = Vec(3, 4)
        assert math.isclose(v.norm(), 5.0)
        assert math.isclose(v.norm(sqrt=False), 25.0)
        assert math.isclose(Vec(1,1,1).norm(), math.sqrt(3))
        assert math.isclose(Vec().norm(), 0.0)

    def test_hw_wt(self):
        v = Vec(1, 0, 7, 0, -2, 0.0)
        assert v.hw() == 3
        assert v.wt() == 3
        assert v.hw(idxs=[0, 2, 4]) == 3
        assert v.hw(idxs=[1, 3, 5]) == 0
        assert Vec().hw() == 0
        with pytest.raises(IndexError): v.hw(idxs=[0, 10])
        with pytest.raises(IndexError): v.hw(idxs=[0, "a"]) # type error in list comp in hw

    def test_slice_method(self):
        v = Vec(10, 20, 30, 40, 50)
        sliced_v = v.slice([0, 2, 4])
        assert isinstance(sliced_v, Vec)
        assert approx_equal_vec(sliced_v, Vec(10, 30, 50))
        assert sliced_v.dim == 3
        assert approx_equal_vec(v.slice([]), Vec(dim=0))
        with pytest.raises(IndexError): v.slice([0,10])
        with pytest.raises(IndexError): v.slice([0, "a"])

    def test_permute_method(self):
        v = Vec(10, 20, 30)
        permuted_v = v.permute([2, 0, 1])
        assert isinstance(permuted_v, Vec)
        assert approx_equal_vec(permuted_v, Vec(30, 10, 20))
        assert permuted_v.dim == 3
        assert approx_equal_vec(v.permute([0,0,0]), Vec(10,10,10))
        assert approx_equal_vec(v.permute([]), Vec(dim=0))
        with pytest.raises(IndexError): v.permute([0,5])
        with pytest.raises(IndexError): v.permute([0, "a"])

    def test_hw_dist(self):
        v1 = Vec(1, 1, 0, 0)
        v2 = Vec(1, 0, 1, 0)
        assert v1.hw_dist(v2) == 2
        with pytest.raises(ValueError): v1.hw_dist(Vec(1,0))

    def test_euclid_dist(self):
        v1 = Vec(1, 2)
        v2 = Vec(4, 6)
        assert math.isclose(v1.euclid_dist(v2), 5.0)
        assert math.isclose(Vec(1,2,3).euclid_dist(Vec(1,2,3)), 0.0)
        with pytest.raises(ValueError): v1.euclid_dist(Vec(1,2,3))

    def test_l1_norm(self):
        v = Vec(-1, 2, -3.5)
        assert math.isclose(v.l1_norm(), 1 + 2 + 3.5)
        assert math.isclose(Vec().l1_norm(), 0)

    def test_l1_dist(self):
        v1 = Vec(1, -2, 3)
        v2 = Vec(4, 1, -1)
        assert math.isclose(v1.l1_dist(v2), 3 + 3 + 4)
        with pytest.raises(ValueError): v1.l1_dist(Vec(1,1))

    def test_inner_product(self):
        v1 = Vec(1, 2, 3)
        v2 = Vec(4, 5, 6)
        assert math.isclose(v1.inner(v2), 32.0)
        assert math.isclose(Vec(1,2).inner(Vec(0,0)), 0.0)
        assert math.isclose(Vec().inner(Vec()), 0.0)
        with pytest.raises(ValueError): v1.inner(Vec(1,1))
        with pytest.raises(TypeError): v1.inner([1,2,3])

    def test_abs_method(self):
        v = Vec(-1, 0, 2.5, -3)
        assert approx_equal_vec(v.abs(), Vec(1, 0, 2.5, 3))
        assert approx_equal_vec(Vec().abs(), Vec())

class TestVectorComparisons:
    comp1 = Vec(1, 2, 3)
    comp2 = Vec(1, 2, 3)
    comp3 = Vec(1, 2, 4)
    comp_short = Vec(1,2)

    def test_eq_ne(self):
        assert (self.comp1 == self.comp2) is True
        assert (self.comp1 == self.comp3) is False
        assert (self.comp1 == self.comp_short) is False # Different dims
        assert (self.comp1 == [1,2,3]) is False # Different types

        assert (self.comp1 != self.comp2) is False
        assert (self.comp1 != self.comp3) is True
        assert (self.comp1 != self.comp_short) is True
        assert (self.comp1 != [1,2,3]) is True

    def test_ordering_comparisons_same_dim(self):
        assert (Vec(1,2,3) < Vec(2,3,4)) is True
        assert (Vec(1,2,3) < Vec(1,3,4)) is False
        assert (Vec(1,2,3) < Vec(1,2,3)) is False

        assert (self.comp1 <= self.comp2) is True
        assert (self.comp1 <= self.comp3) is True
        assert (Vec(1,3,3) <= Vec(1,2,4)) is False

        assert (Vec(2,3,4) > Vec(1,2,3)) is True
        assert (Vec(1,3,4) > Vec(1,2,3)) is False
        assert (Vec(1,2,3) > Vec(1,2,3)) is False

        assert (self.comp3 >= self.comp1) is True
        assert (self.comp2 >= self.comp1) is True

    def test_ordering_comparisons_different_dim(self):
        with pytest.raises(ValueError): _ = self.comp1 < self.comp_short
        with pytest.raises(ValueError): _ = self.comp1 <= self.comp_short
        with pytest.raises(ValueError): _ = self.comp1 > self.comp_short
        with pytest.raises(ValueError): _ = self.comp1 >= self.comp_short

class TestVectorShiftLikeOps:
    def test_lshift_pad(self):
        v = Vec(1, 2, 3)
        v_ret = v << 4
        assert v is v_ret
        assert v.dim == 5
        assert approx_equal_list(v.values, [1.0, 2.0, 3.0, 0.0, 0.0])
        v << 2
        assert v.dim == 5
        assert approx_equal_list(v.values, [1.0, 2.0, 3.0, 0.0, 0.0])
        v_empty = Vec()
        v_empty << 0
        assert v_empty.dim == 1
        assert approx_equal_list(v_empty.values, [0.0])

    def test_rshift_take_last(self):
        v = Vec(10,20,30,40,50)
        v_ret = v >> 3
        assert v is v_ret
        assert v.dim == 3
        assert approx_equal_list(v.values, [30.0, 40.0, 50.0])

        v_again = Vec(10,20,30,40,50)
        v_again >> 5
        assert v_again.dim == 5
        assert approx_equal_list(v_again.values, [10.0,20.0,30.0,40.0,50.0])

        v_short = Vec(10,20)
        v_short >> 3 # Takes all if N > len
        assert v_short.dim == 2
        assert approx_equal_list(v_short.values, [10.0, 20.0])

        v_zero = Vec(10,20,30)
        v_zero >> 0
        assert v_zero.dim == 0
        assert approx_equal_list(v_zero.values, [])

    def test_shift_invalid_operand(self):
        v = Vec(1,2,3)
        with pytest.raises(ValueError): v << -1
        with pytest.raises(ValueError): v >> -1
        with pytest.raises(TypeError): _ = v << 1.0
        with pytest.raises(TypeError): _ = v >> 1.0

class TestVectorDunderMethods:
    def test_len(self):
        assert len(Vec(1, 2, 3)) == 3
        assert len(Vec(dim=5)) == 5
        assert len(Vec()) == 0

    def test_repr(self):
        v = Vec(1, 2.5)
        assert repr(v) == "Vec([1.0, 2.5])"
        v_empty_dim = Vec(dim=2)
        assert repr(v_empty_dim) == "Vec([0.0, 0.0])"
        v_empty = Vec()
        assert repr(v_empty) == "Vec([])"

    def test_iter(self):
        v_list = [1.0, 2.0, 3.0]
        v = Vec(v_list)
        assert approx_equal_list(list(v), v_list)
        count = 0
        for x_vec, x_list_val in zip(v, v_list):
            assert math.isclose(x_vec, x_list_val)
            count +=1
        assert count == len(v_list)
        empty_v = Vec()
        assert list(empty_v) == []

class TestVectorPrintMethod:
    def test_print_runs(self, capsys):
        v = Vec(1,2,3,4,5,6)
        v.print()
        captured_default = capsys.readouterr()
        assert captured_default.out.strip() == "1.0,2.0,3.0,4.0,5.0,6.0"

        v.print(grouping=2, sep='-')
        captured_grouped = capsys.readouterr()
        assert captured_grouped.out.strip() == "1.02.0-3.04.0-5.06.0"

        Vec(dim=3).print()
        captured_empty_dim = capsys.readouterr()
        assert captured_empty_dim.out.strip() == "0.0,0.0,0.0"

        Vec().print()
        captured_empty = capsys.readouterr()
        assert captured_empty.out.strip() == ""

    def test_print_invalid_args(self):
        v = Vec(1,2)
        with pytest.raises(ValueError): v.print(grouping=0)
        with pytest.raises(TypeError): v.print(sep=123)
