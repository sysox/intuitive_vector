# C:/Users/user/PycharmProjects/intuitive_vector/tests/test_vector.py
import pytest
import math
from vector import Vec, Number  # Assuming your __init__.py or vector.py exports these


# Helper for float comparisons in vectors and lists
def approx_equal_vec(v1: Vec, v2: Vec, tol=1e-9) -> bool:
    if v1.dim != v2.dim:
        return False
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
        assert v.dim == 3  # Max index + 1
        assert approx_equal_list(v.values, [10.0, 20.0, 30.0])

    def test_from_dict_with_dim_padding(self):
        v = Vec({0: 10, 2: 30}, dim=4)
        assert v.dim == 4
        assert approx_equal_list(v.values, [10.0, 0.0, 30.0, 0.0])

    def test_from_dict_with_dim_truncation(self):
        v = Vec({0: 10, 1: 20, 2: 30}, dim=2)
        assert v.dim == 2
        assert approx_equal_list(v.values, [10.0, 20.0])

    def test_from_pair_indices_value(self):
        v = Vec(([0, 3], 5))  # ([indices], value)
        assert v.dim == 4  # Max index + 1
        assert approx_equal_list(v.values, [5.0, 0.0, 0.0, 5.0])

    def test_from_pair_with_dim_padding(self):
        v = Vec(([0, 2], 5), dim=5)
        assert v.dim == 5
        assert approx_equal_list(v.values, [5.0, 0.0, 5.0, 0.0, 0.0])

    def test_from_pair_and_numbers(self):
        v = Vec(([0, 3], 10), 99, ([1, 4], 88), dim=6)
        # Expected: [10, 99 then 88, 10, 88, 0]
        # Order of processing args:
        # 1. ([0,3], 10) -> [10,0,0,10]
        # 2. 99 -> [10,0,0,10,99] (appended)
        # 3. ([1,4], 88) -> applied to current list.
        #    max_idx is 4. current len is 5. extend to len 5.
        #    self.values[1] = 88, self.values[4] = 88
        #    [10, 88, 0, 10, 88]
        # 4. dim=6 -> pad to [10, 88, 0, 10, 88, 0]
        assert v.dim == 6
        assert approx_equal_list(v.values, [10.0, 88.0, 0.0, 10.0, 88.0, 0.0])

    def test_truncation_by_dim_kwarg(self):
        v = Vec(1, 2, 3, 4, 5, dim=3)
        assert v.dim == 3
        assert approx_equal_list(v.values, [1.0, 2.0, 3.0])

    def test_padding_by_dim_kwarg(self):
        v = Vec(1, 2, dim=4)
        assert v.dim == 4
        assert approx_equal_list(v.values, [1.0, 2.0, 0.0, 0.0])

    def test_invalid_type_in_init(self):
        with pytest.raises(TypeError):
            Vec("a", "b")
        with pytest.raises(TypeError):
            Vec([1, "b"])
        with pytest.raises(TypeError):
            Vec({0: 1, "a": 2})
        with pytest.raises(TypeError):
            Vec({0: 1, 1: "b"})
        with pytest.raises(TypeError):
            Vec((["a"], 1))
        with pytest.raises(TypeError):
            Vec(([0], "b"))

    def test_negative_dim_in_init(self):
        with pytest.raises(ValueError):
            Vec(dim=-1)


class TestVectorAccessAndModification:
    def test_getitem_valid(self):
        v = Vec(10, 20, 30)
        assert math.isclose(v[0], 10.0)
        assert math.isclose(v[1], 20.0)
        assert math.isclose(v[-1], 30.0)
        assert math.isclose(v[-3], 10.0)

    def test_getitem_invalid_index(self):
        v = Vec(10, 20)
        with pytest.raises(IndexError):
            _ = v[2]
        with pytest.raises(IndexError):
            _ = v[-3]

    def test_getitem_slice(self):
        v = Vec(10, 20, 30, 40, 50)
        # Slicing a Vec returns a list of Numbers
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
        v.set_values(d={0: 100, 3: 300})  # Extends
        assert approx_equal_list(v.values, [100.0, 2.0, 0.0, 300.0])
        assert v.dim == 4

    def test_set_values_pair(self):
        v = Vec(1, 2, 3)
        v.set_values(p=([0, 4], 77))  # Extends
        assert approx_equal_list(v.values, [77.0, 2.0, 3.0, 0.0, 77.0])
        assert v.dim == 5

    def test_set_values_multiple_and_dim_update(self):
        v = Vec(dim=1)  # [0.0]
        v.set_values(val=1, seq=[2, 3], d={0: 10, 4: 40}, p=([1, 5], 55))
        # Initial: [0.0], dim=1
        # val=1: [0.0, 1.0]
        # seq=[2,3]: [0.0, 1.0, 2.0, 3.0]
        # d={0:10, 4:40}: [10.0, 1.0, 2.0, 3.0, 40.0] (len 5)
        # p=([1,5], 55): [10.0, 55.0, 2.0, 3.0, 40.0, 55.0] (len 6)
        assert approx_equal_list(v.values, [10.0, 55.0, 2.0, 3.0, 40.0, 55.0])
        assert v.dim == 6

    def test_set_values_invalid_types(self):
        v = Vec(1)
        with pytest.raises(TypeError):
            v.set_values(val="a")
        with pytest.raises(TypeError):
            v.set_values(seq=[1, "a"])
        with pytest.raises(TypeError):
            v.set_values(d={"a": 1})
        with pytest.raises(TypeError):
            v.set_values(d={0: "a"})
        with pytest.raises(TypeError):
            v.set_values(p=(["a"], 1))
        with pytest.raises(TypeError):
            v.set_values(p=([0], "a"))


class TestVectorArithmetic:
    v_a = Vec(1, 2, 3)
    v_b = Vec(4, 5, 6)
    v_c = Vec(2, 4, 0)  # For division tests

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
        v2 = Vec(2, 5, 3)
        assert approx_equal_vec(v1 / v2, Vec(2.0, 2.0, 6.0))

    def test_reverse_true_division_scalar(self):
        assert approx_equal_vec(12 / Vec(2, 3, 4), Vec(6.0, 4.0, 3.0))

    def test_floor_division_scalar(self):
        assert approx_equal_vec(Vec(5, 6, 7) // 2, Vec(2, 3, 3))

    def test_floor_division_vec(self):
        assert approx_equal_vec(Vec(7, 8, 9) // Vec(2, 3, 4), Vec(3, 2, 2))

    def test_reverse_floor_division_scalar(self):
        assert approx_equal_vec(13 // Vec(2, 3, 5), Vec(6, 4, 2))

    def test_modulo_scalar(self):
        assert approx_equal_vec(Vec(5, 6, 7) % 3, Vec(2, 0, 1))
        assert approx_equal_vec(Vec(5.5, 6.5) % 3, Vec(2.5, 0.5))  # math.fmod behavior

    def test_modulo_vec(self):
        assert approx_equal_vec(Vec(5, 6, 7) % Vec(2, 3, 5), Vec(1, 0, 2))

    def test_reverse_modulo_scalar(self):
        assert approx_equal_vec(7 % Vec(2, 3, 4), Vec(1, 1, 3))

    def test_xor_vec(self):
        v1 = Vec(1, 0, 1, 0)  # Values will be cast to int for XOR
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
        with pytest.raises(ZeroDivisionError): _ = 1 / Vec(1, 0, 1)


class TestVectorMethods:
    def test_norm(self):
        v = Vec(3, 4)
        assert math.isclose(v.norm(), 5.0)
        assert math.isclose(v.norm(sqrt=False), 25.0)
        assert math.isclose(Vec(1, 1, 1).norm(), math.sqrt(3))

    def test_hw_wt(self):
        v = Vec(1, 0, 7, 0, -2, 0.0)
        assert v.hw() == 3
        assert v.wt() == 3
        assert v.hw(idxs=[0, 2, 4]) == 3
        assert v.hw(idxs=[1, 3, 5]) == 0
        with pytest.raises(IndexError):
            v.hw(idxs=[0, 10])

    def test_slice_method(self):
        v = Vec(10, 20, 30, 40, 50)
        sliced_v = v.slice([0, 2, 4])
        assert isinstance(sliced_v, Vec)
        assert approx_equal_vec(sliced_v, Vec(10, 30, 50))
        assert sliced_v.dim == 3
        with pytest.raises(IndexError):
            v.slice([0, 10])

    def test_permute_method(self):
        v = Vec(10, 20, 30)
        permuted_v = v.permute([2, 0, 1])
        assert isinstance(permuted_v, Vec)
        assert approx_equal_vec(permuted_v, Vec(30, 10, 20))
        assert permuted_v.dim == 3
        with pytest.raises(IndexError):
            v.permute([0, 5])

    def test_hw_dist(self):
        v1 = Vec(1, 1, 0, 0)  # Assumes binary context after modulo
        v2 = Vec(1, 0, 1, 0)
        # (v1-v2) = (0,1,-1,0)
        # (v1-v2)%2 = (0,1,1,0) if -1%2 is 1 (Python behavior)
        # wt((0,1,1,0)) = 2
        assert v1.hw_dist(v2) == 2
        with pytest.raises(ValueError):
            v1.hw_dist(Vec(1, 0))

    def test_euclid_dist(self):
        v1 = Vec(1, 2)
        v2 = Vec(4, 6)  # diff = (-3, -4), norm = sqrt(9+16) = 5
        assert math.isclose(v1.euclid_dist(v2), 5.0)
        with pytest.raises(ValueError):
            v1.euclid_dist(Vec(1, 2, 3))

    def test_l1_norm(self):
        v = Vec(-1, 2, -3.5)
        assert math.isclose(v.l1_norm(), 1 + 2 + 3.5)

    def test_l1_dist(self):
        v1 = Vec(1, -2, 3)
        v2 = Vec(4, 1, -1)  # diffs: |-3|, |-3|, |4| => 3,3,4
        assert math.isclose(v1.l1_dist(v2), 3 + 3 + 4)
        with pytest.raises(ValueError):
            v1.l1_dist(Vec(1, 1))

    def test_inner_product(self):
        v1 = Vec(1, 2, 3)
        v2 = Vec(4, 5, 6)  # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert math.isclose(v1.inner(v2), 32.0)
        with pytest.raises(ValueError):
            v1.inner(Vec(1, 1))
        with pytest.raises(TypeError):
            v1.inner([1, 2, 3])  # Must be Vec instance

    def test_abs_method(self):
        v = Vec(-1, 0, 2.5, -3)
        assert approx_equal_vec(v.abs(), Vec(1, 0, 2.5, 3))


class TestVectorComparisons:
    comp1 = Vec(1, 2, 3)
    comp2 = Vec(1, 2, 3)
    comp3 = Vec(1, 2, 4)
    comp4 = Vec(0, 3, 2)
    comp_short = Vec(1, 2)

    def test_eq_ne(self):
        assert (self.comp1 == self.comp2) is True
        assert (self.comp1 == self.comp3) is False
        assert (self.comp1 == self.comp_short) is False  # Different dims
        assert (self.comp1 == [1, 2, 3]) is False  # Different types

        assert (self.comp1 != self.comp2) is False
        assert (self.comp1 != self.comp3) is True
        assert (self.comp1 != self.comp_short) is True
        assert (self.comp1 != [1, 2, 3]) is True

    def test_ordering_comparisons_same_dim(self):
        assert (self.comp1 < self.comp3) is True  # 1<1 F, 2<2 F, 3<4 T -> all must be true
        # The _compare_elementwise implies all elements must satisfy the condition.
        # So (1,2,3) < (1,2,4) is true because 1<=1, 2<=2, 3<4.
        # Let's re-check the logic for <, <= etc.
        # A common definition for v1 < v2 is if all v1[i] < v2[i].
        # Or lexicographical. The current implementation is element-wise 'all'.
        # Vec(1,2,3) < Vec(2,3,4) -> True
        # Vec(1,2,3) < Vec(1,3,4) -> True
        # Vec(1,2,3) < Vec(1,2,4) -> True
        # Vec(1,2,3) < Vec(1,2,3) -> False

        assert (Vec(1, 2, 3) < Vec(2, 3, 4)) is True
        assert (Vec(1, 2, 3) < Vec(1, 2, 4)) is True  # This depends on strict < for all or one.
        # The lambda a,b: a < b with all() means all must be strictly less.
        # So Vec(1,2,3) < Vec(1,2,4) should be False if one element is not strictly less.
        # Let's assume the implementation is all(op(a,b)).
        # For Vec(1,2,3) < Vec(1,2,4): 1<1 (F), 2<2 (F), 3<4 (T). all() is False.
        assert (Vec(1, 2, 3) < Vec(1, 2, 4)) is False  # Based on all(a<b)
        assert (Vec(0, 0, 0) < Vec(1, 1, 1)) is True

        assert (self.comp1 <= self.comp2) is True
        assert (self.comp1 <= self.comp3) is True  # 1<=1, 2<=2, 3<=4
        assert (Vec(1, 3, 3) <= Vec(1, 2, 4)) is False  # 3<=2 is False

        assert (self.comp3 > self.comp1) is True  # 1>1 F, 2>2 F, 4>3 T. all() is False.
        assert (Vec(2, 3, 4) > Vec(1, 2, 3)) is True
        assert (Vec(1, 2, 4) > Vec(1, 2, 3)) is False  # Based on all(a>b)

        assert (self.comp3 >= self.comp1) is True  # 1>=1, 2>=2, 4>=3
        assert (self.comp2 >= self.comp1) is True

    def test_ordering_comparisons_different_dim(self):
        with pytest.raises(ValueError): _ = self.comp1 < self.comp_short
        with pytest.raises(ValueError): _ = self.comp1 <= self.comp_short
        with pytest.raises(ValueError): _ = self.comp1 > self.comp_short
        with pytest.raises(ValueError): _ = self.comp1 >= self.comp_short


class TestVectorShiftLikeOps:
    def test_lshift_pad(self):
        v = Vec(1, 2, 3)
        v_ret = v << 4  # Pad to ensure index 4 exists (length 5)
        assert v is v_ret  # In-place
        assert v.dim == 5
        assert approx_equal_list(v.values, [1.0, 2.0, 3.0, 0.0, 0.0])
        v << 2  # Already long enough
        assert v.dim == 5
        assert approx_equal_list(v.values, [1.0, 2.0, 3.0, 0.0, 0.0])

    def test_rshift_take_last(self):
        v = Vec(10, 20, 30, 40, 50)
        v_ret = v >> 3  # Keep last 3
        assert v is v_ret  # In-place
        assert v.dim == 3
        assert approx_equal_list(v.values, [30.0, 40.0, 50.0])
        v >> 5  # Request more than available, keeps all remaining
        assert v.dim == 3
        assert approx_equal_list(v.values, [30.0, 40.0, 50.0])
        v >> 0  # Keep last 0
        assert v.dim == 0
        assert approx_equal_list(v.values, [])

    def test_shift_invalid_operand(self):
        v = Vec(1, 2, 3)
        with pytest.raises(ValueError): v << -1
        with pytest.raises(ValueError): v >> -1
        with pytest.raises(TypeError): v << 1.0  # Must be int
        with pytest.raises(TypeError): v >> 1.0  # Must be int


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
        for x, y in zip(v, v_list):
            assert math.isclose(x, y)
            count += 1
        assert count == len(v_list)


class TestVectorPrintMethod:  # Basic check that it runs
    def test_print_runs(self, capsys):
        v = Vec(1, 2, 3, 4, 5, 6)
        v.print()
        captured_default = capsys.readouterr()
        assert captured_default.out.strip() == "1,2,3,4,5,6"

        v.print(grouping=2, sep='-')
        captured_grouped = capsys.readouterr()
        assert captured_grouped.out.strip() == "12-34-56"

        Vec(dim=3).print()  # Test empty vector with dim
        captured_empty_dim = capsys.readouterr()
        assert captured_empty_dim.out.strip() == "0.0,0.0,0.0"  # or "0,0,0" depending on str(0.0)

        Vec().print()
        captured_empty = capsys.readouterr()
        assert captured_empty.out.strip() == ""

    def test_print_invalid_args(self):
        v = Vec(1, 2)
        with pytest.raises(ValueError):
            v.print(grouping=0)
        with pytest.raises(TypeError):
            v.print(sep=123)
