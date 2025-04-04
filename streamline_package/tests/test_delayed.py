import unittest
from streamline.delayed import Delayed, eval_delay, step, delay_lib
from streamline.delayed.delayed import _Library
import streamline as sl
from streamline.pipeline import Function


class TestDelayed(unittest.TestCase):
    """
    Tests for the Delayed class and related functionality in the delayed module.
    """
    
    def setUp(self):
        self.delayed = Delayed(prefix='test')
        
    def test_basic_initialization(self):
        """Test basic initialization of Delayed objects."""
        d1 = Delayed()
        self.assertIsNone(d1._prefix)
        
        d2 = Delayed(prefix='test')
        self.assertEqual(d2._prefix, 'test')
        
    def test_repr(self):
        """Test string representation of Delayed objects."""
        d = Delayed(prefix='test')
        self.assertEqual(repr(d), 'test')
        
    def test_getattr(self):
        """Test attribute access on Delayed objects."""
        d = Delayed(prefix='test')
        attr_d = d.attribute
        self.assertIsInstance(attr_d, Delayed)
        self.assertEqual(repr(attr_d), 'test.attribute')
        
        # Nested attribute access
        nested_d = d.attribute.nested
        self.assertIsInstance(nested_d, Delayed)
        self.assertEqual(repr(nested_d), 'test.attribute.nested')
        
    def test_call(self):
        """Test function call representation on Delayed objects."""
        d = Delayed(prefix='test')
        call_d = d(1, 2, key='value')
        self.assertIsInstance(call_d, Delayed)
        self.assertEqual(repr(call_d), "test(*(1, 2), **{'key': 'value'})")
        
    def test_add(self):
        """Test addition operation on Delayed objects."""
        d = Delayed(prefix='test')
        add_d = d + 5
        self.assertIsInstance(add_d, Delayed)
        self.assertEqual(repr(add_d), 'test+5')
        
    def test_comparison_operators(self):
        """Test comparison operators on Delayed objects."""
        d = Delayed(prefix='test')
        
        # Test less than
        lt_d = d < 5
        self.assertIsInstance(lt_d, Delayed)
        self.assertEqual(repr(lt_d), 'test<5')
        
        # Test less than or equal
        le_d = d <= 5
        self.assertIsInstance(le_d, Delayed)
        self.assertEqual(repr(le_d), 'test<=5')
        
        # Test greater than
        gt_d = d > 5
        self.assertIsInstance(gt_d, Delayed)
        self.assertEqual(repr(gt_d), 'test>5')
        
        # Test greater than or equal
        ge_d = d >= 5
        self.assertIsInstance(ge_d, Delayed)
        self.assertEqual(repr(ge_d), 'test>=5')
        
        # Test equality
        eq_d = d == 5
        self.assertIsInstance(eq_d, Delayed)
        self.assertEqual(repr(eq_d), '(test == 5)')
        
    def test_logical_operators(self):
        """Test logical operators on Delayed objects."""
        d1 = Delayed(prefix='test1')
        d2 = Delayed(prefix='test2')
        
        # Test OR
        or_d = d1 | d2
        self.assertIsInstance(or_d, Delayed)
        self.assertEqual(repr(or_d), '(test1 | test2)')
        
        # Test AND
        and_d = d1 & d2
        self.assertIsInstance(and_d, Delayed)
        self.assertEqual(repr(and_d), '(test1 & test2)')
        
    def test_contains_and_isin(self):
        """Test contains and isin methods on Delayed objects."""
        d = Delayed(prefix='test')
        
        # Test contains
        contains_d = d.contains(5)
        self.assertIsInstance(contains_d, Delayed)
        self.assertEqual(repr(contains_d), '(5 in test)')
        
        # Test isin
        isin_d = d.isin([1, 2, 3])
        self.assertIsInstance(isin_d, Delayed)
        self.assertEqual(repr(isin_d), '(test in [1, 2, 3])')
        
    def test_not(self):
        """Test logical NOT operation on Delayed objects."""
        d = Delayed(prefix='test')
        not_d = ~d
        self.assertIsInstance(not_d, Delayed)
        self.assertEqual(repr(not_d), '~(test)')
        
    def test_eval_delay(self):
        """Test evaluating Delayed expressions."""
        # Simple expression
        d = Delayed(prefix='x + y')
        result = eval_delay(d, {'x': 2, 'y': 3})
        self.assertEqual(result, 5)
        
        # More complex expression
        d = Delayed(prefix='x * y + z')
        result = eval_delay(d, {'x': 2, 'y': 3, 'z': 4})
        self.assertEqual(result, 10)
        
        # Test with function calls
        def test_func(x):
            return x * 2
            
        d = Delayed(prefix='test_func(x)')
        result = eval_delay(d, {'x': 5, 'test_func': test_func})
        self.assertEqual(result, 10)
        
    def test_step_variable(self):
        """Test the predefined 'step' variable."""
        self.assertIsInstance(step, Delayed)
        self.assertEqual(repr(step), 'step')
        
        # Test using step in pipeline context
        test_step = Function(fun=lambda: 1, arg_cat='test_cat', tags={'test_tag'})
        
        # Test accessing attributes
        expr = step.arg_cat == 'test_cat'
        result = eval_delay(expr, {'step': test_step})
        self.assertTrue(result)
        
        # Test tags contains
        expr = step.tags.contains('test_tag')
        result = eval_delay(expr, {'step': test_step})
        self.assertTrue(result)
        
        # Test negative case
        expr = step.tags.contains('non_existent_tag')
        result = eval_delay(expr, {'step': test_step})
        self.assertFalse(result)


class TestLibrary(unittest.TestCase):
    """
    Tests for the _Library class in the delayed module.
    """
    
    def test_initialization(self):
        """Test initialization of the _Library class."""
        # Default initialization
        lib = _Library()
        
        # Check that built-in functions are included
        self.assertIn('min', lib.globals)
        self.assertIn('max', lib.globals)
        self.assertIn('type', lib.globals)
        self.assertIn('str', lib.globals)
        self.assertIn('float', lib.globals)
        self.assertIn('int', lib.globals)
        self.assertIn('round', lib.globals)
        
        # Initialize with custom globals
        custom_globals = {'custom_func': lambda x: x * 2}
        lib = _Library(globals=custom_globals)
        
        # Check that custom function is included
        self.assertIn('custom_func', lib.globals)
        self.assertEqual(lib.globals['custom_func'](5), 10)
        
    def test_getattr(self):
        """Test attribute access on _Library objects."""
        lib = _Library(globals={'test_func': lambda x: x * 2})
        
        # Access existing global
        attr = lib.test_func
        self.assertIsInstance(attr, Delayed)
        self.assertEqual(repr(attr), 'test_func')
        
        # Access non-existent global (should raise AssertionError)
        with self.assertRaises(AssertionError):
            attr = lib.non_existent_func
            
    def test_delayed_decorator(self):
        """Test the delayed decorator."""
        lib = _Library()
        
        @lib.delayed('test_func')
        def test_func(x):
            return x * 2
        
        # Check that the function is added to globals
        self.assertIn('test_func', lib.globals)
        self.assertEqual(lib.globals['test_func'](5), 10)
        
        # Check that the decorated function returns a Delayed object
        result = test_func(5)
        self.assertIsInstance(result, Delayed)
        self.assertEqual(repr(result), 'test_func(*(5,), **{})')
        
    def test_delay_lib(self):
        """Test the predefined delay_lib instance."""
        self.assertIsInstance(delay_lib, _Library)
        
        # Test using a built-in function
        expr = delay_lib.type(5).__name__ == 'int'
        result = eval_delay(expr, {})
        self.assertTrue(result)
        
        # Test chaining operations
        expr = delay_lib.str(5) == '5'
        result = eval_delay(expr, {})
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main() 