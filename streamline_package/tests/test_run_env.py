import unittest
import os
import json
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

import streamline as sl
from streamline.pipeline import Pipeline, Function, Var
from streamline.pipeline.run_env import RunEnv, DEFAULT, SAVE_PANDAS_DF, load_runenv


class TestRunEnv(unittest.TestCase):
    """Tests for the RunEnv class."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = RunEnv({
            'a': 1,
            'b': 'test',
            'c': [1, 2, 3],
            'd': {'key': 'value'}
        })
        
    def test_initialization(self):
        """Test initialization of RunEnv."""
        # Test empty initialization
        empty_env = RunEnv()
        self.assertEqual(len(empty_env.keys()), 0)
        
        # Test initialization with dictionary
        env = RunEnv({'a': 1, 'b': 2})
        self.assertEqual(env['a'], 1)
        self.assertEqual(env['b'], 2)
        
    def test_repr_and_str(self):
        """Test string representation of RunEnv."""
        env = RunEnv({'a': 1})
        self.assertEqual(repr(env), repr({'a': 1}))
        self.assertEqual(str(env), str({'a': 1}))
        
    def test_contains(self):
        """Test checking if a key is in RunEnv."""
        self.assertTrue('a' in self.env)
        self.assertFalse('non_existent' in self.env)
        
    def test_getitem(self):
        """Test getting items from RunEnv."""
        self.assertEqual(self.env['a'], 1)
        self.assertEqual(self.env['b'], 'test')
        self.assertEqual(self.env['c'], [1, 2, 3])
        self.assertEqual(self.env['d'], {'key': 'value'})
        
        # Test getting non-existent item
        with self.assertRaises(KeyError):
            self.env['non_existent']
            
    def test_setitem(self):
        """Test setting items in RunEnv."""
        self.env['e'] = 5
        self.assertEqual(self.env['e'], 5)
        
        # Update existing item
        self.env['a'] = 10
        self.assertEqual(self.env['a'], 10)
        
    def test_delitem(self):
        """Test deleting items from RunEnv."""
        del self.env['a']
        self.assertFalse('a' in self.env)
        
        # Test deleting non-existent item
        with self.assertRaises(KeyError):
            del self.env['non_existent']
            
    def test_copy(self):
        """Test copying RunEnv."""
        # Shallow copy
        shallow_copy = self.env.copy(deep=False)
        self.assertIsInstance(shallow_copy, RunEnv)
        self.assertEqual(shallow_copy['a'], self.env['a'])
        self.assertEqual(shallow_copy['c'], self.env['c'])
        
        # Deep copy
        deep_copy = self.env.copy(deep=True)
        self.assertIsInstance(deep_copy, RunEnv)
        self.assertEqual(deep_copy['a'], self.env['a'])
        self.assertEqual(deep_copy['c'], self.env['c'])
        self.assertEqual(len(self.env['c']), 3)
        
        # Verify that deep copy is independent
        self.env['c'].append(4)
        self.assertEqual(len(self.env['c']), 4)
        self.assertEqual(len(deep_copy['c']), 3)
        
    def test_keys_values_items(self):
        """Test keys, values, and items methods."""
        keys = list(self.env.keys())
        self.assertEqual(set(keys), {'a', 'b', 'c', 'd'})
        
        values = list(self.env.values())
        self.assertEqual(len(values), 4)
        
        items = list(self.env.items())
        self.assertEqual(len(items), 4)
        
    def test_iter(self):
        """Test iteration over RunEnv."""
        keys = []
        for key in self.env:
            keys.append(key)
        self.assertEqual(set(keys), {'a', 'b', 'c', 'd'})
        
    def test_add_step(self):
        """Test adding steps to RunEnv."""
        step = Function(fun=lambda: 1)
        self.env._add_step(step, {'param': 'value'})
        
        # Check that preamble was created
        self.assertTrue('__preamble__' in self.env)
        self.assertTrue('__steps__' in self.env['__preamble__'])
        
        # Check that step was added
        steps = self.env['__preamble__']['__steps__']
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]['step'], step)
        self.assertEqual(steps[0]['kwargs'], {'param': 'value'})
        
    def test_gen_pipeline(self):
        """Test generating a pipeline from RunEnv."""
        step1 = Function(fun=lambda: 1)
        step2 = Function(fun=lambda: 2)
        
        self.env._add_step(step1, {})
        self.env._add_step(step2, {})
        
        pipeline = self.env.gen_pipeline()
        self.assertIsInstance(pipeline, Pipeline)
        
        # Check that the pipeline has the correct steps
        steps = pipeline.to_list()
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0], step1)
        self.assertEqual(steps[1], step2)
        
    def test_add_var(self):
        """Test adding variables to RunEnv."""
        self.env.add_var('new_var', 42)
        self.assertEqual(self.env['new_var'], 42)
        
    def test_gen_run_kwargs(self):
        """Test generating keyword arguments for running a pipeline."""
        step1 = Function(fun=lambda: 1, arg_cat='cat1')
        step2 = Function(fun=lambda: 2, arg_cat='cat2')
        
        self.env._add_step(step1, {'param1': 'value1'})
        self.env._add_step(step2, {'param2': 'value2'})
        
        kwargs = self.env.gen_run_kwargs()
        self.assertEqual(kwargs, {'cat1_param1': 'value1', 'cat2_param2': 'value2'})
        
    def test_save_and_load(self):
        """Test saving and loading RunEnv."""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Add a DataFrame to the environment
            self.env['df'] = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            
            # Save the environment
            self.env.save(temp_dir)
            
            # Check that files were created
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'dedicated')))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'dedicated', 'df.parquet')))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'env.dill')))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'info.json')))
            
            # Load the environment
            loaded_env = load_runenv(temp_dir)
            
            # Check that variables were loaded correctly
            self.assertEqual(loaded_env['a'], self.env['a'])
            self.assertEqual(loaded_env['b'], self.env['b'])
            self.assertEqual(loaded_env['c'], self.env['c'])
            self.assertEqual(loaded_env['d'], self.env['d'])
            
            # Check that DataFrame was loaded correctly
            pd.testing.assert_frame_equal(loaded_env['df'], self.env['df'])
            
    def test_save_with_options(self):
        """Test saving with custom options."""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Add a DataFrame to the environment
            self.env['df'] = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            
            # Save with CSV format
            self.env.save(temp_dir, save_options=SAVE_PANDAS_DF(format='csv'))
            
            # Check that CSV file was created
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'dedicated', 'df.csv')))
            
            # Load the environment
            loaded_env = load_runenv(temp_dir)
            
            # Check that DataFrame was loaded correctly
            pd.testing.assert_frame_equal(loaded_env['df'], self.env['df'])
            
    def test_save_with_selected_vars(self):
        """Test saving with selected variables."""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save only selected variables
            self.env.save(temp_dir, save_vars=['a', 'b'])
            
            # Load the environment
            loaded_env = load_runenv(temp_dir)
            
            # Check that only selected variables were loaded
            self.assertEqual(loaded_env['a'], self.env['a'])
            self.assertEqual(loaded_env['b'], self.env['b'])
            self.assertFalse('c' in loaded_env)
            self.assertFalse('d' in loaded_env)


class TestDefaultAndSavePandasDF(unittest.TestCase):
    """Tests for the DEFAULT and SAVE_PANDAS_DF classes."""
    
    def test_default(self):
        """Test the DEFAULT sentinel class."""
        default = DEFAULT()
        self.assertIsInstance(default, DEFAULT)
        
    def test_save_pandas_df(self):
        """Test the SAVE_PANDAS_DF configuration class."""
        # Default initialization
        config = SAVE_PANDAS_DF()
        self.assertEqual(config['pandas_df']['format'], 'parquet')
        self.assertEqual(config['pandas_df']['kwargs'], {})
        
        # Custom format
        config = SAVE_PANDAS_DF(format='csv')
        self.assertEqual(config['pandas_df']['format'], 'csv')
        
        # With kwargs
        config = SAVE_PANDAS_DF(format='parquet', index=True, compression='gzip')
        self.assertEqual(config['pandas_df']['format'], 'parquet')
        self.assertEqual(config['pandas_df']['kwargs'], {'index': True, 'compression': 'gzip'})
        
        # Invalid format
        with self.assertRaises(AssertionError):
            SAVE_PANDAS_DF(format='invalid')


if __name__ == '__main__':
    unittest.main() 