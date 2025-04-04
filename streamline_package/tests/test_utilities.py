import unittest
import os
import tempfile
import pandas as pd
import json
import inspect

from streamline.utilities.io import (
    save_json, load_json, save_obj, load_obj, save_pandas_df, load_pandas_df
)
from streamline.utilities.serialize import get_global_dependencies, mainify
from streamline.utilities.tags import _Tags, tags


class TestIO(unittest.TestCase):
    """Tests for the IO utility functions."""
    
    def test_save_load_json(self):
        """Test saving and loading JSON data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test data
            data = {'key1': 'value1', 'key2': 42, 'key3': [1, 2, 3]}
            
            # Save JSON
            json_path = os.path.join(temp_dir, 'test.json')
            save_path = save_json(data, json_path)
            
            # Check that file was created
            self.assertTrue(os.path.exists(json_path))
            
            # Check that file has correct extension
            self.assertTrue(save_path.endswith('.json'))
            
            # Check that returned path is correct
            self.assertEqual(save_path, json_path)
            
            # Check file content
            with open(json_path, 'r') as f:
                content = f.read()
                parsed = json.loads(content)
                self.assertEqual(parsed, data)
            
            # Load JSON
            loaded_data = load_json(json_path)
            
            # Check loaded data
            self.assertEqual(loaded_data, data)
            
            # Test auto-adding extension
            json_path_no_ext = os.path.join(temp_dir, 'test2')
            save_path = save_json(data, json_path_no_ext)
            
            # Check that file was created with extension
            self.assertTrue(os.path.exists(json_path_no_ext + '.json'))
            self.assertEqual(save_path, json_path_no_ext + '.json')
    
    def test_save_load_obj(self):
        """Test saving and loading objects with dill."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test data
            class TestClass:
                def __init__(self, value):
                    self.value = value
                    
                def get_value(self):
                    return self.value
            
            data = TestClass(42)
            
            # Save object
            obj_path = os.path.join(temp_dir, 'test.dill')
            save_path = save_obj(data, obj_path)
            
            # Check that file was created
            self.assertTrue(os.path.exists(obj_path))
            
            # Check that file has correct extension
            self.assertTrue(save_path.endswith('.dill'))
            
            # Check that returned path is correct
            self.assertEqual(save_path, obj_path)
            
            # Load object
            loaded_data = load_obj(obj_path)
            
            # Check loaded data
            self.assertEqual(loaded_data.value, data.value)
            self.assertEqual(loaded_data.get_value(), data.get_value())
            
            # Test auto-adding extension
            obj_path_no_ext = os.path.join(temp_dir, 'test2')
            save_path = save_obj(data, obj_path_no_ext)
            
            # Check that file was created with extension
            self.assertTrue(os.path.exists(obj_path_no_ext + '.dill'))
            self.assertEqual(save_path, obj_path_no_ext + '.dill')
    
    def test_save_load_pandas_df(self):
        """Test saving and loading pandas DataFrames."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test data
            df = pd.DataFrame({
                'A': [1, 2, 3],
                'B': ['a', 'b', 'c'],
                'C': [1.1, 2.2, 3.3]
            })
            
            # Save DataFrame as parquet (default)
            df_path = os.path.join(temp_dir, 'test')
            save_path = save_pandas_df(df, df_path)
            
            # Check that file was created
            self.assertTrue(os.path.exists(df_path + '.parquet'))
            
            # Check that file has correct extension
            self.assertTrue(save_path.endswith('.parquet'))
            
            # Check that returned path is correct
            self.assertEqual(save_path, df_path + '.parquet')
            
            # Load DataFrame
            loaded_df = load_pandas_df(df_path)
            
            # Check loaded data
            pd.testing.assert_frame_equal(loaded_df, df)
            
            # Test saving as CSV
            df_path_csv = os.path.join(temp_dir, 'test_csv')
            save_path = save_pandas_df(df, df_path_csv, format='csv')
            
            # Check that file was created
            self.assertTrue(os.path.exists(df_path_csv + '.csv'))
            
            # Check that file has correct extension
            self.assertTrue(save_path.endswith('.csv'))
            
            # Load DataFrame as CSV
            loaded_df = load_pandas_df(df_path_csv, format='csv')
            
            # Check loaded data (CSV may convert numeric types)
            pd.testing.assert_frame_equal(loaded_df, df, check_dtype=False)
            
            # Test with custom arguments
            df_path_custom = os.path.join(temp_dir, 'test_custom')
            save_path = save_pandas_df(df, df_path_custom, index=True)
            
            # Check that file was created
            self.assertTrue(os.path.exists(df_path_custom + '.parquet'))
            
            # Test invalid format
            with self.assertRaises(AssertionError):
                save_pandas_df(df, df_path, format='invalid')
                
            with self.assertRaises(AssertionError):
                load_pandas_df(df_path, format='invalid')


# class TestSerialize(unittest.TestCase):
#     """Tests for the serialize utility functions."""
    
#     def test_get_global_dependencies(self):
#         """Test getting global dependencies from functions."""
#         # Function with no global dependencies
#         def func_no_globals(x, y):
#             return x + y
            
#         deps = get_global_dependencies(func_no_globals)
#         self.assertEqual(deps, set())
        
#         # Function with global dependencies
#         global_var = 42
        
#         def func_with_globals(x):
#             return x + global_var + min(1, 2) + isinstance(x, int)
            
#         deps = get_global_dependencies(func_with_globals)
#         self.assertIn('global_var', deps)
#         self.assertIn('min', deps)
#         self.assertIn('isinstance', deps)
        
#         # Lambda function
#         lambda_func = lambda x: x + global_var
        
#         deps = get_global_dependencies(lambda_func)
#         self.assertIn('global_var', deps)
    
    # def test_mainify(self):
    #     """Test making an object available in the __main__ module."""
    #     # Define a test class
    #     class TestClass:
    #         def __init__(self, value):
    #             self.value = value
                
    #         def get_value(self):
    #             return self.value
        
    #     # Create an instance
    #     obj = TestClass(42)
        
    #     # Make the class available in __main__
    #     mainify(TestClass)
        
    #     # Check that the class is in __main__
    #     import __main__
    #     self.assertTrue(hasattr(__main__, 'TestClass'))
        
    #     # Check that we can instantiate it from __main__
    #     main_obj = __main__.TestClass(42)
    #     self.assertEqual(main_obj.get_value(), obj.get_value())


class TestTags(unittest.TestCase):
    """Tests for the tags utility."""
    
    def test_tags_class(self):
        """Test the _Tags class."""
        # Check that _Tags has the expected attributes
        self.assertTrue(hasattr(_Tags, 'STEP_FUNCTION'))
        self.assertTrue(hasattr(_Tags, 'STEP_DELETE'))
        self.assertTrue(hasattr(_Tags, 'STEP_VARIABLES_DICT'))
        self.assertTrue(hasattr(_Tags, 'STEP_IMPORT_LIB'))
        
        # Check that attributes have the expected type
        self.assertIsInstance(_Tags.STEP_FUNCTION, set)
        self.assertIsInstance(_Tags.STEP_DELETE, set)
        self.assertIsInstance(_Tags.STEP_VARIABLES_DICT, set)
        self.assertIsInstance(_Tags.STEP_IMPORT_LIB, set)
        
        # Check attribute values
        self.assertEqual(_Tags.STEP_FUNCTION, {'function'})
        self.assertEqual(_Tags.STEP_DELETE, {'delete'})
        self.assertEqual(_Tags.STEP_VARIABLES_DICT, {'add_variables'})
        self.assertEqual(_Tags.STEP_IMPORT_LIB, {'import_lib'})
    
    def test_tags_instance(self):
        """Test the tags instance."""
        # Check that tags is an instance of _Tags
        self.assertIsInstance(tags, _Tags)
        
        # Check that it has the expected attributes
        self.assertTrue(hasattr(tags, 'STEP_FUNCTION'))
        self.assertTrue(hasattr(tags, 'STEP_DELETE'))
        self.assertTrue(hasattr(tags, 'STEP_VARIABLES_DICT'))
        self.assertTrue(hasattr(tags, 'STEP_IMPORT_LIB'))
        
        # Check that attribute values match _Tags
        self.assertEqual(tags.STEP_FUNCTION, _Tags.STEP_FUNCTION)
        self.assertEqual(tags.STEP_DELETE, _Tags.STEP_DELETE)
        self.assertEqual(tags.STEP_VARIABLES_DICT, _Tags.STEP_VARIABLES_DICT)
        self.assertEqual(tags.STEP_IMPORT_LIB, _Tags.STEP_IMPORT_LIB)


if __name__ == '__main__':
    unittest.main() 