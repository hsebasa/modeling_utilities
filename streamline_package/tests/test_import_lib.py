import unittest
import pandas as pd
import numpy as np

from streamline.pipeline.step import ImportLib
from streamline.pipeline.run_env import RunEnv
from streamline.pipeline.step import Var


class TestImportLib(unittest.TestCase):
    """Tests for the ImportLib class."""
    
    def test_init(self):
        """Test initialization of ImportLib."""
        # Initialize with a single module and output variable
        imp = ImportLib('pandas', 'pd_lib')
        self.assertEqual(imp.module_name, 'pandas')
        self.assertEqual(imp.var_name, 'pd_lib')
        
        # Initialize with a single module and no output variable (uses module name)
        imp = ImportLib('pandas')
        self.assertEqual(imp.module_name, 'pandas')
        self.assertEqual(imp.var_name, 'pandas')
        
        # Test with invalid input
        with self.assertRaises(AssertionError):
            ImportLib(123)  # Non-string input
    
    def test_str_repr(self):
        """Test string representation."""
        imp = ImportLib('pandas', 'pd_lib')
        self.assertIn('ImportLib', str(imp))
        self.assertIn('pandas', str(imp))
        self.assertIn('pd_lib', str(imp))
        
        self.assertIn('ImportLib', repr(imp))
        self.assertIn('pandas', repr(imp))
        self.assertIn('pd_lib', repr(imp))
    
    def test_rename_variables(self):
        """Test renaming variables."""
        # Create an ImportLib with an output variable
        imp = ImportLib('pandas', 'pd_lib')
        
        # Rename the output variable
        mappings = {'pd_lib': 'data_lib'}
        renamed = imp.rename(mappings)
        
        # Check that the output variable was renamed
        self.assertEqual(renamed.var_name, 'data_lib')
        
        # Check that module name is unchanged
        self.assertEqual(renamed.module_name, 'pandas')
        
        # Test with no applicable mappings
        mappings = {'other_var': 'new_name'}
        renamed = imp.rename(mappings)
        
        # Check that nothing changed
        self.assertEqual(renamed.var_name, 'data_lib')
    
    def test_get_dependencies(self):
        """Test getting dependencies."""
        imp = ImportLib('pandas', 'pd_lib')
        
        # ImportLib doesn't have dependencies
        deps = imp.get_dependencies()
        self.assertEqual(deps, set())
    
    def test_get_outputs(self):
        """Test getting outputs."""
        imp = ImportLib('pandas', 'pd_lib')
        
        # Output should be the variable name
        outputs = imp.get_outputs()
        self.assertEqual(outputs, {'pd_lib'})
    
    def test_call_pandas(self):
        """Test calling with pandas module."""
        # Create ImportLib for pandas
        imp = ImportLib('pandas', 'pd_lib')
        
        # Create empty environment
        env = RunEnv()
        
        # Call the ImportLib
        result = imp(env, {})
        
        # Check that pandas was imported with correct name
        self.assertIn('pd_lib', env)
        self.assertEqual(env['pd_lib'], pd)
        
        # Try using the imported module
        self.assertTrue(hasattr(env['pd_lib'], 'DataFrame'))
        df = env['pd_lib'].DataFrame({'a': [1, 2, 3]})
        self.assertEqual(len(df), 3)
    
    def test_call_numpy(self):
        """Test calling with numpy module."""
        # Create ImportLib for numpy
        imp = ImportLib('numpy', 'np_lib')
        
        # Create empty environment
        env = RunEnv()
        
        # Call the ImportLib
        result = imp(env, {})
        
        # Check that numpy was imported with correct name
        self.assertIn('np_lib', env)
        self.assertEqual(env['np_lib'], np)
        
        # Try using the imported module
        self.assertTrue(hasattr(env['np_lib'], 'array'))
        arr = env['np_lib'].array([1, 2, 3])
        self.assertEqual(arr.shape, (3,))
    
    def test_call_submodule(self):
        """Test calling with a submodule."""
        # Create ImportLib for pandas.io
        imp = ImportLib('pandas.io', 'pd_io')
        
        # Create empty environment
        env = RunEnv()
        
        # Call the ImportLib
        result = imp(env, {})
        
        # Check that submodule was imported with correct name
        self.assertIn('pd_io', env)
        self.assertEqual(env['pd_io'], pd.io)
        
        # Try using the imported submodule
        self.assertTrue(hasattr(env['pd_io'], 'formats'))
    
    def test_invalid_module(self):
        """Test with invalid module name."""
        # Create ImportLib with non-existent module
        imp = ImportLib('non_existent_module', 'bad_lib')
        
        # Create empty environment
        env = RunEnv()
        
        # Calling should raise ModuleNotFoundError
        with self.assertRaises(ModuleNotFoundError):
            imp(env, {})
    
    def test_tags(self):
        """Test tags for ImportLib."""
        imp = ImportLib('pandas', 'pd_lib')
        
        # Check that ImportLib has the correct tag
        self.assertTrue(imp.has_tag('import_lib'))


if __name__ == '__main__':
    unittest.main() 