import unittest
import os
import tempfile
import pandas as pd

from streamline.pipeline.pipeline import Pipeline, StepNotFound
from streamline.pipeline.step import Function, Delete, VariablesDict, Var
from streamline.pipeline.run_env import RunEnv


class TestPipeline(unittest.TestCase):
    """Tests for the Pipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple pipeline for testing
        self.pipeline = Pipeline()
        
        # Add a few steps
        self.pipeline.add_step(Function(lambda x: x + 1, args=[Var('x')], out_var='y'))
        self.pipeline.add_step(Function(lambda y: y * 2, args=[Var('y')], out_var='z'))
        
        # Create test data
        self.env = RunEnv({'x': 5})
    
    def test_init(self):
        """Test Pipeline initialization."""
        # Test empty initialization
        p = Pipeline()
        self.assertEqual(len(p.to_list()), 0)
        
        # Test initialization with steps
        steps = [
            Function(lambda x: x + 1, args=[Var('x')], out_var='y'),
            Function(lambda y: y * 2, args=[Var('y')], out_var='z')
        ]
        p = Pipeline(steps)
        self.assertEqual(len(p.to_list()), 2)
    
    def test_str_repr(self):
        """Test string representation."""
        # Test string representation
        self.assertIn("Pipeline", str(self.pipeline))
        self.assertIn("Pipeline", repr(self.pipeline))
    
    def test_len(self):
        """Test length method."""
        self.assertEqual(len(self.pipeline), 2)
    
    def test_getitem(self):
        """Test getitem method."""
        # Get first step
        step = self.pipeline.loc[0]
        self.assertIsInstance(step, Pipeline)
        
        # Get slice
        self.assertIsInstance(step.to_list()[0], Function)
        
        # Get slice
        steps = self.pipeline.loc[0:2]
        self.assertIsInstance(steps, Pipeline)
        self.assertEqual(len(steps), 2)
        
        # Test invalid index
        with self.assertRaises(StepNotFound):
            self.pipeline.loc[10]
    
    def test_iter(self):
        """Test iteration."""
        steps = list(self.pipeline)
        self.assertEqual(len(steps), 2)
        self.assertIsInstance(steps[0], Function)
        self.assertIsInstance(steps[1], Function)
    
    def test_add_step(self):
        """Test adding steps."""
        p = Pipeline()
        
        # Add Function step
        p.add_step(Function(lambda x: x + 1, args=[Var('x')], out_var='y'))
        self.assertEqual(len(p), 1)
        
        # Add Delete step
        p.add_step(Delete('y'))
        self.assertEqual(len(p), 2)
        
        # Add VariablesDict step
        p.add_step(VariablesDict(dict(z=10)))
        self.assertEqual(len(p), 3)
    
    def test_add(self):
        """Test adding pipelines."""
        p1 = Pipeline([Function(lambda x: x + 1, args=[Var('x') ], out_var='y')])
        p2 = Pipeline([Function(lambda y: y * 2, args=[Var('y')], out_var='z')])
        
        # Add pipelines
        p3 = p1 + p2
        self.assertEqual(len(p3), 2)
        
        # Test running combined pipeline
        result = p3.run({'x': 5})
        self.assertEqual(result.get('z'), 12)  # (5 + 1) * 2 = 12
    
    def test_iadd(self):
        """Test in-place addition."""
        p = Pipeline([Function(lambda x: x + 1, args=[Var('x')], out_var='y')])
        p += Pipeline([Function(lambda y: y * 2, args=[Var('y')], out_var='z')])
        
        self.assertEqual(len(p), 2)
        
        # Test running modified pipeline
        result = p.run({'x': 5})
        self.assertEqual(result.get('z'), 12)  # (5 + 1) * 2 = 12
    
    def test_extend(self):
        """Test extending pipeline."""
        p = Pipeline([Function(lambda x: x + 1, args=[Var('x')], out_var='y')])
        p.extend([
            Function(lambda y: y * 2, args=[Var('y')], out_var='z'),
            Function(lambda z: z - 3, args=[Var('z')], out_var='a')
        ])
        
        self.assertEqual(len(p), 3)
        
        # Test running extended pipeline
        result = p.run({'x': 5})
        self.assertEqual(result.get('a'), 9)  # ((5 + 1) * 2) - 3 = 9
    
    def test_run(self):
        """Test running pipeline."""
        # Run with dictionary
        result = self.pipeline.run({'x': 5})
        self.assertEqual(result.get('z'), 12)  # (5 + 1) * 2 = 12
        
        # Run with RunEnv
        result = self.pipeline.run(RunEnv({'x': 5}))
        self.assertEqual(result.get('z'), 12)
        
        # Test with keyword arguments
        result = self.pipeline.run(dict(x=5))
        self.assertEqual(result.get('z'), 12)
        
        # Test with both dictionary and keyword arguments
        result = self.pipeline.run({'x': 3, 'y': 7})
        self.assertEqual(result.get('z'), 8)  # The step will use y=4 overwritting y=7
    
    def test_run_with_progress(self):
        """Test running with progress tracking."""
        # Simple test to ensure no errors
        result = self.pipeline.run({'x': 5})
        self.assertEqual(result.get('z'), 12)
    
    def test_get_dependencies(self):
        """Test getting dependencies."""
        deps = self.pipeline.get_dependencies()
        self.assertEqual(deps, {'x'})
        
        # Test with pipeline that has multiple dependencies
        p = Pipeline([
            Function(lambda x, y: x + y, args=[Var('x'), Var('y')], out_var='z'),
            Function(lambda z, a: z * a, args=[Var('z'), Var('a')], out_var='b')
        ])
        deps = p.get_dependencies()
        self.assertEqual(deps, {'x', 'y', 'a'})
    
    def test_get_outputs(self):
        """Test getting outputs."""
        outputs = self.pipeline.get_outputs()
        self.assertEqual(outputs, {'y', 'z'})
    
    def test_rename_variables(self):
        """Test renaming variables."""
        # Create mappings
        mappings = {'x': 'input', 'y': 'intermediate', 'z': 'output'}
        
        # Rename variables
        p = self.pipeline.rename(mappings)
        
        # Check dependencies
        self.assertEqual(p.get_dependencies(), {'input'})
        
        # Check outputs
        self.assertEqual(p.get_outputs(), {'intermediate', 'output'})
        
        # Run pipeline to verify it works
        result = p.run({'input': 5})
        self.assertEqual(result.get('output'), 12)
    
    def test_apply(self):
        """Test applying a function to the pipeline."""
        # Define a function to apply
        def double_function(step):
            if isinstance(step, Function):
                # Create a new function that doubles the output of the original function
                original_func = step._fun
                new_func = lambda *args, **kwargs: original_func(*args, **kwargs) * 2
                return Function(new_func, args=step.args, kw=step.kw, out_var=step.out_var)
            return step
        
        # Apply the function
        p = self.pipeline.apply(double_function)
        
        # Run the pipeline
        result = p.run({'x': 5})
        
        # First step: (5 + 1) * 2 = 12, Second step: (12 * 2) * 2 = 48
        self.assertEqual(result.get('z'), 48)
    
    def test_save_load(self):
        """Test saving and loading pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save pipeline
            path = os.path.join(temp_dir, 'pipeline.dill')
            self.pipeline.save(path)
            
            # Check that file exists
            self.assertTrue(os.path.exists(path))
            
            # Load pipeline
            loaded = Pipeline.load(path)
            
            # Check that loaded pipeline works
            result = loaded.run({'x': 5})
            self.assertEqual(result.get('z'), 12)
    
    def test_get_step_by_tag(self):
        """Test getting steps by tag."""
        # Create pipeline with tagged steps
        p = Pipeline([
            Function(lambda x: x + 1, args=['x'], out_var='y'),
            Delete('x'),
            VariablesDict(dict(z=10))
        ])
        
        # Get function steps
        function_steps = p.get_steps_by_tag('function')
        self.assertEqual(len(function_steps), 1)
        self.assertIsInstance(function_steps[0], Function)
        
        # Get delete steps
        delete_steps = p.get_steps_by_tag('delete')
        self.assertEqual(len(delete_steps), 1)
        self.assertIsInstance(delete_steps[0], Delete)
        
        # Get variable dict steps
        var_steps = p.get_steps_by_tag('add_variables')
        self.assertEqual(len(var_steps), 1)
        self.assertIsInstance(var_steps[0], VariablesDict)
    
    def test_get_step_by_output(self):
        """Test getting steps by output."""
        # Get step that outputs 'y'
        steps = self.pipeline.get_steps_by_output('y')
        self.assertEqual(len(steps), 1)
        self.assertIsInstance(steps[0], Function)
        
        # Get step that outputs 'z'
        steps = self.pipeline.get_steps_by_output('z')
        self.assertEqual(len(steps), 1)
        self.assertIsInstance(steps[0], Function)
        
        # Get steps for non-existent output
        steps = self.pipeline.get_steps_by_output('non_existent')
        self.assertEqual(len(steps), 0)
    
    def test_get_step_by_input(self):
        """Test getting steps by input."""
        # Get step that inputs 'x'
        steps = self.pipeline.get_steps_by_input('x')
        self.assertEqual(len(steps), 1)
        self.assertIsInstance(steps[0], Function)
        
        # Get step that inputs 'y'
        steps = self.pipeline.get_steps_by_input('y')
        self.assertEqual(len(steps), 1)
        self.assertIsInstance(steps[0], Function)
        
        # Get steps for non-existent input
        steps = self.pipeline.get_steps_by_input('non_existent')
        self.assertEqual(len(steps), 0)
    
    def test_get_subgraph(self):
        """Test getting a subgraph of the pipeline."""
        # Create a more complex pipeline
        p = Pipeline([
            Function(lambda x: x + 1, args=[Var('x')], out_var='y'),
            Function(lambda y: y * 2, args=[Var('y')], out_var='z'),
            Function(lambda z: z - 3, args=[Var('z')], out_var='a'),
            Function(lambda a: a / 2, args=[Var('a')], out_var='b')
        ])
        
        # Get subgraph from 'y' to 'a'
        subgraph = p.get_subgraph(['y'], ['a'])
        self.assertEqual(len(subgraph), 2)  # Should include y->z and z->a steps
        
        # Run the subgraph
        result = subgraph.run({'y': 6})
        self.assertEqual(result.get('a'), 9)  # (6 * 2) - 3 = 9
    
    def test_validate(self):
        """Test pipeline validation."""
        # Valid pipeline should validate without errors
        self.pipeline.validate({'x': 5})
        
        # Create invalid pipeline (missing dependency)
        p = Pipeline([
            Function(lambda y: y * 2, args=[Var('y')], out_var='z')  # 'y' is not provided
        ])
        
        # Validate should raise an error
        with self.assertRaises(AssertionError):
            p.validate({'x': 5})
    
    def test_copy(self):
        """Test copying the pipeline."""
        # Copy pipeline
        p_copy = self.pipeline.copy()
        
        # Check that it's a different object
        self.assertIsNot(p_copy, self.pipeline)
        
        # Check that it has the same number of steps
        self.assertEqual(len(p_copy), len(self.pipeline))
        
        # Check that the steps are copies, not the same objects
        self.assertIsNot(p_copy.to_list()[0], self.pipeline.to_list()[0])
        
        # Check that it works the same
        result = p_copy.run({'x': 5})
        self.assertEqual(result.get('z'), 12)


if __name__ == '__main__':
    unittest.main() 