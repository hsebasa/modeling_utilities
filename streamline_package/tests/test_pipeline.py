import unittest
import os
import tempfile
import pandas as pd

from streamline.pipeline.pipeline import Pipeline, StepNotFound
from streamline.pipeline.step import Function, Delete, VariablesDict, Var
from streamline.pipeline.run_env import RunEnv
from streamline.pipeline.modeling import BaseModel


class MockModel(BaseModel):
    def __init__(self):
        self.fitted = False

    def fit(self, X, y):
        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise ValueError("Model is not fitted yet.")
        return [x * 2 for x in X]


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

    # def test_get_subgraph_comprehensive(self):
    #     """Test the get_subgraph method with various input/output configurations and edge cases."""
    #     # Create a complex pipeline with branching logic and multiple dependencies
    #     p = Pipeline([
    #         # Step 0: Initialization
    #         VariablesDict({'initial': 10, 'config': 'test'}),
            
    #         # Step 1: First transformation
    #         Function(lambda initial: initial * 2, args=[Var('initial')], out_var='doubled'),
            
    #         # Step 2: Second transformation using initial and doubled
    #         Function(lambda initial, doubled: initial + doubled, args=[Var('initial'), Var('doubled')], out_var='combined'),
            
    #         # Step 3: A branch that only depends on initial
    #         Function(lambda initial: initial ** 2, args=[Var('initial')], out_var='squared'),
            
    #         # Step 4: A branch that only depends on doubled
    #         Function(lambda doubled: doubled - 5, args=[Var('doubled')], out_var='adjusted'),
            
    #         # Step 5: A transform using two intermediate results
    #         Function(lambda combined, squared: combined / squared, args=[Var('combined'), Var('squared')], out_var='ratio'),
            
    #         # Step 6: Cleaning up intermediate results
    #         Delete([Var('doubled'), Var('squared')]),
            
    #         # Step 7: Final calculation using multiple inputs
    #         Function(lambda adjusted, ratio, config: f"{config}: {adjusted + ratio}", 
    #                 args=[Var('adjusted'), Var('ratio'), Var('config')], out_var='result')
    #     ])

    #     # Test 1: Full pipeline - with specified inputs and outputs (should return the whole pipeline)
    #     full_graph = p.get_subgraph(['initial', 'config'], ['result'])
    #     self.assertEqual(len(full_graph), len(p)-2, "Full graph should have the same number of steps as the original pipeline minus the variable assignment and delete steps")
        
    #     # Run the full subgraph to verify it works
    #     result = full_graph.run({'initial': 10, 'config': 'test'})
    #     self.assertEqual(result.get('result'), 'test: 15.3', "Full subgraph should produce the correct result")

    #     # Test 2: Subgraph from middle - extracting only the steps needed for a specific output from a specific input
    #     mid_graph = p.get_subgraph(['doubled'], ['ratio'])
    #     # This should include steps for: combined (using doubled), squared (needed for ratio), and ratio calculation
    #     self.assertGreater(len(mid_graph), 0, "Mid graph should have at least one step")
    #     self.assertLess(len(mid_graph), len(p), "Mid graph should have fewer steps than the original pipeline")
        
    #     # Run the mid subgraph to verify it works correctly
    #     result = mid_graph.run({'doubled': 20, 'initial': 10})  # Initial is needed for squared
    #     self.assertIn('ratio', result, "Mid subgraph should produce the 'ratio' output")
    #     self.assertEqual(result.get('ratio'), 30/100, "Mid subgraph should calculate the correct ratio")

    #     # Test 3: Complex branch - following one specific branch of the computation
    #     branch_graph = p.get_subgraph(['initial'], ['adjusted'])
    #     # This should only include steps that transform initial to doubled and doubled to adjusted
    #     self.assertGreater(len(branch_graph), 0, "Branch graph should have at least one step")
    #     self.assertLess(len(branch_graph), len(p), "Branch graph should have fewer steps than the original pipeline")
        
    #     # Run the branch subgraph
    #     result = branch_graph.run({'initial': 10})
    #     self.assertEqual(result.get('adjusted'), 15, "Branch subgraph should calculate the correct adjusted value")

    #     # Test 4: Multiple outputs - tracking back from multiple output variables
    #     multi_out_graph = p.get_subgraph(['initial'], ['adjusted', 'ratio'])
    #     # This should include all steps needed for both adjusted and ratio
    #     self.assertGreater(len(multi_out_graph), 0, "Multi-output graph should have at least one step")
        
    #     # Run the multi-output subgraph
    #     result = multi_out_graph.run({'initial': 10, 'config': 'test'})
    #     self.assertIn('adjusted', result, "Multi-output subgraph should produce the 'adjusted' output")
    #     self.assertIn('ratio', result, "Multi-output subgraph should produce the 'ratio' output")

    #     # Test 5: Empty subgraph - when inputs or outputs don't exist
    #     empty_graph = p.get_subgraph(['nonexistent'], ['result'])
    #     self.assertEqual(len(empty_graph), 0, "Subgraph with nonexistent input should be empty")
        
    #     empty_graph = p.get_subgraph(['initial'], ['nonexistent'])
    #     self.assertEqual(len(empty_graph), 0, "Subgraph with nonexistent output should be empty")
        
    #     # Test 6: Skip delete steps - ensure delete steps don't interfere with backward dependency tracking
    #     skip_delete_graph = p.get_subgraph(['initial'], ['result'])
    #     # The subgraph should include the Delete step if it's part of the path to the output
    #     delete_steps = [step for step in skip_delete_graph if isinstance(step, Delete)]
    #     self.assertTrue(len(delete_steps) <= 1, "Subgraph should properly handle Delete steps")
        
    #     # Test 7: Direct input to output - when an input is directly used as an output
    #     direct_graph = p.get_subgraph(['config'], ['result'])
    #     self.assertGreater(len(direct_graph), 0, "Direct input-to-output graph should have at least one step")
        
    #     # Run the direct subgraph with all required inputs
    #     result = direct_graph.run({'config': 'test', 'adjusted': 15, 'ratio': 0.3})
    #     self.assertEqual(result.get('result'), 'test: 15.3', "Direct subgraph should produce the correct result")
        
    #     # Test 8: Empty pipeline
    #     empty_pipeline = Pipeline()
    #     empty_subgraph = empty_pipeline.get_subgraph(['x'], ['y'])
    #     self.assertEqual(len(empty_subgraph), 0, "Subgraph of empty pipeline should be empty")
    

class TestPipelineAddModel(unittest.TestCase):
    def setUp(self):
        self.pipeline = Pipeline()

    def test_add_model_with_fit_and_predict(self):
        model = MockModel
        self.pipeline.add_model(
            model=model,
            args=[],
            out_var="model_instance",
            fit_args=["X_train", "y_train"],
            fit_out_var="fitted_model",
            predict_run=True,
            predict_args=["X_test"],
            predict_out_var="predictions"
        )

        steps = self.pipeline.to_list()
        self.assertEqual(len(steps), 3)  # Instance, fit, and predict steps
        self.assertIsInstance(steps[0], Function)
        self.assertIsInstance(steps[1], Function)
        self.assertIsInstance(steps[2], Function)

    def test_add_model_without_predict(self):
        model = MockModel
        self.pipeline.add_model(
            model=model,
            args=[],
            out_var="model_instance",
            fit_args=["X_train", "y_train"],
            fit_out_var="fitted_model",
            predict_run=False
        )

        steps = self.pipeline.to_list()
        self.assertEqual(len(steps), 2)  # Instance and fit steps
        self.assertIsInstance(steps[0], Function)
        self.assertIsInstance(steps[1], Function)

    def test_model_execution(self):
        model = MockModel
        self.pipeline.add_model(
            model=model,
            args=[],
            out_var="model_instance",
            fit_args=[Var("X_train"), Var("y_train")],
            fit_out_var="fitted_model",
            predict_run=True,
            predict_args=[Var("X_test")],
            predict_out_var="predictions"
        )

        env = {
            "X_train": [1, 2, 3],
            "y_train": [2, 4, 6],
            "X_test": [4, 5]
        }
        result = self.pipeline.run(env=env)
        self.assertTrue(result["fitted_model"].fitted)
        self.assertEqual(result["predictions"], [8, 10])


class TestPipelineGenPredictor(unittest.TestCase):
    def setUp(self):
        self.pipeline = Pipeline()
        self.pipeline.add_step(Function(lambda x: x + 1, args=[Var('x')], out_var='y'))
        self.pipeline.add_step(Function(lambda y: y * 2, args=[Var('y')], out_var='z'))

    def test_gen_predictor(self):
        predictor = self.pipeline.gen_predictor(out_var='z')
        self.assertTrue(hasattr(predictor, 'predict'))

        result = predictor.predict(x=5)
        self.assertEqual(result, 12)  # (5 + 1) * 2 = 12

    def test_gen_predictor_missing_dependency(self):
        predictor = self.pipeline.gen_predictor(out_var='z')
        with self.assertRaises(AssertionError) as context:
            predictor.predict()  # Missing 'x'
        self.assertIn("Missing dependencies for prediction", str(context.exception))

    def test_gen_predictor_with_copy_env(self):
        predictor = self.pipeline.gen_predictor(out_var='z', copy_env=True)
        env = {'x': 5}
        result = predictor.predict(**env)
        self.assertEqual(result, 12)  # (5 + 1) * 2 = 12
        self.assertEqual(env, {'x': 5})  # Ensure the original environment is not modified

    def test_gen_predictor_without_copy_env(self):
        predictor = self.pipeline.gen_predictor(out_var='z', copy_env=False)
        env = {'x': 5}
        env = predictor._predict(**env)
        result = env['z']
        self.assertEqual(result, 12)  # (5 + 1) * 2 = 12
        self.assertIn('y', env)  # Ensure the environment is modified
        self.assertEqual(env['y'], 6)  # Intermediate result

if __name__ == '__main__':
    unittest.main()