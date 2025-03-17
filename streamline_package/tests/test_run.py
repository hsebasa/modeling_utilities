from streamline.pipeline.pipeline import Pipeline, Step, StepNotFound, _Vertical, concat
from streamline.delayed import Delayed, step
from streamline.functions import Function

import streamline as sl

import pandas as pd
import numpy as np
import unittest
import tempfile
import json
import os


class TestStringMethods(unittest.TestCase):
    def test_run_env(self):
        folder = tempfile.TemporaryDirectory().name
        renv = sl.RunEnv({
            'a': 3,
            'df': pd.DataFrame([[4]])
        })
        
        renv.save(
            folder
        )
        
        assert os.path.exists(os.path.join(folder, 'dedicated'))
        assert os.path.exists(os.path.join(folder, 'dedicated', 'df.parquet'))
        assert os.path.exists(os.path.join(folder, 'info.json'))
        assert os.path.exists(os.path.join(folder, 'env.dill'))

        with open(os.path.join(folder, 'info.json')) as f:
            info = json.loads(f.read())
            
        assert info['version'] == sl.__version__
        assert info['dedicated']['df']['type'] == 'pandas_df'
        assert info['saved_vars'] == ['a', 'df']

        renv2 = sl.load_runenv(folder)
        assert renv2['a'] == renv['a']
        assert renv2['df'].values.tolist() == renv['df'].values.tolist()
        
    def test_pipeline(self):
        pipe = sl.Pipeline([
            sl.Step(lambda env, kw: 3, 's')
        ]).add_step(
            kw={'s': 'd', 'e': 'n', 'h': 3},
            a=Function(
                fun=lambda x, s, e, f, h: (x**2+1, s, e, f, h),
                in_vars='a',
                in_vars_kw={'e': 'a'},
                in_kw={'f': 8, 'h': 5},
                out_var=('b', 'c', 'e', 'f', 'h')
            )
        )
        env = pipe.run(sl.RunEnv({'a': 3, 'e': 'k'}))
        assert env['a'] == 3
        assert env['e'] == 'n'
        assert env['b'] == 10
        assert env['c'] == 'd'
        assert env['f'] == 8
        assert env['h'] == 3

        pipe2 = env.gen_pipeline()
        env2 = pipe2.run(sl.RunEnv({'a': 3}))
        assert all([env[a] == env2[a] for a in ['a', 'e', 'b', 'c', 'f', 'h']])

    def test_pipeline2(self):
        def dummy():
            return np.random.rand()
            
        def dummy2(env, kw):
            env['myid'] = id(env)
            env['lst'].append(id(env))
        
        pipe = Pipeline([
            Step(lambda env, kw: 3, 's')
        ]).add_step(
            arg_cat='f',
            # kw={'s': 'd', 'e': 'n', 'h': 3},
            a=Function(
                fun=lambda x, s, e, f, h: (x**2+1, s, e, f, h),
                in_vars='a',
                in_vars_kw={'e': 'a'},
                in_kw={'f': 8, 'h': 5},
                out_var=('b', 'c', 'e', 'f', 'h')
            )
        ).add_import_lib(
            {'numpy': 'np', 'pandas': 'pd'},
            index=0
        ).add_step(Function(dummy, out_var='r')).add_step(dummy2)
        env = pipe.run({'a': 3, 'e': 'k', 'lst': []}, {'f_s': 'd', 'f_e': 'n', 'f_h': 3})
        
        assert env['a'] == 3
        assert env['e'] == 'n'
        assert env['b'] == 10
        assert env['c'] == 'd'
        assert env['f'] == 8
        assert env['h'] == 3
    
    def test_delay(self):
        step = sl.Delayed(prefix='step')
        d = sl.delay_lib.type((step.tags.contains(3)) & (step.tags.contains(1))).__name__.isin(['bool'])
        res = sl.eval_delay(d, env={'step': sl.Step(arg_cat='s', fun=lambda x: x, tags={1, 2})})
        assert res

        res = sl.eval_delay(step.tags.contains(3), env={'step': sl.Step(arg_cat='s', fun=lambda x: x, tags={1, 2})})
        assert not res
        

class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.step1 = Step(fun=lambda env, kw: 1, arg_cat='step1')
        self.step2 = Step(fun=lambda env, kw: 2)
        self.pipeline = Pipeline(a=[self.step1, self.step2])

    def test_pipeline_initialization(self):
        self.assertEqual(len(self.pipeline.to_list()), 2)

    def test_pipeline_print(self):
        self.assertIn('Pipeline(steps=[', self.pipeline.print())

    def test_pipeline_repr(self):
        self.assertIn('Pipeline(steps=[', repr(self.pipeline))

    def test_pipeline_items(self):
        self.assertEqual(list(self.pipeline.items()), [self.step1, self.step2])

    def test_pipeline_append(self):
        step3 = Step(fun=lambda env, kw: 3)
        self.pipeline._append(step3)
        self.assertEqual(len(self.pipeline.to_list()), 3)

    def test_pipeline_apply(self):
        def dummy_func(step):
            return step
        result = self.pipeline.apply(dummy_func)
        self.assertIsInstance(result, _Vertical)
        self.assertEqual(len(result), 2)

    def test_pipeline_getitem(self):
        result = self.pipeline._getitem([0, 1])
        self.assertEqual(result, [self.step1, self.step2])

    def test_pipeline_delitem(self):
        self.pipeline._delitem([0])
        self.assertEqual(len(self.pipeline.to_list()), 1)

    def test_pipeline_setitem(self):
        step3 = Step(fun=lambda env, kw: 3)
        self.pipeline._setitem([0], step3)
        self.assertEqual(self.pipeline.to_list()[0], step3)

    def test_pipeline_insert(self):
        step3 = Step(fun=lambda env, kw: 3)
        self.pipeline._insert(1, step3)
        self.assertEqual(self.pipeline.to_list()[1], step3)

    def test_pipeline_add_step(self):
        step3 = Step(fun=lambda env, kw: 3)
        self.pipeline.add_step(step3)
        self.assertEqual(len(self.pipeline.to_list()), 3)

    def test_pipeline_run(self):
        env = self.pipeline.run()
        self.assertIsNotNone(env)

    def test_pipeline_run_parallel(self):
        envs = self.pipeline.run_parallel()
        self.assertIsInstance(envs, list)

    def test_concat(self):
        pipeline2 = Pipeline(a=[self.step1])
        combined_pipeline = concat([self.pipeline, pipeline2])
        self.assertEqual(len(combined_pipeline.to_list()), 3)

    def test_pipeline_step_not_found(self):
        with self.assertRaises(StepNotFound):
            self.pipeline._getitem(999)

    def test_pipeline_empty(self):
        empty_pipeline = Pipeline()
        self.assertEqual(len(empty_pipeline.to_list()), 0)

    def test_pipeline_empty_run(self):
        empty_pipeline = Pipeline()
        env = empty_pipeline.run()
        self.assertIsNotNone(env)

    def test_pipeline_empty_run_parallel(self):
        empty_pipeline = Pipeline()
        envs = empty_pipeline.run_parallel()
        self.assertIsInstance(envs, list)

    def test_pipeline_loc_with_vertical(self):
        vertical = _Vertical([True, False])
        result = self.pipeline.loc[vertical]
        self.assertIsInstance(result, Pipeline)
        self.assertEqual(result._Pipeline__steps, [self.step1])

    def test_pipeline_loc_with_delayed(self):
        delayed = Delayed(prefix='step')
        result = self.pipeline.loc[delayed.arg_cat=='step1']
        self.assertIsInstance(result, Pipeline)
        self.assertEqual(result._Pipeline__steps, [self.step1])

if __name__ == '__main__':
    unittest.main()
