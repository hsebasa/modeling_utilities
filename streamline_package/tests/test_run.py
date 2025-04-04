from streamline.pipeline import Pipeline, StepNotFound, concat, Function, Var
from streamline.pipeline.pipeline import _Vertical
from streamline.delayed import Delayed, step

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
            Function(lambda *a, **kw: 3, arg_cat='s')
        ]).add_step(
            step=Function(
                fun=lambda x, s, e, f, h: (x**2+1, s, e, f, h),
                args=[Var('a')],
                kw={'e': Var('a'), 'f': 8, 'h': 5, 's': 'd', 'e': 'n', 'h': 3},
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
            
        def dummy2():
            myid = id(self)
            return myid, [myid]
        
        pipe = Pipeline([
            Function(lambda *args, **kw: 3, arg_cat='s')
        ]).add_function(
            # kw={'s': 'd', 'e': 'n', 'h': 3},
            a=Function(
                fun=lambda x, s, e, f, h: (x**2+1, s, e, f, h),
                args=[Var('a')],
                kw={'e': Var('a'), 'f': 8, 'h': 5},
                out_var=('b', 'c', 'e', 'f', 'h'),
                arg_cat='f',
            )
        ).add_import_lib(
            {'numpy': 'np', 'pandas': 'pd'},
            index=0
        ).add_function(Function(dummy, out_var='r')).add_function(dummy2)
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
        res = sl.eval_delay(d, env={'step': Function(arg_cat='s', fun=lambda x: x, tags={1, 2})})
        assert res

        res = sl.eval_delay(step.tags.contains(3), env={'step': Function(arg_cat='s', fun=lambda x: x, tags={1, 2})})
        assert not res

    def test_rename(self):
        pipe2 = Pipeline([Function(lambda a, b, c: (a, b, c), args=[Var('s')], kw={'b': Var('t'), 'c': 4})]).add_delete([Var('m')])
        pipe3 = pipe2.copy(deep=True)
        pipe3.rename({'s': 'n', 't': 'a', 'm': 'd'})
        
        run2 = pipe2.run({'s': 2, 't': 3, 'm': 5})
        assert run2['s'] == 2
        assert run2['t'] == 3
        assert 'm' not in run2
        assert run2['_'] == (2, 3, 4)

        run3 = pipe3.run({'n': 2, 'a': 3, 'd': 4})
        assert run3['n'] == 2
        assert run3['a'] == 3
        assert 'd' not in run3
        assert 's' not in run3
        assert 't' not in run3
        assert run3['_'] == (2, 3, 4)
    
    def test_rename_with_lambda(self):
        pipe2 = Pipeline([Function(lambda a, b, c: (a, b, c), args=[Var('s')], kw={'b': Var('t'), 'c': 4})]).add_delete([Var('m')])
        pipe3 = pipe2.copy(deep=False)
        pipe3.rename(lambda x: 'pipe3_'+x, arg_cat='step1')

        run2 = pipe2.run({'s': 2, 't': 3, 'm': 5})
        assert run2['s'] == 2
        assert run2['t'] == 3
        assert 'm' not in run2
        assert run2['_'] == (2, 3, 4)

        run3 = pipe3.run({'pipe3_s': 2, 'pipe3_t': 3, 'pipe3_m': 5})
        assert run3['pipe3_s'] == 2
        assert run3['pipe3_t'] == 3
        assert 's' not in run3
        assert 't' not in run3
        assert run3['_'] == (2, 3, 4)

    def test_add_variables(self):
        pipe = Pipeline([
            Function(lambda *args, **kw: 3, arg_cat='s')
        ]).add_variables_dict({'a': 1, 'b': Var('d')})
        
        env = pipe.run({'d': 5})
        self.assertEqual(env['a'], 1)
        self.assertEqual(env['b'], 5)
    
    def test_env_info(self):
        pipe = Pipeline([
            Function(lambda : 3, arg_cat='s')
        ]).add_step(
            step=Function(
                fun=lambda x, s, e, f, h: (x**2+1, s, e, f, h),
                args=Var('a'),
                kw={'e': Var('a'), 'f': 8, 'h': 5},
                out_var=('b', 'c', 'e', 'f', 'h'),
                arg_cat='f',
            )
        ).add_import_lib(
            {'numpy': 'np', 'pandas': 'pd'},
            index=0
        ).add_step(
            Function(np.random.rand, out_var='r')
        ).add_delete(
            ['f', 'h', 'a']  # remove variables 'f' and 'h'
        ).add_variables_dict({
            'j': '3'
        })
        
        env_info = pipe.get_env_info()
        res_env_info = [
            {'required_vars': set(),
            'unresolved_vars': set(),
            'unresolved_vars_accum': set(),
            'env_vars': set(),
            'removed_vars': set(),
            'added_vars': {'np', 'pd'},
            'steptype': 'Function',
            'arg_cat': '',
            'tags': {'function', 'import_lib'}},
            {'required_vars': set(),
            'unresolved_vars': set(),
            'unresolved_vars_accum': set(),
            'env_vars': {'np', 'pd'},
            'removed_vars': set(),
            'added_vars': set(),
            'steptype': 'Function',
            'arg_cat': 's',
            'tags': {'function'}},
            {'required_vars': {'a'},
            'unresolved_vars': {'a'},
            'unresolved_vars_accum': {'a'},
            'env_vars': {'np', 'pd'},
            'removed_vars': set(),
            'added_vars': {'b', 'c', 'e', 'f', 'h'},
            'steptype': 'Function',
            'arg_cat': 'f',
            'tags': {'function'}},
            {'required_vars': set(),
            'unresolved_vars': set(),
            'unresolved_vars_accum': {'a'},
            'env_vars': {'b', 'c', 'e', 'f', 'h', 'np', 'pd'},
            'removed_vars': set(),
            'added_vars': {'r'},
            'steptype': 'Function',
            'arg_cat': '',
            'tags': {'function'}},
            {'required_vars': set(),
            'unresolved_vars': set(),
            'unresolved_vars_accum': {'a'},
            'env_vars': {'b', 'c', 'e', 'f', 'h', 'np', 'pd', 'r'},
            'removed_vars': {'a', 'f', 'h'},
            'added_vars': set(),
            'steptype': 'Delete',
            'arg_cat': '',
            'tags': {'delete'}},
            {'required_vars': set(),
            'unresolved_vars': set(),
            'unresolved_vars_accum': {'a'},
            'env_vars': {'b', 'c', 'e', 'np', 'pd', 'r'},
            'removed_vars': set(),
            'added_vars': {'j'},
            'steptype': 'Function',
            'arg_cat': '',
            'tags': {'add_variables'}}
        ]
        assert env_info == res_env_info

        pipe4 = pipe.copy()
        pipe4.rename(lambda x: 'pipe4_'+x)
        pipe4.get_env_info()

        env_info4 = pipe4.get_env_info()
        res_env_info4 = [
            {'required_vars': set(),
            'unresolved_vars': set(),
            'unresolved_vars_accum': set(),
            'env_vars': set(),
            'removed_vars': set(),
            'added_vars': {'pipe4_np', 'pipe4_pd'},
            'steptype': 'Function',
            'arg_cat': '',
            'tags': {'function', 'import_lib'}},
            {'required_vars': set(),
            'unresolved_vars': set(),
            'unresolved_vars_accum': set(),
            'env_vars': {'pipe4_np', 'pipe4_pd'},
            'removed_vars': set(),
            'added_vars': set(),
            'steptype': 'Function',
            'arg_cat': 's',
            'tags': {'function'}},
            {'required_vars': {'pipe4_a'},
            'unresolved_vars': {'pipe4_a'},
            'unresolved_vars_accum': {'pipe4_a'},
            'env_vars': {'pipe4_np', 'pipe4_pd'},
            'removed_vars': set(),
            'added_vars': {'pipe4_b', 'pipe4_c', 'pipe4_e', 'pipe4_f', 'pipe4_h'},
            'steptype': 'Function',
            'arg_cat': 'f',
            'tags': {'function'}},
            {'required_vars': set(),
            'unresolved_vars': set(),
            'unresolved_vars_accum': {'pipe4_a'},
            'env_vars': {'pipe4_b',
            'pipe4_c',
            'pipe4_e',
            'pipe4_f',
            'pipe4_h',
            'pipe4_np',
            'pipe4_pd'},
            'removed_vars': set(),
            'added_vars': {'pipe4_r'},
            'steptype': 'Function',
            'arg_cat': '',
            'tags': {'function'}},
            {'required_vars': set(),
            'unresolved_vars': set(),
            'unresolved_vars_accum': {'pipe4_a'},
            'env_vars': {'pipe4_b',
            'pipe4_c',
            'pipe4_e',
            'pipe4_f',
            'pipe4_h',
            'pipe4_np',
            'pipe4_pd',
            'pipe4_r'},
            'removed_vars': {'pipe4_a', 'pipe4_f', 'pipe4_h'},
            'added_vars': set(),
            'steptype': 'Delete',
            'arg_cat': '',
            'tags': {'delete'}},
            {'required_vars': set(),
            'unresolved_vars': set(),
            'unresolved_vars_accum': {'pipe4_a'},
            'env_vars': {'pipe4_b',
            'pipe4_c',
            'pipe4_e',
            'pipe4_np',
            'pipe4_pd',
            'pipe4_r'},
            'removed_vars': set(),
            'added_vars': {'pipe4_j'},
            'steptype': 'Function',
            'arg_cat': '',
            'tags': {'add_variables'}}
        ]
        self.assertEqual(env_info4, res_env_info4)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.step1 = Function(fun=lambda : 1, arg_cat='step1')
        self.step2 = Function(fun=lambda : 2)
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
        step3 = Function(fun=lambda env, kw: 3)
        self.pipeline._append(step3)
        self.assertEqual(len(self.pipeline.to_list()), 3)

    def test_pipeline_apply(self):
        def dummy_func(step):
            return step
        result = self.pipeline.map(dummy_func)
        self.assertIsInstance(result, _Vertical)
        self.assertEqual(len(result), 2)

    def test_pipeline_getitem(self):
        result = self.pipeline._getitem([0, 1])
        self.assertEqual(result, [self.step1, self.step2])

    def test_pipeline_delitem(self):
        self.pipeline._delitem([0])
        self.assertEqual(len(self.pipeline.to_list()), 1)

    def test_pipeline_setitem(self):
        step3 = Function(fun=lambda env, kw: 3)
        self.pipeline._setitem([0], step3)
        self.assertEqual(self.pipeline.to_list()[0], step3)

    def test_pipeline_insert(self):
        step3 = Function(fun=lambda env, kw: 3)
        self.pipeline._insert(1, step3)
        self.assertEqual(self.pipeline.to_list()[1], step3)

    def test_pipeline_add_step(self):
        step3 = Function(fun=lambda env, kw: 3)
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
