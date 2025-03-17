

# Test

```
python -m unittest discover
```

## Graphical report
Run graphical tests by installing nose
```
pip install nose2[coverage_plugin]
pip install nose2-html-report
```

And then
```
nose2 --with-coverage --coverage-report html --coverage ./
```