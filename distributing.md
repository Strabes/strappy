Make sure you have build and twine:

```
$ pip install build twine
```

Build the source archive and wheel:
```
$ python -m build
```

Check that package is ready for distribution:

```
$ twine check dist/*
```

Upload to TestPyPI:

```
twine upload -r testpypi dist/*
```