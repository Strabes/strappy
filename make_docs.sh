# Run sphinx autodoc from docs/
make clean
sphinx-apidoc -o source strappy
make html
