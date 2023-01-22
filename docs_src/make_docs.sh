# Run sphinx autodoc from docs_src/
make clean
sphinx-apidoc -o ./source ../strappy
make html
mv ../docs/html/* ../docs/