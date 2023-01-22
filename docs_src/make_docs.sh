# Run sphinx autodoc from docs_src/
make clean
sphinx-apidoc -o ./source ../strappy
make html
cp index.html ../docs
cp .nojekyll ../docs