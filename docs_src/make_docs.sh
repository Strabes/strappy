# Run sphinx autodoc from docs_src/
make clean
sphinx-apidoc -o ./source ../strappy
make html
rm -R ../docs/*
mv build/html ../docs/