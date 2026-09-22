.PHONY: help install test figures notebook html clean

help:
	@echo "make install   install the package and notebook dependencies"
	@echo "make test      run the regression tests against the paper's numbers"
	@echo "make figures   render every figure to figures/"
	@echo "make notebook  regenerate and execute the notebook"
	@echo "make html      export the executed notebook to HTML"

install:
	python -m pip install -e ".[notebook,dev]"

test:
	python -m pytest -q

figures:
	python -c "import matplotlib; matplotlib.use('Agg'); import render_figures"

notebook:
	python build_notebook.py
	jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=600 notebooks/hodgkin-huxley-1952.ipynb

html: notebook
	jupyter nbconvert --to html notebooks/hodgkin-huxley-1952.ipynb

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache *.egg-info
	rm -f notebooks/*.html
