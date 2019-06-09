.PHONY : tests code_tests docs docs/api

tests : code_tests docs

code_tests :
	py.test --cov openformat --cov-fail-under=100 --cov-report=term-missing --cov-report=html -v

requirements.txt : requirements.in setup.py
	pip-compile -v $<

docs :
	sphinx-build docs docs/build

docs/api :
	sphinx-apidoc -o docs . setup.py
