.PHONY : tests code_tests docs

tests : code_tests docs

code_tests :
	py.test --cov openformat --cov-fail-under=100 --cov-report=term-missing --cov-report=html -v

requirements.txt : requirements.in setup.py
	pip-compile -v $<

docs :
	sphinx-apidoc -o docs . setup.py
	sphinx-build docs docs/build
