.PHONY : tests code_tests

tests : code_tests

code_tests :
	py.test --cov openformat --cov-fail-under=100 --cov-report=term-missing --cov-report=html -v
