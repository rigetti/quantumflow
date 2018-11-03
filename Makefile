
.DEFAULT_GOAL := help

PROJECT = quantumflow
FILES = quantumflow examples tests tools setup.py

# Kudos: Adapted from Auto-documenting default target
# https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

test:		## Run unittests with current backend
	pytest --disable-pytest-warnings tests/

testall:	## Run full tox build and test
	tox

coverage:	## Report test coverage using current backend
	@echo
	pytest --disable-pytest-warnings --cov=quantumflow --cov-report term-missing tests/
	@echo
	@echo "** Note: Only active backend will have full test coverage **"
	@echo

lint:		## Delint python source
	flake8 $(FILES)

typecheck:	## Static typechecking 
	mypy -p quantumflow --ignore-missing-imports --disallow-untyped-defs

docs:		## Build documentation
	(cd docs; make html)
	open docs/build/html/index.html

doctest:	## Run doctests in documentation
	(cd docs; make doctest)

doccover:   ## Report documentation coverage
	(cd docs; make coverage && open build/coverage/python.txt)

docclean: 	## Clean documentation build
	(cd docs; make clean)

lines:	## Count lines of code (Includes blank lines)
	@wc -l quantumflow/*.py quantumflow/*/*.py examples/*.py tests/*.py setup.py

pragmas:	## Report all pragmas in code
	@echo
	@echo "** Code that needs something done **"
	@grep 'TODO' --color -r -n $(FILES) || echo "No TODO pragmas"
	@echo
	@echo "** Code that needs fixing **" || echo "No FIXME pragmas"
	@grep 'FIXME' --color -r -n $(FILES)
	@echo
	@echo "** Code that needs documentation **" || echo "No DOCME pragmas"
	@grep 'DOCME' --color -r -n $(FILES)
	@echo
	@echo "** Code that needs more tests **" || echo "No TESTME pragmas"
	@grep 'TESTME' --color -r -n $(FILES)
	@echo
	@echo "** Acknowledgments **"
	@grep 'kudos:' --color -r -n -i $(FILES) || echo "No kudos"
	@echo
	@echo "** Pragma for test coverage **"
	@grep 'pragma: no cover' --color -r -n $(FILES) || echo "No Typecheck Pragmas"
	@echo
	@echo "** flake8 linting pragmas **"
	@echo "(http://flake8.pycqa.org/en/latest/user/error-codes.html)"
	@grep '# noqa:' --color -r -n $(FILES) || echo "No flake8 pragmas"
	@echo
	@echo "** Typecheck pragmas **"
	@grep '# type:' --color -r -n $(FILES) || echo "No Typecheck Pragmas"

meta:	## Report versions of dependent packages
	@echo
	@python -m quantumflow.meta

.PHONY: help
.PHONY: docs
.PHONY: build