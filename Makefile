.PHONY: install test lint format package
install:
	python -m pip install -e '.[ocr,test]'
test:
	pytest
lint:
	ruff check src tests tools
format:
	ruff format src tests tools
package:
	python -m build
