release: setup.py requirements.txt
	python3 setup.py sdist bdist_wheel	
	twine upload --skip-existing dist/*
