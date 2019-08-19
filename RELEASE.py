from os import system as cmd

cmd("python3 setup.py sdist bdist_wheel")
cmd("python3 -m twine upload dist/*")
