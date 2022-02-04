DRAUPNIR


https://stackoverflow.com/questions/53779509/upload-failed-403-invalid-or-non-existent-authentication-information-python
https://towardsdatascience.com/how-to-package-your-python-code-df5a7739ab2e
https://python-packaging-tutorial.readthedocs.io/en/latest/setup_py.html


CHANGES.txt: log of changes with each release

LICENSE.txt: text of the license you choose (do choose one!)

MANIFEST.in: description of what non-code files to include

README.txt: description of the package – should be written in ReST or Markdown (for PyPi):

setup.py: the script for building/installing package.

bin/: This is where you put top-level scripts

( some folks use scripts )

docs/: the documentation

package_name/: The main package – this is where the code goes.

test/: your unit tests.

setup.py: file is what describes your package, and tells setuptools how to package, build and install it
setup.cfg¶: Provides a way to give the end user some ability to customize the install