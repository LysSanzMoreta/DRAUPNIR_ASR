Data from

https://academic.oup.com/bioinformatics/article/24/3/333/253952#supplementary-data


#Package configuration files CHANGES.txt: log of changes with each release LICENSE.txt: text of the license you choose (do choose one!)
MANIFEST.in: description of what non-code files to include README.txt: description of the package – should be written in ReST or Markdown (for PyPi): setup.py: the script for building/installing package. Describes your package, and tells setuptools how to package, build and install it bin/: This is where you put top-level scripts docs/: the documentation package_name/: The main package – this is where the code goes. test/: your unit tests. setup.cfg¶: Provides a way to give the end user some ability to customize the install

#Steps to build package https://www.youtube.com/watch?v=0qXXsP5d3hU
cd folder with setup.py 
python3 -m pip install --upgrade build
python3 pip install --upgrade twine 
python -m twine upload --repository pypi dist/* -u ... -p ... # for new version of package python -m twine upload --skip-existing --repository pypi dist/* -u ... -p ... #to re-upload same version