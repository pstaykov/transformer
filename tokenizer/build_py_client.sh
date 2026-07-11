python3 -m venv venv
source venv/bin/activate
pip install pybind11 setuptools
pip install . --no-build-isolation
echo "Erfolgreiche instalation"