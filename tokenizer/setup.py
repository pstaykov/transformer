from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Hier listen wir alle .cpp-Dateien auf, die der Compiler benötigt
cpp_sources = [
    "tools/encode_cli.cpp",   # Hier liegen deine Bindings und die main()
    "src/bpe_model.cpp",      # Hier liegt die eigentliche Tokenizer-Logik
    "src/pretokenizer.cpp",
    "src/unicode_utils.cpp",
]

ext_modules = [
    Pybind11Extension(
        "bbpe_tokenizer",
        cpp_sources,
        cxx_std=17,
        define_macros=[('PYBIND_BUILD', '1')],
        # Dem Compiler sagen, dass er im Ordner 'include' nach den .hpp Dateien suchen soll
        include_dirs=[
            "include",
            "tools"
        ],
    ),
]

setup(
    name="bbpe_tokenizer",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)