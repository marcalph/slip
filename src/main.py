import pathlib

languages = """
    es
    fr
    de
""".split()

splits = """
    train
    test
    dev
""".split()


datadir = pathlib.Path("./common-voice/cv-corpus")

langs = tuple(sorted(languages))
lang2target = {lang: target for target, lang in enumerate(langs)}

print("lang2target:", lang2target)
print("langs:", langs)
