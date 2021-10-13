from src.distillation.monolingual import Monolingual
a = []
languages = ["eu", "eu_gn"]
for i in languages:
    pot_lang = i.split("_")
    for lang in pot_lang:
        a.append(lang)

a = list(set(a))

print(a)