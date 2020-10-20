import os

from vocabulary_utils import regroupement_statistiques, mots_specifiques

def main():
    titres = os.listdir('books')
    statistiques = regroupement_statistiques(titres)
    mots_specifiques(statistiques)


if __name__ == "__main__":
    main()
