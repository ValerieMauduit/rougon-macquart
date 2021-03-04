import os

import kwargs as kwargs
from nltk import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

from sets import MY_STOP_WORDS, PERSONNAGES, PONCTUATION


def filtre_mots_importants(texte, langue='french', ponctuation=True, personnages=True):
    stopWords = (set(stopwords.words(langue)) | MY_STOP_WORDS)
    noms_persos = set()
    if ponctuation:
        stopWords = (stopWords | PONCTUATION)
    if personnages:
        noms_persos = PERSONNAGES
    mots = word_tokenize(texte, language=langue)
    return [mot for mot in mots if ((mot.lower() not in stopWords) and (mot not in noms_persos))]


def pretraitement_texte(fichier, filtre=True **kwargs):
    fileText = open(os.path.join('books', fichier), 'r')
    text = fileText.read()
    fileText.close()

    text = text.replace('…', '...').replace('’', "'").replace("'", "'")
    if filtre:
        mots = filtre_mots_importants(text, **kwargs)
    else:
        mots = word_tokenize(text, **kwargs)

    return pd.Series(mots).value_counts()

# TODO: analyse des mots pour voir si je les prends en minuscules ou toujours en majuscules
# TODO: passage de paramètres à vérifier
# TODO: les n premiers mots si je retire le mot pour peu qu'il soit dans un autre top
# TODO: tutorials nltk sur ce jeu de données
# TODO: un fichier de prétraitement du texte :
#      changer les caractères de merde / filtrer les mots à retirer /
#      passer en minuscules tous les mots qui le mériteraient / tokénizer séparément /
#      compter les fréquences ou occurences pour un seul texte
# TODO: des utilitaires de comparaison de textes, séparés
# TODO: répertoire en dur !

def regroupement_statistiques(livres):
    # livres: list of text files names
    #
    # Output:
    # dataframe which index is made of words of all the texts, columns are books names and values are frequencies
    # TODO: options garder ou pas 
    stats = pd.DataFrame()
    for titre in livres:
        comptage = pretraitement_texte(titre, personnages=True)
        stats[titre.split('.')[0]] = comptage
    stats.fillna(0, inplace=True)
    stats['TOTAL'] = stats.sum(axis=1)
    total = stats.sum()
    total.name = '### TOTAL ###'
    stats = stats / total * 100000
    # stats = stats.append(total)
    stats.sort_values(by='TOTAL', ascending=False, inplace=True)

    return stats


def mots_specifiques(statistiques):
    livres = statistiques.columns[:-1]
    statistiques['std'] = statistiques.iloc[:, :-1].std(axis=1)
    statistiques = statistiques.round()

    print('=' * 120)
    print('Les quarante-deux mots globalement les plus ordinaires')
    statistiques.sort_values(by='TOTAL', ascending=False, inplace=True)
    courants = statistiques.loc[
        (statistiques['TOTAL'] >= statistiques['TOTAL'].median()) &
        (statistiques['std'] <= statistiques['std'].median())
        ].copy()
    courants.sort_values(by='TOTAL', ascending=False, inplace=True)
    for n in range(7):
        print("".join(str.ljust(m, 20) for m in courants.index[(n * 6): ((n + 1) * 6)]))

    print('=' * 120)
    print('Les quarante-deux mots les plus discriminants')
    statistiques.sort_values(by=['std', 'TOTAL'], ascending=False, inplace=True)
    for n in range(7):
        print("".join(str.ljust(m, 20) for m in statistiques.index[(n * 6): ((n + 1) * 6)]))

    print('=' * 120)
    for livre in livres:
        print('-' * 120)
        print(livre)
        print('Les douze mots les plus spécifiques du bouquin')
        onevsall = statistiques[[livre]].copy()
        onevsall['autres'] = statistiques[[col for col in livres if col != livre]].mean(axis=1)
        onevsall['diff'] = onevsall[livre] - onevsall['autres']
        onevsall.sort_values(by='diff', ascending=False, inplace=True)
        print("".join(str.ljust(m, 20) for m in onevsall.index[:6]))
        print("".join(str.ljust(m, 20) for m in onevsall.index[6:12]))


def top_uniques(text):
    pass
