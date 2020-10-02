import os

from nltk import word_tokenize
from nltk.corpus import stopwords
import pandas as pd


def filtre_mots_importants(texte, langue='french', ponctuation=True, personnages=True):
    stopWords = (
        set(stopwords.words(langue)) |
        {
            'a', 'ainsi', 'alors',
            'ça', 'car', 'cette', 'comme', "c'est",
            'dit', 'donc', 'dont', "d'un", "d'une",
            'elles',
            'leurs',
            'où',
            'plus',
            'quand', "qu'elle", "qu'il", "qu'on",
            'sans', "s'en", "s'était", 'si',
            'tous', 'tout', 'toute',
        }
    )
    noms_persos = set()
    if ponctuation:
        stopWords = (stopWords | {',', '"', '?', ';', '.', ':', '!', '—', '...', '«', '»', '(', ')', "'"})
    if personnages:
        noms_persos = {
            'Antoine', 'Aristide',
            'Catherine', 'Charles',
            'Félicité',
            'Gervaise', 'Granoux',
            'Hélène', 'Henri',
            'Jean',
            'Lantier', 'Lisa',
            'M.', 'Marie', 'Marthe', 'Maxime', 'Miette', 'Mouret',
            'Pascal', 'Pierre',
            'Roudier',
            'Sidonie', 'Silvère',
            'Vuillet',
        }
    mots = word_tokenize(texte, language=langue)
    return [mot for mot in mots if ((mot.lower() not in stopWords) and (mot not in noms_persos))]


def pretraitement_texte(fichier, **kwargs):
    fileText = open(os.path.join('books', fichier), 'r')
    text = fileText.read()
    fileText.close()

    text = text.replace('…', '...').replace('’', "'").replace("'", "'")
    mots = filtre_mots_importants(text, **kwargs)
    return pd.Series(mots).value_counts()


def regroupement_statistiques(livres):
    # livres: list of text files names
    #
    # Output:
    # dataframe which index is made of words of all the texts, columns are books names and values are frequencies
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