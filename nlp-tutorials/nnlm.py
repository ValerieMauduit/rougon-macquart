# Neural Network Language Model
# Ce que j'essaie de reproduire : https://github.com/graykode/nlp-tutorial

import torch
import torch.nn as nn
import torch.optim as optim

class Text():
    def __init__(self, fic_name, not_word_characters='', case_sensitive=True, end_sentences='.?!'):
        self._fic_name = fic_name

        with open(fic_name, 'r') as fic:
            self.text = fic.read()
        if not case_sensitive:
            self.text = self.text.lower()

        self.paragraphs = self.text.split('\n\n')
        self.paragraphs = [x.replace('\n', '').replace('\t', '') for x in self.paragraphs if len(x) > 0]
        self.text = '\n\n'.join(self.paragraphs)

        paragraphs = self.paragraphs.copy()
        for character in end_sentences:
            paragraphs = [paragraph.replace(character, '|').strip('|') for paragraph in paragraphs]
        self.sentences = [[sentence.strip() for sentence in paragraph.split('|')] for paragraph in paragraphs]

        sentences = self.sentences.copy()
        for character in not_word_characters:
            sentences = [
                [sentence.replace(character, ' ').strip() for sentence in paragraph] for paragraph in sentences
            ]
        self.words = [[sentence.split() for sentence in paragraph] for paragraph in sentences]
        self.flat_words = [word for paragraph in self.text for sentence in paragraph for word in sentence]

    def words_dict(self):
        return {word: position for position, word in enumerate(self.flat_words)}

    def vocabulary_length(self):
        return len(self.flat_words)

    def batch(self, role):
        if role == 'input':
            from_pos, to_pos = 0, -1
        elif role == 'target':
            from_pos, to_pos = -1, len(self.text)
        batch = []
        for paragraph in self.words:
            for sentence in paragraph:
                batch.append(sentence[from_pos:to_pos])
        return batch


class NNLM(nn.Module):
    def __init__(self, text, steps=2, hidden_size=2, embedding_size=2):
        super(NNLM, self).__init__()
        voc = text.vocabulary_length()
        self.C = nn.Embedding(voc, embedding_size)
        self.H = nn.Linear(steps * embedding_size, hidden_size, bias=False)
        self.d = nn.Parameter(torch.ones(hidden_size))
        self.U = nn.Linear(hidden_size, voc, bias=False)
        self.W = nn.Linear(steps * embedding_size, voc, bias=False)
        self.b = nn.Parameter(torch.ones(voc))


text = Text('./books/small_example.txt', not_word_characters='.,;:?!«»…’', case_sensitive=False, end_sentences='.;,')
# text = Text('./books/01-fortune-rougon.txt', not_word_characters='.,;:?!«»…’', case_sensitive=False, end_sentences='.;,')

input_batch = text.batch('input')
target_batch = text.batch('target')

# model = NNLM(text)
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#


# input_batch, target_batch = make_batch()
# input_batch = torch.LongTensor(input_batch)
# target_batch = torch.LongTensor(target_batch)