import logging
import re
import string
import unicodedata
import warnings

from tqdm import tqdm
from collections import defaultdict

warnings.simplefilter(action='ignore')

all_letters = string.ascii_lowercase + "1234567890 "

expansions_dict = {' rd,': ' road,',
                   ' rd ': ' road ',
                   ' apts,': ' apartment,',
                   ' apts ': ' apartment ',
                   ' apt,': ' apartment,',
                   ' apt ': ' apartment ',
                   ' appts,': ' apartment,',
                   ' appts ': ' apartment ',
                   'apartments': 'apartment',
                   ' ngr,': ' nagar,',
                   ' ngr ': ' nagar ',
                   ',opp ': ',opposite ',
                   ' opp ': ' opposite ',
                   ',nr ': ',near ',
                   ' nr ': ' near ',
                   ' extn,': ' extension,',
                   ' extn ': ' extension ',
                   ' & ': ' and ',
                   '&': ' and ',
                   ' sec ': ' sector '}


class AddressCleaner:
    def __init__(self, samples):
        self.samples = samples

    @staticmethod
    def unicodeToAscii(s):
        return ' '.join(
            c for c in unicodedata.normalize('NFD', s.lower().strip())
            if unicodedata.category(c) != 'Mn'
            and c in all_letters
        )

    @staticmethod
    def removeSpecialChars(x, replaceWith=' '):
        delimiters = "; |, |\*|\n|-|\s"
        removeSpecialChars = '[^A-Za-z0-9 ]+'
        return re.sub(' +', ' ',
                      re.sub(removeSpecialChars, replaceWith,
                             ' '.join(map(str.strip, re.split(delimiters, x.lower()))))).strip()

    @staticmethod
    def removeDoubleSpaces(x):
        return re.sub('\s+', ' ', x).strip()

    @staticmethod
    def hashAllNumsExpectPincodes(s):
        pincodeId = re.compile(r'[1-9]{1}[0-9]{5}')
        replaceWithHash = re.compile(r'\d')
        pincodeSubsequences = re.findall(pincodeId, s)
        return ' '.join([replaceWithHash.sub(' ', i) if i not in pincodeSubsequences else i for i in s.split()])

    @staticmethod
    def camelCaseSplit(s):
        return ' '.join(re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', s)).split())

    @staticmethod
    def expand_expansions(s):
        def replace(match):
            return expansions_dict[match.group(0)]

        expansions_re = re.compile('(%s)' % '|'.join(expansions_dict.keys()))
        return expansions_re.sub(replace, s)

    def clean(self, tqdm_disable=False):
        cleanedAddresses = []
        for x in tqdm(self.samples, disable=tqdm_disable):
            x = self.removeSpecialChars(x)
            x = self.removeDoubleSpaces(x)
            x = self.expand_expansions(x)
            cleanedAddresses.append(x.strip())
        return cleanedAddresses


def createOutputContainer(tokens, tags):
    tagToKey = {'STATE': 'STATE',
                 'LOC': 'LOCALITY',
                 'CITY': 'CITY',
                 'PIN': 'PINCODE',
                 'PREM': 'PREMISES'}

    tempComponent = ''
    currentKey = ''
    out = defaultdict(list)
    for key, val in zip(tags, tokens):

        if key[0] == 'B':
            if currentKey != '' and tempComponent != '':
                out[tagToKey[currentKey]].append(tempComponent)
            tempComponent = val
            currentKey = key.split("-")[1]

        elif key[0] == "I":
            if currentKey == '':
                print(
                    f'Warning: internal component detected without begin component - currentElement:{(val, key)}')
                currentKey = key.split("-")[1]
            if key.split("-")[1] != currentKey:
                print(
                    f'Warning: internal token component mismatch - currentKey:{currentKey}, currentElement:{(val, key)}')
            tempComponent += ' ' + val

        elif key[0] == 'O':
            if currentKey == '' and tempComponent == '':
                continue
            out[tagToKey[currentKey]].append(tempComponent)
            tempComponent = ''
            currentKey = ''

    if currentKey != '' and tempComponent != '':
        out[tagToKey[currentKey]].append(tempComponent)

    return dict(out)


if __name__ == '__main__':
    pass
