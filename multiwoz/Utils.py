from nltk import RegexpTokenizer

def get_tokenize():
    return RegexpTokenizer(r'\w+|#\w+|<\w+>|%\w+|[^\w\s]+').tokenize