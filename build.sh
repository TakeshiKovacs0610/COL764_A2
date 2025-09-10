#   bash Task0/tokenize_corpus.sh Data/Doc Data/stopwords.txt temp --time 


# bash Task1/build_index.sh Data/Doc temp/vocab.txt temp/out_index --time

# bash Task2/phrase_search.sh temp/out_index Data/CORD19/queries.json temp/output Data/stopwords.txt --time

# bash Task3/vsm.sh temp/out_index Data/CORD19/queries.json temp/results Data/stopwords.txt 100 --time

# bash Task4/bm25_retrieval.sh temp/out_index Data/CORD19/queries.json temp/output Data/stopwords.txt 100 --time