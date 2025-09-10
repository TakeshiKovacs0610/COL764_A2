# bash tokenize_corpus.sh Data/Doc Data/stopwords.txt temp --time

# bash build_index.sh Data/Doc temp/vocab.txt temp/out_index --time

# bash phrase_search.sh temp/out_index Data/CORD19/queries.json temp/output Data/stopwords.txt --time

# bash vsm.sh temp/out_index Data/CORD19/queries.json temp/results Data/stopwords.txt 100 --time

# bash bm25_retrieval.sh temp/out_index Data/CORD19/queries.json temp/output Data/stopwords.txt 100 --time