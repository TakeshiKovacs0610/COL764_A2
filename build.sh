# bash tokenize_corpus.sh Data/Doc temp --time

# bash build_index.sh Data/Doc temp/vocab.txt temp/out_index --time

# bash phrase_search.sh temp/out_index Data/CORD19/queries.json temp/results --time

# bash vsm.sh temp/out_index Data/CORD19/queries.json temp/results 200 --time

# bash bm25_retrieval.sh temp/out_index Data/CORD19/queries.json temp/results 200 --time

# bash feedback.sh temp/out_index Data/CORD19/queries.json temp/results 200 --time