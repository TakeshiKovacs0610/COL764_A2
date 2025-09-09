# Files to update : 

1. bm25_retrieval.py
    1. Implement the Okapi BM25 model. The model requires hyper-parameters.
2. bm25_retrieval.sh
    1. This file should call the python script with the correct arguments
3. ../Task1/build_index.py
    1. This file should be updated to include any necessary changes for bm25


# High level tasks performed by each file :
1. bm25_retrieval.py
    1. you have to write two functions with different signatures, one which takes
        a single query and returns a list of results, and another which takes a query file (in the format
        described earlier) and returns a TREC-eval file suitable to compute the accuracy numbers.
        Fine tune the hyperparameters to get the best results and hard code the finally selected hyperpa-
        rameters in the code.

    Below are the informations about the functions. 
    def bm25_query (query: str, index: object, k: int) -> list #given the query string, return the top-k documents with scores

    def bm25 (queryFile: str, index_dir: str, stopword_file: str, k: int, outFile: str) -> None #given the file containing queries, number of results k per query, write the results for each query in the output file

2. 


