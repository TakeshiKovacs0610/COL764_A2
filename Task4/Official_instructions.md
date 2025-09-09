
## Implementation Details : 


Task 4: BM-25 Retrieval :  
Your submission should also consist of another program called bm25_retrieval.{py|cpp}  
. It should contain the bm25_query() and bm25() functions.

This program will be executed using a shell script (to be provided by you) named bm25_retrieval.sh.  
The shell script will be invoked via terminal as follows:

bm25_retrieval.sh <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> <PATH_OF_STOPWORDS_FILE> 

where:

<INDEX_DIR>: absolute path of the directory where index.json is saved  

<QUERY_FILE_PATH>: absolute path of the query file  
<OUTPUT_DIR>: absolute path of the directory where docids.txt is to be saved.  
<PATH_OF_STOPWORDS_FILE>: Same as in Task0  
<k>: Count of the k-most relevant documents to be retrieved for each query  

Output:  
(a) The program should generate one file bm25_docids.txt (format given in 3.2)  
(b) Based on the qrels file provided, report the precision, recall and F1 score.



Task 4: Probabilistic retrieval  

**Objective:** Implement the Okapi BM25 model. The model requires hyper-parameters. Study this model carefully and choose the parameters that you think are best.

**Function:** Note that you have to write two functions with different signatures, one which takes a single query and returns a list of results, and another which takes a query file (in the format described earlier) and returns a TREC-eval file suitable to compute the accuracy numbers.  
Fine tune the hyperparameters to get the best results and hard code the finally selected hyperparameters in the code.

def bm25_query (query: str, index: object, k: int) -> list #given the query string, return the top-k documents with scores



def bm25 (queryFile: str, index_dir: str, stopword_file: str, k: int, outFile: str) -> None #given the file containing queries, number of results k per query, write the results for each query in the output file

---





## Logic 








## Doubts / Grey Areas 


