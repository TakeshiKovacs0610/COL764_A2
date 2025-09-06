Okay, the current task is to implement the pre-search algorithm in case of an in-boolean retriever. And by that, what the code is supposed to do is, okay, what is the code supposed to do? It is supposed to load up the index.json file from build index because that will be used.

And now it will get queries, right? So a query will be a string, a simple string that say COVID-19 infections or say a string like cat and dog. Now, once you have this string, what you first of all need to do is correctly parse it into a tokenizable form in the same way that the tokenization was performed for the corpus and then the index was built upon.

So that, the way that was done was with the help of the spaCy tokenizer. Now, a blank model of that tokenizer was used without limitization or stemming. So, I think spaCy.blank English is the command that is used. That command was used.

So, for the corpus, you had the relevant text and it was just passed into the tokenizer and then the output was given. So, in a similar fashion, what we need here is first of all, you will get the query from the query.json file or from the user input.

Then you will run the spaCy tokenizer. So, say the tokenizer is cat and dog. Input text is cat and dog. So, it will get broken down into three tokens.

Cat and dog. The cat is the first token. The second token is add. And third token is dog. Now, here we are needed to run a phrase search.

So, we need to find all the documents which have exactly this phrase, cat and dog, inside it. So, what you learn from this phrase is that it occurs with cat. Then and comes one character after cat and dog comes one character after and. So, to look this up, what you will do is for each of these tokens, you will find, okay, which document cat occurs in, which document and occurs in, which document dog occurs in.

Now, you will take an intersection of these documents. Now, say you end up with a set of five documents, which all have these three terms, cat and dog. Now, for each of these documents, you will collect the postings list from each of the terms. So, now you know at all the index position where cat occurs, all the index position where and occurs, and all the index position where dog occurs.

And these indexes are stored in increasing order or in ascending order in the list. So, a simple forward pass through each of these in parallel should allow you to find if the document, this document exactly has the phrase cat and dog inside it. So, you need to think over the implementation for this.

How exactly this is to be implemented in code is something you need to think about. This is different from a simple Boolean retrieval where you had connectives like AND, OR, NOR, which were to be added in the query between terms. Here, it is simply a phrase search, nothing else.

Now, this is to be done in the most efficient data structure way possible so that it takes as less amount of time as possible. So, for additional context, you will have the tokenize corpus Python file which tells you how the tokenization happened for the corpus and then to build the vocabulary.

And then the build index.py file which took in this vocabulary along with the corpus and performed the tokenization in the similar form and generated the output that is there. For the inverted index, it generated with the postings list and everything. So, looking at those two code files will give you an understanding of how the data is stored inside index.json which is to be loaded here.

So, how that loading needs to be done is something you'll know. 


You also have for reference the assignment PDF attached in the directory. Now look at the directory structure. You will find a task 0 folder, a task 1 folder and a task 2 folder. The question I am asking you to solve is to be done in the task 2 folder.

But relevant documents would be present in task 1 and task 0 folders for you to look up. and the assignment document will be present in the main directory itself. 

A very important information to keep in mind is that there is no use of stopwords file. This was an error in the code earlier that has not yet been fixed and that is also something that needs to be fixed. And the error is that in the tokenization process, the stopwords.txt file is being used to remove stopwords.

This was done in a previous version where spaCy was not being used. Now that we are using spaCy, we do not need to do that. Simply whatever is being tokenized by spaCy needs to be converted into the vocab.txt file. So this change needs to be done in task 0 tokenized corpus.py file.

This is another thing that you need to do along with making the similar change in build index.py file in task 1 folder. And then finally use this updated information to make, think over and create the phrase search.py file. While you are doing this, keep looking into the assignment PDF that is there in the directory.

It is your ground truth. It is your ground truth. It is another, it needs to specify, it needs to follow the specifications mentioned in the assignment. It's not perfectly a ground truth because this, again, the stopwords.txt file that's mentioned there is not being used here.

But for everything else, it can be used for reference. When you generate the final changes, clearly explain each change that you have made, why you've made it. And if you have taken and made any assumptions, again, explain those assumptions carefully and perfectly to me. The code should be readable.

It should follow a simple coding style as if written by an undergraduate student, undergraduate CS student with, yeah, that I'll leave the rest to you. 