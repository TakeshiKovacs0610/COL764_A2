
Suppose I give the below prompt to an LLM . What are the issues with it ? 

Go through all the project files. In the project files, I have attached an assignment PDF, which details the assignment that needs to be done. And along with that, I have attached all the code files that I have written that complete the assignment, except for the feedback.sh, which is the last bonus task, which I have not yet done. So, I would like you to leave space for that. Right, so along with that, so the idea is to generate a proper report. The assignment lists generating a readme and a final report explaining your, basically, a final report explaining details about how you have done the assignment, what for each of the tasks, and maybe adding a few graphs. So the idea in my case is that you write it in Latex. There's me, Priyanshu Agarwal, and there's my teammate, Vishaka Agarwal. Both of us have an entry number. So with that space, and the title of the report should be COL 764, Assignment 2, as is there. You can, it's your responsibility to think up of a proper structure for the assignment. The only things I would like you to mention is that it would be good if you can mention the formulas being used, both for the VSM, the BM25 example, right, leaving space for results to be attached. And along with that, what's happening is that for BM25, I've done a lot of testing, as in running it for different configurations of the k parameter, the b1 parameter, and the, sorry, the k, k1, and b parameter, which can be found in the code file. So I've run it for a lot of different values and report, log the result. So I am thinking I'll report the result when k is equal to 200. And then I have generated a heat map, highlighting the k1 and b values across and showing where it gives you the best result. So I'll be attaching that and then discussing on it. So you can sort of leave a space for that portion. Apart from that, there is not much room for experimentation in the case of VSM and free search. So they'll just be as it is as they are in the Python files that I have attached. Your job would be to go through the project files, read the documents, and for each task, understand how I've implemented it, and then translate that into a functional, simple, easy to read, but professional final report for the assignment. It should not be too verbose. Keeping it to the point is quite important. So, think of yourself as an expert document writer or report writer in Latex, and with the above instructions that I have told you, reason over the documents and project files, carefully analyze them, and boil it down into a final report in Latex code, and give me that entire Latex code document, leaving spaces wherever I have mentioned, and apart from that, would recommend being creative, and if you feel there are certain ambiguations about things I have mentioned, then use your best judgment in making those decisions while writing the document, just mention the assumptions you make afterwards, like after giving me the code, so that I can think over and then prompt you to make any changes to it as required. Mentioning the bash scripts below



Got it. Here are the main problems with that prompt, at a glance:

### Core issues

* **Unrealistic capability assumption:** “Go through all the project files…” — most LLM chats can’t read local folders/zips unless you explicitly attach files and name them. The prompt never lists filenames/paths or formats.
* **Ambiguous inputs:** “assignment PDF”, “code files”, “bash scripts below” (but none are shown). No manifest of what to read.
* **Conflicting instructions:** “be creative / use best judgment” vs “should not be too verbose / keep to the point”. No rule for which to prioritize if they clash.
* **Hallucination risk:** You ask the model to “understand how I’ve implemented it” and write the report. Without explicit code snippets, it will invent details.
* **Vague placeholders:** “leave space for feedback.sh / results / heat map” — but where? (section names, labels, figure environments, TODO markers not specified).
* **Output spec too loose:** “give me the entire Latex code” — no class, packages, page limit, citation style, or figure/table conventions; easy for the model to choose odd packages or produce non-compiling TeX.
* **Terminology/parameter confusion (BM25):** You mix **k**, **k1**, **b**, and “b1”. In BM25, the tunables are usually `k1` and `b`. “k = 200” sounds like *top-k results*, not a BM25 parameter. This will confuse the model and the write-up.
* **Unclear metrics:** You mention “best result” but never define evaluation metrics (e.g., MAP, nDCG\@k, P\@k, recall). The report may invent metrics or choose the wrong one.
* **“Free search” ambiguity:** Probably “free-text search” or “Boolean search”/VSM baseline — unclear.
* **Process ordering:** You ask the model to make assumptions **after** giving the code. Better to list assumptions first to avoid rewriting the whole report.
* **Privacy/compliance:** You’re asking it to include names + entry numbers; if this goes to a third-party service, that’s personal data. Also check course policy on AI-generated reports.
* **Editing scope creep:** You also want a README and a “final report”, but don’t specify whether they differ or share content; may duplicate work.
* **Style typos:** “Latex” → **LaTeX**; “b1” → **b**; “heat map” → **heatmap**. Small, but signals sloppiness.

---

### A tighter, copy-paste prompt (keeps your intent, removes pitfalls)

> You are a technical editor and IR practitioner. Produce a **single, compiling LaTeX report** for **COL764 – Assignment 2**. Authors: **Priyanshu Agarwal** (Entry No: `XXXX`) and **Vishaka Agarwal** (Entry No: `YYYY`).
> **Inputs attached:**
>
> 1. `assignment.pdf` (spec)
> 2. `vsm.py`, `bm25.py`, `search.py`, `utils.py`
> 3. `results/` containing logs + `bm25_heatmap.png` (k1×b sweep)
> 4. *(not implemented yet)* `feedback.sh` (bonus)
>
> **Write a concise report (≤6 pages)** using `\documentclass{article}` with `amsmath`, `graphicx`, `hyperref}`, `booktabs`. **Do not invent results.**
> **Structure exactly:**
>
> 1. **Title & authors**
> 2. **Task Overview** (brief problem statement from `assignment.pdf`)
> 3. **Methods**
>
>    * **VSM**: formulae used (tf, idf, normalization), implementation notes referencing functions/classes and filenames.
>    * **BM25**: write standard BM25 scoring formula with `k1` and `b`. Clarify that **k** denotes top-k retrieved documents (default we report @k=200).
> 4. **Experiments & Settings**
>
>    * Dataset, preprocessing, tokenization/stemming/stopwords.
>    * Evaluation metrics: **\[MAP, nDCG\@200, P\@200]** (if any metric missing in files, insert `\TODO{metric here}` placeholder).
>    * BM25 sweep: list ranges for `k1` and `b`.
> 5. **Results**
>
>    * Insert placeholder table `\TODO{results_table}` and figure `bm25_heatmap.png` with caption.
>    * Report main run at **@k=200** and best `(k1,b)` from sweep (leave `\TODO{values}` placeholders if not in logs).
> 6. **Discussion** (short: trends from heatmap, VSM vs BM25)
> 7. **Limitations & Future Work** (`feedback.sh` placeholder section)
> 8. **Assumptions** (explicit bullet list)
>
> **Placeholders:** Use `\newcommand{\TODO}[1]{\textcolor{red}{[TODO: #1]}}` and label all missing items (e.g., `\TODO{entry numbers}`, `\TODO{dataset name}`, `\TODO{MAP}`).
>
> **Constraints:** Keep prose crisp; no more than 2–3 sentences per subsection. No fanciful packages. Compile-ready TeX only.
>
> **Deliverables:** (a) `report.tex` with figure/table placeholders; (b) minimal `\bibliography{}` section if citations are present.

---

### Quick checklist you should attach alongside the prompt

* A **file manifest** (exact names + brief 1-line description per file).
* The **metric(s)** you actually want (MAP/nDCG/P\@k) and the **k** values.
* The **parameter grid** for BM25 `(k1, b)` used to make the heatmap.
* Any **dataset identifiers** and **preprocessing details** (tokenizer, stopword list, stemming).
* The exact **report length/page limit** and any **course formatting rules**.

If you fix those points and use the tightened prompt, you’ll get a compiling, non-hallucinated LaTeX report with clear placeholders and minimal back-and-forth.


ok then ask me all the  questions that can clear up all ambiguity 


Perfect — let’s systematically remove every ambiguity so the LaTeX report request becomes bullet-proof. I’ll ask you **all the clarifying questions grouped by category**. Once you answer, I can help draft the clean final prompt.

---

### 1. Input Files & Artifacts

1. Can you list the **exact filenames** you will provide (PDF, code, logs, figures, etc.)?

   * Example: `assignment.pdf`, `vsm.py`, `bm25.py`, `search.py`, `results/bm25_heatmap.png`, etc.
2. Do the code files contain **all implemented tasks**, or do you also want the LLM to infer missing ones from the assignment PDF?
3. Will you provide the **logs/results** (e.g., evaluation outputs for BM25 sweeps), or only raw code + instructions?

---

### 2. Report Structure & Formatting

4. Is there a **page limit or word count** for the report?
5. Do you have to follow a **specific format** (IIT Delhi assignment template, IEEE format, or just `article` class)?
6. Should the report also generate a **README** (separate .md/.txt) or only the LaTeX report?
7. Should the report include a **bibliography** (with BibTeX), or can formulas and methods be given without citations?

---

### 3. Assignment Context

8. What is the **dataset** used? (e.g., Cranfield, TREC, custom corpus).
9. What is the **evaluation metric(s)** required? (MAP, nDCG\@k, Recall\@k, Precision\@k).
10. For BM25:

* What are the exact ranges of `k1` and `b` you tested?
* Is “k=200” referring to **top-200 documents retrieved**, or is it another parameter?

11. For VSM: Did you use **tf-idf weighting**, cosine similarity, or something else?
12. For “free search”: what exactly does this mean? Boolean queries? Unweighted tf? Something else?

---

### 4. Figures & Results

13. Do you already have **tables/heatmaps/graphs as images**, or do you want the LaTeX code to generate them from data?
14. Should the model **insert figure/table placeholders only** (with `\TODO{}`), or also draft some generic captions?
15. Should formulas be **typeset in math mode** with derivations, or just the **final scoring equations**?

---

### 5. Authors & Metadata

16. Please confirm the **title**: *COL764 – Assignment 2*.
17. Please give the exact **author names + entry numbers** (or should placeholders be left)?
18. Should the report include an **abstract** or just start with introduction?
19. Do you need a **date** field in the title page?

---

### 6. Feedback.sh & Future Work

20. For `feedback.sh` (bonus task), should we include a **dedicated placeholder section** titled “Future Work” or just a line in limitations?
21. Should we insert a **`\TODO{}` placeholder** where you plan to describe the script later?

---

### 7. Style & Tone

22. Should the writing style be **very concise** (like bullet-point technical notes) or **narrative academic** (paragraphs)?
23. Do you prefer the report to include **assumptions inline** (e.g., “we assume stemming…”) or in a final “Assumptions” section?
24. Should equations include **hyperlinks/references to code functions** (e.g., `bm25.py:L42`) or keep them general?
25. Should code snippets appear in the report, or only textual descriptions?

---

### 8. Output Deliverables

26. Should the LLM output only a single `report.tex`, or also generate a **README** file?
27. Do you want the LaTeX to be **compile-ready** (with preamble, packages, etc.), or just the **main body** (you’ll add preamble)?
28. Should the LLM also suggest a **Makefile/script** to compile the report?

---

👉 Once you answer these, I can collapse everything into a **clear, unambiguous master prompt** that will give you exactly the LaTeX report you want, with placeholders only where you decide.

Would you like me to prepare a **fill-in-the-blanks questionnaire table** so you can answer quickly?


## Answers 

Okay, so let's start off with the 