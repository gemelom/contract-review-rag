import os
import pandas as pd

from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from evaluator import evaluate_predictions

standard_prompt_template = """You are a construction contract review Al assistant.
You will be provided with checkpoint, specific condition delimited by ```.
Your task is to perform the following actions:
1 - check whether the specific condition contradicts with or entails with the checkpoint.
2 - if there are not enough information to determine whether the specificcondition contradicts with or entails with the checkpoint, simply output notfound.
3 - give explanation for your response.
4 - give your final assessment.
Use the following format:
Condition situation: <contradict or entail or not found>
Explanation: <explanation>
Assessment: <If there is a risk, answer A; if there is no risk or not found, answer B. Please answer ONLY with A or B>

checkpoint: ```{checkpoint}```
specific condition: ```{specific_condition}```
"""

rag_prompt_template = """You are a construction contract review Al assistant.
You will be provided with checkpoint, specific condition, identified items, and relevant review delimited by ```.
Your task is to perform the following actions:
1 - summarize what have you learned from the exemplars first concisely, no need to repeat the new question in your answer.
2 - check whether the specific condition contradicts with or entails with the checkpoint.
3 - if there are not enough information to determine whether the specificcondition contradicts with or entails with the checkpoint, simply output notfound.
4 - give explanation for your response.
5 - give your final assessment.
Use the following format:
Summary: <Summary of the exemplars>
Condition situation: <contradict or entail or not found>
Explanation: <Explanation>
Assessment: <If there is a risk, answer A; if there is no risk or not found, answer B. Please answer ONLY with A or B>

checkpoint: ```{checkpoint}```
identified items: ```{identified_items}```
{review_cases}
specific condition: ```{specific_condition}```
"""


def load_data(file_path, source_column=None):
    loader = CSVLoader(file_path=file_path, source_column=source_column)
    data = loader.load()
    return data


def format_docs_for_rag(docs):
    return "\n\n".join(doc.metadata["source"] for doc in docs)


def standard_pipeline():
    review_pair_data = load_data("./data/clause_review_pair.csv", "checkpoint")
    # print(data)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    vectorstore = FAISS.from_documents(review_pair_data, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    print("Retriever created. Will retrieve top 2 documents.")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    llm = ChatTongyi(name="qwen-plus", api_key=api_key)

    prompt = PromptTemplate.from_template(standard_prompt_template)

    checklist_df = pd.read_csv("./data/checklist.csv")
    contract_df = pd.read_csv("./data/contract.csv")
    clause_review_pair_df = pd.read_csv("./data/clause_review_pair.csv")

    clause_corresponding_review = [[] for _ in range(len(contract_df))]
    predictions = []
    for row in contract_df.itertuples():
        print("---")
        review_case = retriever.invoke(row.clause)[0]
        cur_checkpoint = review_case.metadata["source"]
        all_review_cases = clause_review_pair_df[
            clause_review_pair_df["checkpoint"] == cur_checkpoint
        ]
        identified_items = checklist_df[
            checklist_df["Risk_feature"] == cur_checkpoint
        ].Identified_items.array[0]
        # print(identified_items)

        review_cases_str = ""
        for review_case in all_review_cases.itertuples():
            review_cases_str += f"case clause: ```{review_case.clause}```\ncase review: ```{review_case.review}```\n"
        formated_prompt = prompt.format(
            checkpoint=row.checkpoint,
            specific_condition=row.clause,
        )
        print(formated_prompt)
        res = llm.invoke(formated_prompt)
        print(res.content)
        prediction = next(ch for ch in reversed(res.content) if ch in ["A", "B"])
        if prediction not in ["A", "B"]:
            print("Warning: LLM returned an unexpected result")
            prediction = "B"
        predictions.append(prediction)
        # specific_condition = f"{row.Risk_feature}: {row.Identified_items}"
        # review_cases = clause_review_pair_df[
        #     clause_review_pair_df["checkpoint"] == row.Risk_feature
        # ]
        # specific_condition = ""
        # # for i, review_case in enumerate(review_cases["clause"].to_list()):
        # for i in range(len(review_cases)):
        #     # specific_condition += "Clause " + f"{i + 1}: " + review_case + "\n"
        #     specific_condition = review_cases[i]["clause"]
        #     print(specific_condition)
        #
        #     # results = rag_chain.invoke(specific_condition)
        #     results = retriever.invoke(specific_condition)
        #     for result in results:
        #         print(result.metadata["row"])
        #         print(result.page_content)
        #         clause_corresponding_review[result.metadata["row"]].append(
        #             review_cases[i]
        #         )
        print("---")
        # break

    gt_df = pd.read_csv("./data/GT_contract_risk.csv")
    true_labels = gt_df["gt"].tolist()
    evaluate_predictions(true_labels, predictions)


def rag_pipeline():
    checklist_data = load_data("./data/checklist.csv", "Risk_feature")
    contract_data = load_data("./data/contract.csv", "clause")
    review_pair_data = load_data("./data/clause_review_pair.csv", "checkpoint")
    # print(data)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    vectorstore = FAISS.from_documents(review_pair_data, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    print("Retriever created. Will retrieve top 2 documents.")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    llm = ChatTongyi(name="qwen-plus", api_key=api_key)

    prompt = PromptTemplate.from_template(rag_prompt_template)

    def print_prompt_and_pass(input_dict):
        rendered_prompt = prompt.invoke(input_dict)
        print("\n--- Final Prompt ---")
        print(rendered_prompt.text)
        print("--------------------\n")
        return input_dict

    rag_chain = (
        {
            "checkpoint": retriever | format_docs_for_rag,
            # "expert_cases": case_clause_retriever | format_docs_for_rag,
            "specific_condition": RunnablePassthrough(),
        }
        | RunnableLambda(print_prompt_and_pass)
        | prompt
        | llm
        | StrOutputParser()
    )

    checklist_df = pd.read_csv("./data/checklist.csv")
    contract_df = pd.read_csv("./data/contract.csv")
    clause_review_pair_df = pd.read_csv("./data/clause_review_pair.csv")

    clause_corresponding_review = [[] for _ in range(len(contract_df))]
    predictions = []
    for row in contract_df.itertuples():
        print("---")
        review_case = retriever.invoke(row.clause)[0]
        cur_checkpoint = review_case.metadata["source"]
        all_review_cases = clause_review_pair_df[
            clause_review_pair_df["checkpoint"] == cur_checkpoint
        ]
        identified_items = checklist_df[
            checklist_df["Risk_feature"] == cur_checkpoint
        ].Identified_items.array[0]
        # print(identified_items)

        review_cases_str = ""
        for review_case in all_review_cases.itertuples():
            review_cases_str += f"case clause: ```{review_case.clause}```\ncase review: ```{review_case.review}```\n"
        formated_prompt = prompt.format(
            checkpoint=cur_checkpoint,
            specific_condition=row.checkpoint + ": " + row.clause,
            identified_items=identified_items,
            review_cases=review_cases_str,
        )
        print(formated_prompt)
        res = llm.invoke(formated_prompt)
        print(res.content)
        prediction = next(ch for ch in reversed(res.content) if ch in ["A", "B"])
        if prediction not in ["A", "B"]:
            print("Warning: LLM returned an unexpected result")
            prediction = "B"
        predictions.append(prediction)
        # specific_condition = f"{row.Risk_feature}: {row.Identified_items}"
        # review_cases = clause_review_pair_df[
        #     clause_review_pair_df["checkpoint"] == row.Risk_feature
        # ]
        # specific_condition = ""
        # # for i, review_case in enumerate(review_cases["clause"].to_list()):
        # for i in range(len(review_cases)):
        #     # specific_condition += "Clause " + f"{i + 1}: " + review_case + "\n"
        #     specific_condition = review_cases[i]["clause"]
        #     print(specific_condition)
        #
        #     # results = rag_chain.invoke(specific_condition)
        #     results = retriever.invoke(specific_condition)
        #     for result in results:
        #         print(result.metadata["row"])
        #         print(result.page_content)
        #         clause_corresponding_review[result.metadata["row"]].append(
        #             review_cases[i]
        #         )
        print("---")
        # break

    gt_df = pd.read_csv("./data/GT_contract_risk.csv")
    true_labels = gt_df["gt"].tolist()
    evaluate_predictions(true_labels, predictions)


if __name__ == "__main__":
    standard_pipeline()
    # rag_pipeline()
