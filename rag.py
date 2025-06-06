import os
import pandas as pd

from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

standard_prompt_template = """You are a construction contract review Al assistant.
You will be provided with checkpoint, specific condition delimited by ```.
Your task is to perform the following actions:
1 - check whether the specific condition contradicts with or entails with the checkpoint.
2 - if there are not enough information to determine whether the specificcondition contradicts with or entails with the checkpoint, simply output notfound.
3 - give explanation for your response.
Use the following format:
Condition situation: <contradict or entail or not found>
Explanation: <Explanation>

checkpoint: ```{checkpoint}```
specific condition: ```{specific_condition}```
"""


def load_data(file_path, source_column=None):
    loader = CSVLoader(file_path=file_path, source_column=source_column)
    data = loader.load()
    return data


def format_docs_for_rag(docs):
    return "\n\n".join(doc.metadata["source"] for doc in docs)


def main():
    checklist_data = load_data("./data/checklist.csv", "Risk_feature")
    contract_data = load_data("./data/contract.csv", "clause")
    # print(data)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    vectorstore = FAISS.from_documents(contract_data, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    print("Retriever created. Will retrieve top 2 documents.")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    llm = ChatTongyi(name="qwen-plus", api_key=api_key)

    prompt = PromptTemplate.from_template(standard_prompt_template)

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
    for row in checklist_df.itertuples():
        print("---")
        # specific_condition = f"{row.Risk_feature}: {row.Identified_items}"
        review_cases = clause_review_pair_df[
            clause_review_pair_df["checkpoint"] == row.Risk_feature
        ]
        specific_condition = ""
        # for i, review_case in enumerate(review_cases["clause"].to_list()):
        for i in range(len(review_cases)):
            # specific_condition += "Clause " + f"{i + 1}: " + review_case + "\n"
            specific_condition = review_cases[i]["clause"]
            print(specific_condition)

            # results = rag_chain.invoke(specific_condition)
            results = retriever.invoke(specific_condition)
            for result in results:
                print(result.metadata["row"])
                print(result.page_content)
                clause_corresponding_review[result.metadata["row"]].append(
                    review_cases[i]
                )
        print("---")
        break

    gt_df = pd.read_csv("./data/GT_contract_risk.csv")


if __name__ == "__main__":
    main()
