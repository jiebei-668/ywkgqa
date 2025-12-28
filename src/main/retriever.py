from retriv import SparseRetriever, HybridRetriever, DenseRetriever

class Retriever:
    def __init__(self, retrievel_type: str, retriever_version: str):
        if retrievel_type == "sparse":
            self.retriever = SparseRetriever.load(retriever_version)
        elif retrievel_type == "dense":
            self.retriever = DenseRetriever.load(retriever_version)
        elif retrievel_type == "hybrid":
            self.retriever = HybridRetriever.load(retriever_version)
        else:
            raise ValueError("Invalid retriever type")

    def retrieve(self, query: str, top_k: int = 5):
        return self.retriever.search(query=query, return_docs=True, cutoff=top_k)




# 建立密集索引
# retriv_dir是索引保存的路径，匹配的模型默认为rainjay/sbert_nlp_corom_sentence-embedding_chinese-base-ecom
# file_path是要建立索引文件的路径，这个文件至少要有id和text两列
# 返回索引保存的路径和模型
def crtDenseRetriever(retriv_dir: str="/home/jiebei/ywkgqa/.retriv/dr_corom", file_path: str="/home/jiebei/ywkgqa/kg_data/tmpent.csv"):
    # create
    model = "rainjay/sbert_nlp_corom_sentence-embedding_chinese-base-ecom"
    dr = DenseRetriever(
        index_name=retriv_dir,
        model=model,
        normalize=True,
        max_length=128,
        use_ann=True,
    )

    # index create
    dr.index_file(
        path=file_path,  # File kind is automatically inferred
        embeddings_path=None,  # Default value
        use_gpu=True,  # Default value
        batch_size=512,  # Default value
        show_progress=True,  # Default value
        callback=lambda doc: {  # Callback defaults to None.
            "id": doc["id"],
            # "text": doc["title"] + ". " + doc["text"],
            "text": doc["text"],

        },
    )
    return retriv_dir, model


if __name__ == "__main__":
    retriv_path, model = crtDenseRetriever(retriv_dir="/home/jiebei/ywkgqa/.retriv/dr_corom_emb", file_path="/home/jiebei/ywkgqa/kg_data/tmpent.csv")

    dr = DenseRetriever().load(retriv_path)
    results = dr.search(
        query="四合一手术",  # What to search for
        return_docs=True,  # Default value, return the text of the documents
        cutoff=5,  # Default value, number of results to return
    )
    
    
    # 详细地遍历结果
    for i, result in enumerate(results):
        print(f"Result {i + 1}:")
        print(f"  ID: {result['id']}")
        print(f"  Text: {result['text']}")
        print(f"  Score: {result['score']}")
        print("-" * 50)


