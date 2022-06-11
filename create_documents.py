from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever, PreProcessor
from haystack.utils import (clean_wiki_text, convert_files_to_dicts,
                            fetch_archive_from_http)

# Let's first get some files that we want to use
doc_dir = "data/tutorial6"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt6.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)


document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")
# Convert files to dicts
docs = convert_files_to_dicts(
    dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

# Now, let's write the dicts containing documents to our DB.
document_store.write_documents(docs)

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=False,
    split_by="word",
    split_length=150,
    split_respect_sentence_boundary=True
)

dicts = preprocessor.process(docs)
document_store.write_documents(dicts)


retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    max_seq_len_query=64,
    max_seq_len_passage=256,
    batch_size=16,
    use_gpu=True,
    embed_title=True,
    use_fast_tokenizers=True,
)

document_store.update_embeddings(retriever)
document_store.save("faiss_document_store.db")
