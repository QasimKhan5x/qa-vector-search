from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline

doc_store = FAISSDocumentStore.load("faiss_index.faiss")
assert doc_store.faiss_index_factory_str == "Flat"

retriever = DensePassageRetriever(
    document_store=doc_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    max_seq_len_query=64,
    max_seq_len_passage=256,
    batch_size=16,
    use_gpu=True,
    embed_title=True,
    use_fast_tokenizers=True,
)
reader = FARMReader(
    model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

pipe = ExtractiveQAPipeline(reader, retriever)


def get_answers(question):
    """Get a list of answers to a question

    Args:
        question (str): "Who created the Dothraki vocabulary?"

    Returns:
        List[Dict{str: str}]: [
            {
                'answer': 'David J. Peterson',
                'context': 'orld. The language was developed for the TV series by the '
                    'linguist David J. Peterson, working off the Dothraki words '
                    "and phrases in Martin's novels.\n"
            }
         ], length: 3
    """
    prediction = pipe.run(
        query=question,
        params={"Retriever": {"top_k": 10},
                "Reader": {"top_k": 3}
                }
    )
    answers = [{
        'answer': answer.answer,
        'context': answer.context
    } for answer in prediction['answers']]
    return answers
