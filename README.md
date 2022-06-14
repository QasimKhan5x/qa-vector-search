# qa-vector-search

question answering with vector similarity search

## Setup

    git clone https://github.com/deepset-ai/haystack.git
    cd haystack
    pip install -e .[only-faiss,preprocessing] ## or 'only-faiss-gpu' for the GPU-enabled dependencies

## Usage

1. Create data via `python create_documents.py` (If you don't have a gpu, please set all occurences of `use_gpu`to `False` in both files)
2. Provide a `str` query to `get_answers` in `main.py`

        from main import get_answers

        query = "Who created the Dothraki vocabulary?"
        result = get_answers(query)
        """
        [
            {   'answer': 'David J. Peterson',
                'context': 'orld. The language was developed for the TV series by the '
                        'linguist David J. Peterson, working off the Dothraki words '
                        "and phrases in Martin's novels.\n"
                        ','},
            {   'answer': 'David J. Peterson',
                'context': 'orld. The language was developed for the TV series by the '
                        'linguist David J. Peterson, working off the Dothraki words '
                        "and phrases in Martin's novels. ,"},
            {   'answer': 'David J. Peterson',
                'context': "age for ''Game of Thrones''\n"
                        'The Dothraki vocabulary was created by David J. Peterson '
                        'well in advance of the adaptation. HBO hired the Language '
                        'Creatio'},
        ]
        """
