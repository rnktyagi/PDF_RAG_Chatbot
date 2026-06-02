from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
from langchain_core.documents import Document
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer


def ingestion_and_chunking(file_path : str) :

    converter = DocumentConverter()
    result = converter.convert(file_path)

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    chunker = HybridChunker(merge_peers=True , 
                            chunk_size=800 , 
                            overlap=200,
                            tokenizer=tokenizer )
    
    chunks = list(chunker.chunk(result.document))

    for chunk in chunks :
        chunk.text = chunker.contextualize(chunk)

    
    docs = []

    for chunk in chunks:
        pages = sorted({
            prov.page_no
            for item in chunk.meta.doc_items
            for prov in item.prov
        })

        docs.append(
            Document(
                page_content=chunk.text,
                metadata={
                    "source": chunk.meta.origin.filename,
                    "pages": pages,
                    "section": chunk.meta.headings[0] if chunk.meta.headings else None,
                }
            )
        )

    return docs