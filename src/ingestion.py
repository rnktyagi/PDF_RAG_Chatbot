from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
from langchain_core.documents import Document
from docling.datamodel.base_models import DocumentStream
from io import BytesIO

converter = DocumentConverter()
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
chunker = HybridChunker(tokenizer=tokenizer,max_tokens=800,merge_peers=True)

def ingestion_and_chunking(file_bytes: bytes, filename: str) :

    source = DocumentStream(name=filename, stream=BytesIO(file_bytes))
    result = converter.convert(source)
    chunks = list(chunker.chunk(result.document))

    for chunk in chunks :
        chunk.text = chunker.contextualize(chunk)

    docs = []

    for chunk in chunks:
        pages = sorted({
            prov.page_no
            for item in chunk.meta.doc_items
            for prov in item.prov})

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