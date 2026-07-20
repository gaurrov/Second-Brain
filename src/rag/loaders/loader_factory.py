"""
Loader factory — maps a FileType to its DocumentLoader implementation.
Adding support for a new file type means writing a new loader class and
registering it in `_LOADERS` below; nothing else in the ingestion
pipeline needs to change.
"""
from src.core.constants import FileType
from src.rag.loaders.base_loader import DocumentLoader
from src.rag.loaders.docx_loader import DOCXLoader
from src.rag.loaders.pdf_loader import PDFLoader
from src.rag.loaders.txt_loader import TXTLoader

_LOADERS: dict[FileType, DocumentLoader] = {
    FileType.PDF: PDFLoader(),
    FileType.DOCX: DOCXLoader(),
    FileType.TXT: TXTLoader(),
}


def get_loader(file_type: FileType) -> DocumentLoader:
    loader = _LOADERS.get(file_type)
    if loader is None:
        raise ValueError(f"No loader registered for file type: {file_type}")
    return loader
