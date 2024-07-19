"""PDF Knowledge."""
from typing import Any, Dict, List, Optional, Union

from dbgpt.core import Document
from dbgpt.rag.knowledge.base import (
    ChunkStrategy,
    DocumentType,
    Knowledge,
    KnowledgeType,
)
import json

from unstructured.documents.elements import ElementType
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import clean
import opencc
import re

from dbgpt.rag.knowledge.unstructrued_element import CleanedMetadata, CleanedElement


class PDFKnowledge(Knowledge):
    """PDF Knowledge."""

    def __init__(
            self,
            file_path: Optional[str] = None,
            knowledge_type: KnowledgeType = KnowledgeType.DOCUMENT,
            loader: Optional[Any] = None,
            language: Optional[str] = "zh",
            metadata: Optional[Dict[str, Union[str, List[str]]]] = None,
            **kwargs: Any,
    ) -> None:
        """Create PDF Knowledge with Knowledge arguments.

        Args:
            file_path(str,  optional): file path
            knowledge_type(KnowledgeType, optional): knowledge type
            loader(Any, optional): loader
            language(str, optional): language
        """
        super().__init__(
            path=file_path,
            knowledge_type=knowledge_type,
            data_loader=loader,
            metadata=metadata,
            **kwargs,
        )
        self._language = language

    def _load(self) -> List[Document]:
        """Load pdf document from loader."""
        if self._loader:
            documents = self._loader.load()
        else:
            pages = []
            page = []
            documents = []
            pattern = re.compile(r'[\u4e00-\u9fa5]')

            elements = partition_pdf(self._path, strategy='hi_res', hi_res_model_name='yolox',
                                     infer_table_structure=True,
                                     form_extraction_skip_tables=[], languages=['eng', 'chi_tra', 'chi_sim'],
                                     include_page_breaks=False)
            converter = opencc.OpenCC('t2s')

            for element in elements:
                ele_json = json.loads(json.dumps(element.to_dict(), indent=2))
                if ele_json.get('type') == ElementType.PAGE_BREAK and page:
                    pages.append(page)
                    page = []
                else:
                    element_id = ele_json.get('element_id')
                    element_type = ele_json.get('type')

                    if element_type in [ElementType.TABLE] and ele_json.get('metadata').get('text_as_html'):
                        element_text = ele_json.get('metadata').get('text_as_html')
                    else:
                        element_text = ele_json.get('text')

                    # 文本清洗(后续抽象成一个方法)
                    # 只保留指定的元素
                    if element_type not in [ElementType.TITLE, ElementType.TEXT, ElementType.UNCATEGORIZED_TEXT, ElementType.NARRATIVE_TEXT, ElementType.PARAGRAPH, ElementType.TABLE]:
                        continue
                    element_text = clean(element_text)
                    # 繁体中文转简体中文，后面需要对语言类型(用户输入、chunk的语言进行统一)
                    element_text = converter.convert(element_text)
                    # 去除文本中的空格
                    element_text = element_text.replace(' ', '')
                    # 去除纯英文文本
                    if not pattern.search(element_text):
                        continue

                    page_number = ele_json.get('metadata').get('page_number')
                    file_name = ele_json.get('metadata').get('filename')

                    metadata = CleanedMetadata(file_name, page_number)
                    cleaned_element = CleanedElement(element_id, element_type, element_text, metadata)

                    page.append(cleaned_element.to_dict())

            if page:
                pages.append(page)

            for index, page in enumerate(pages):
                page_str = json.dumps(page, indent=2)
                metadata = {"source": self._path}
                document = Document(content=page_str, metadata=metadata)
                documents.append(document)
            return documents
        return [Document.langchain2doc(lc_document) for lc_document in documents]

    @classmethod
    def support_chunk_strategy(cls) -> List[ChunkStrategy]:
        """Return support chunk strategy."""
        return [
            ChunkStrategy.CHUNK_BY_SIZE,
            ChunkStrategy.CHUNK_BY_PAGE,
            ChunkStrategy.CHUNK_BY_SEPARATOR,
            ChunkStrategy.CHUNK_BY_UNSTRUCTURED
        ]

    @classmethod
    def default_chunk_strategy(cls) -> ChunkStrategy:
        """Return default chunk strategy."""
        return ChunkStrategy.CHUNK_BY_SIZE

    @classmethod
    def type(cls) -> KnowledgeType:
        """Return knowledge type."""
        return KnowledgeType.DOCUMENT

    @classmethod
    def document_type(cls) -> DocumentType:
        """Document type of PDF."""
        return DocumentType.PDF
