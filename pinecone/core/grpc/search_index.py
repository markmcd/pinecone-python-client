import gzip
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import csv
import json

from pinecone.core.grpc.index_grpc import GRPCIndexBase
from pinecone.core.grpc.protos.search_service_pb2_grpc import SearchServiceStub
from pinecone.core.grpc.protos.search_service_pb2 import UpsertRequest, QueryRequest, UpsertResponse, QueryResponse, \
    FetchRequest, FetchResponse, DeleteRequest
from pinecone.core.grpc.protos.search_service_pb2 import TextVector as ProtoTextVector
import logging
from typing import Optional, List, Dict, Any, Union, Iterable, Iterator

from tqdm import tqdm
from uuid import uuid4
from pinecone.core.utils import dict_to_proto_struct

_logger = logging.getLogger(__name__)

__all__ = ['SearchIndex', 'Document']


@dataclass
class Document:
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None

    def to_proto(self) -> ProtoTextVector:
        metadata = dict_to_proto_struct(self.metadata) if self.metadata is not None else None
        return ProtoTextVector(id=self.id, text=self.text, metadata=metadata)


class SearchIndex(GRPCIndexBase):

    def __init__(self, index_name: str, embedding_model: str):
        super().__init__(index_name)
        self.embedding_model = embedding_model

    @property
    def stub_class(self):
        return SearchServiceStub

    def upsert(self,
               documents: Union[List[Document], List[str]],
               namespace: Optional[str] = None,
               batch_size: Optional[int] = None,
               show_progress: bool = True,
               timeout: Optional[int] = None) -> UpsertResponse:
        documents = [ProtoTextVector(id=str(uuid4()), text=v) if isinstance(v, str)
                     else v.to_proto()
                     for v in documents]
        if batch_size is None:
            request = UpsertRequest(vectors=documents, namespace=namespace, embedding_model=self.embedding_model)
            return self._wrap_grpc_call(self.stub.Upsert, request, timeout=timeout)

        pbar = tqdm(total=len(documents), disable=not show_progress, desc='Upserted vectors')
        total_upserted = 0
        for i in range(0, len(documents), batch_size):
            request = UpsertRequest(vectors=documents[i:i + batch_size], namespace=namespace)
            response = self._wrap_grpc_call(self.stub.Upsert, request, timeout=timeout)
            total_upserted += response.upserted_count
            pbar.update(batch_size)

        return UpsertResponse(upserted_count=total_upserted)

    def query(self,
              text: str,
              namespace: Optional[str] = None,
              top_k: Optional[int] = None,
              metadata_filter: Optional[Dict[str, Any]] = None,
              timeout: Optional[int] = None,
              include_text: bool = False,
              include_values: bool = False,
              include_metadata: bool = False) -> QueryResponse:
        metadata_filter = dict_to_proto_struct(metadata_filter) if metadata_filter is not None else None
        request = QueryRequest(text=text,
                               namespace=namespace,
                               top_k=top_k,
                               filter=metadata_filter,
                               embedding_model=self.embedding_model,
                               include_text=include_text,
                               include_values=include_values,
                               include_metadata=include_metadata)
        return self._wrap_grpc_call(self.stub.Query, request, timeout=timeout)

    def query_by_id(self,
                    id: str,
                    namespace: Optional[str] = None,
                    top_k: Optional[int] = None,
                    metadata_filter: Optional[Dict[str, Any]] = None,
                    timeout: Optional[int] = None,
                    include_text: bool = False,
                    include_values: bool = False,
                    include_metadata: bool = False) -> QueryResponse:
        metadata_filter = dict_to_proto_struct(metadata_filter) if metadata_filter is not None else None
        request = QueryRequest(id=id,
                               namespace=namespace,
                               top_k=top_k,
                               filter=metadata_filter,
                               embedding_model=self.embedding_model,
                               include_text=include_text,
                               include_values=include_values,
                               include_metadata=include_metadata)
        return self._wrap_grpc_call(self.stub.Query, request, timeout=timeout)

    def fetch(self,
              ids: List[str],
              namespace: Optional[str] = None,
              timeout: Optional[int] = None) -> FetchResponse:
        request = FetchRequest(ids=ids, namespace=namespace)
        self._wrap_grpc_call(self.stub.Fetch, request, timeout=timeout)

    def delete_by_ids(self,
                      ids: List[str],
                      namespace: Optional[str] = None,
                      timeout: Optional[int] = None) -> FetchResponse:
        request = DeleteRequest(ids=ids, namespace=namespace)
        self._wrap_grpc_call(self.stub.Delete, request, timeout=timeout)

    def delete_by_metadata_filter(self,
                                  metadata_filter: Dict[str, Any],
                                  namespace: Optional[str] = None,
                                  timeout: Optional[int] = None) -> FetchResponse:
        metadata_filter = dict_to_proto_struct(metadata_filter) if metadata_filter is not None else None
        request = DeleteRequest(filter=metadata_filter, namespace=namespace)
        self._wrap_grpc_call(self.stub.Delete, request, timeout=timeout)

    def delete_all(self,
                   namespace: Optional[str] = None,
                   timeout: Optional[int] = None) -> FetchResponse:
        request = DeleteRequest(namespace=namespace, delete_all=True)
        self._wrap_grpc_call(self.stub.Delete, request, timeout=timeout)

    def upsert_from_dataset(self,
                            dataset: Union[pd.DataFrame, Path, str],
                            text_col_name: str,
                            id_col_name: str,
                            metadata_col_names: Optional[List[str]] = None,
                            namespace: Optional[str] = None,
                            batch_size: int = 100,
                            show_progress: bool = True,
                            timeout: Optional[int] = None) -> UpsertResponse:
        pbar = tqdm(disable=not show_progress, desc='Upserted vectors')
        total_upserted = 0
        for chunk in self._iter_doc_dataset(dataset, text_col_name, id_col_name, batch_size, metadata_col_names):
            request = UpsertRequest(vectors=[doc.to_proto() for doc in chunk],
                                    namespace=namespace,
                                    embedding_model=self.embedding_model)
            response = self._wrap_grpc_call(self.stub.Upsert, request, timeout=timeout)
            total_upserted += response.upserted_count
            pbar.update(batch_size)

        return UpsertResponse(upserted_count=total_upserted)

    @staticmethod
    def _iter_doc_dataset(dataset: Union[pd.DataFrame, Path, str],
                          text_col_name: str,
                          id_col_name: str,
                          batch_size: int,
                          metadata_col_names: Optional[List[str]] = None) -> Iterator[List[Document]]:

        if isinstance(dataset, pd.DataFrame):
            cur_chunk: List[Document] = []
            for tup in dataset.itertuples(index=False):
                doc = Document(text=getattr(tup, text_col_name),
                               id=getattr(tup, id_col_name),
                               metadata={col_name: getattr(tup, col_name) for col_name in metadata_col_names}
                               if metadata_col_names is not None else None)
                if len(cur_chunk) < batch_size:
                    cur_chunk.append(doc)
                else:
                    yield cur_chunk
                    cur_chunk = [doc]

            if len(cur_chunk) > 0:
                yield cur_chunk
        elif isinstance(dataset, Path) or isinstance(dataset, str):
            path = Path(dataset)
            if not path.exists():
                raise FileNotFoundError(f"{path} does not exist")
            if path.is_dir():
                cur_chunk: List[Document] = []
                for f_path in path.iterdir():
                    for chunk in SearchIndex._iter_single_file(f_path,
                                                               text_col_name,
                                                               id_col_name,
                                                               batch_size,
                                                               metadata_col_names):
                        cure_chunk_reminder = batch_size - len(cur_chunk)
                        cur_chunk.extend(chunk[:cure_chunk_reminder])
                        if len(cur_chunk) == batch_size:
                            yield cur_chunk
                            cur_chunk = chunk[cure_chunk_reminder:]
                if len(cur_chunk) > 0:
                    yield cur_chunk
            else:
                for chunk in SearchIndex._iter_single_file(path,
                                                           text_col_name,
                                                           id_col_name,
                                                           batch_size,
                                                           metadata_col_names):
                    yield chunk
        else:
            raise ValueError(f"dataset of type {type(dataset)} is not supported")

    @staticmethod
    def _iter_single_file(file_path: Path,
                          text_col_name: str,
                          id_col_name: str,
                          batch_size: int,
                          metadata_col_names: Optional[List[str]] = None) -> Iterator[List[Document]]:
        with gzip.open(file_path, 'rb') if file_path.suffix == '.gz' else open(file_path, 'r') as f:

            if file_path.suffixes in (['.csv'], ['.csv', '.gz']):
                reader = csv.DictReader(f)
            elif file_path.suffixes in (['.jsonl'], ['.jsonl', '.gz']):
                reader = (json.loads(line) for line in f)
            else:
                raise ValueError(f'Unsupported file format: {file_path.suffixes}')

            cur_chunk: List[Document] = []
            for row in reader:
                metadata = {col_name: row[col_name]
                            for col_name in metadata_col_names} if metadata_col_names else None
                doc = Document(text=row[text_col_name], id=row[id_col_name], metadata=metadata)
                if len(cur_chunk) < batch_size:
                    cur_chunk.append(doc)
                else:
                    yield cur_chunk
                    cur_chunk = [doc]

            if len(cur_chunk) > 0:
                yield cur_chunk
