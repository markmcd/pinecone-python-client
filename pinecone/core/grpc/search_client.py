from dataclasses import dataclass

from pinecone.core.grpc.index_grpc import GRPCIndexBase
from pinecone.core.grpc.protos.search_service_pb2_grpc import SearchServiceStub
from pinecone.core.grpc.protos.search_service_pb2 import UpsertRequest, QueryRequest, UpsertResponse, QueryResponse, \
    FetchRequest, FetchResponse, DeleteRequest
from pinecone.core.grpc.protos.search_service_pb2 import TextVector as ProtoTextVector
import logging
from typing import Optional, List, Dict, Any, Union

from tqdm import tqdm
from uuid import uuid4
from pinecone.core.utils import dict_to_proto_struct

_logger = logging.getLogger(__name__)


@dataclass
class Text:
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
               texts: Union[List[Text], List[str]],
               namespace: Optional[str] = None,
               batch_size: Optional[int] = None,
               show_progress: bool = True,
               timeout: Optional[int] = None) -> UpsertResponse:
        texts = [ProtoTextVector(id=str(uuid4()), text=v) if isinstance(v, str)
                 else v.to_proto()
                 for v in texts]
        if batch_size is None:
            request = UpsertRequest(vectors=texts, namespace=namespace, embedding_model=self.embedding_model)
            return self._wrap_grpc_call(self.stub.Upsert, request, timeout=timeout)

        pbar = tqdm(total=len(texts), disable=not show_progress, desc='Upserted vectors')
        total_upserted = 0
        for i in range(0, len(texts), batch_size):
            request = UpsertRequest(vectors=texts[i:i + batch_size], namespace=namespace)
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
