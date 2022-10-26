# @generated by generate_proto_mypy_stubs.py.  Do not edit!
import sys
from google.protobuf.descriptor import (
    Descriptor as google___protobuf___descriptor___Descriptor,
    FileDescriptor as google___protobuf___descriptor___FileDescriptor,
)

from google.protobuf.internal.containers import (
    RepeatedCompositeFieldContainer as google___protobuf___internal___containers___RepeatedCompositeFieldContainer,
    RepeatedScalarFieldContainer as google___protobuf___internal___containers___RepeatedScalarFieldContainer,
)

from google.protobuf.message import (
    Message as google___protobuf___message___Message,
)

from google.protobuf.struct_pb2 import (
    Struct as google___protobuf___struct_pb2___Struct,
)

from typing import (
    Iterable as typing___Iterable,
    Mapping as typing___Mapping,
    MutableMapping as typing___MutableMapping,
    Optional as typing___Optional,
    Text as typing___Text,
)

from typing_extensions import (
    Literal as typing_extensions___Literal,
)


builtin___bool = bool
builtin___bytes = bytes
builtin___float = float
builtin___int = int


DESCRIPTOR: google___protobuf___descriptor___FileDescriptor = ...

class Vector(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    class SparseValuesEntry(google___protobuf___message___Message):
        DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
        key: builtin___int = ...
        value: builtin___float = ...

        def __init__(self,
            *,
            key : typing___Optional[builtin___int] = None,
            value : typing___Optional[builtin___float] = None,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"key",b"key",u"value",b"value"]) -> None: ...
    type___SparseValuesEntry = SparseValuesEntry

    id: typing___Text = ...
    values: google___protobuf___internal___containers___RepeatedScalarFieldContainer[builtin___float] = ...

    @property
    def sparse_values(self) -> typing___MutableMapping[builtin___int, builtin___float]: ...

    @property
    def metadata(self) -> google___protobuf___struct_pb2___Struct: ...

    def __init__(self,
        *,
        id : typing___Optional[typing___Text] = None,
        values : typing___Optional[typing___Iterable[builtin___float]] = None,
        sparse_values : typing___Optional[typing___Mapping[builtin___int, builtin___float]] = None,
        metadata : typing___Optional[google___protobuf___struct_pb2___Struct] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"metadata",b"metadata"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"id",b"id",u"metadata",b"metadata",u"sparse_values",b"sparse_values",u"values",b"values"]) -> None: ...
type___Vector = Vector

class ScoredVector(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    class SparseValuesEntry(google___protobuf___message___Message):
        DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
        key: builtin___int = ...
        value: builtin___float = ...

        def __init__(self,
            *,
            key : typing___Optional[builtin___int] = None,
            value : typing___Optional[builtin___float] = None,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"key",b"key",u"value",b"value"]) -> None: ...
    type___SparseValuesEntry = SparseValuesEntry

    id: typing___Text = ...
    score: builtin___float = ...
    values: google___protobuf___internal___containers___RepeatedScalarFieldContainer[builtin___float] = ...

    @property
    def sparse_values(self) -> typing___MutableMapping[builtin___int, builtin___float]: ...

    @property
    def metadata(self) -> google___protobuf___struct_pb2___Struct: ...

    def __init__(self,
        *,
        id : typing___Optional[typing___Text] = None,
        score : typing___Optional[builtin___float] = None,
        values : typing___Optional[typing___Iterable[builtin___float]] = None,
        sparse_values : typing___Optional[typing___Mapping[builtin___int, builtin___float]] = None,
        metadata : typing___Optional[google___protobuf___struct_pb2___Struct] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"metadata",b"metadata"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"id",b"id",u"metadata",b"metadata",u"score",b"score",u"sparse_values",b"sparse_values",u"values",b"values"]) -> None: ...
type___ScoredVector = ScoredVector

class UpsertRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    namespace: typing___Text = ...

    @property
    def vectors(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[type___Vector]: ...

    def __init__(self,
        *,
        vectors : typing___Optional[typing___Iterable[type___Vector]] = None,
        namespace : typing___Optional[typing___Text] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"namespace",b"namespace",u"vectors",b"vectors"]) -> None: ...
type___UpsertRequest = UpsertRequest

class UpsertResponse(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    upserted_count: builtin___int = ...

    def __init__(self,
        *,
        upserted_count : typing___Optional[builtin___int] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"upserted_count",b"upserted_count"]) -> None: ...
type___UpsertResponse = UpsertResponse

class DeleteRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    ids: google___protobuf___internal___containers___RepeatedScalarFieldContainer[typing___Text] = ...
    delete_all: builtin___bool = ...
    namespace: typing___Text = ...

    @property
    def filter(self) -> google___protobuf___struct_pb2___Struct: ...

    def __init__(self,
        *,
        ids : typing___Optional[typing___Iterable[typing___Text]] = None,
        delete_all : typing___Optional[builtin___bool] = None,
        namespace : typing___Optional[typing___Text] = None,
        filter : typing___Optional[google___protobuf___struct_pb2___Struct] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"filter",b"filter"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"delete_all",b"delete_all",u"filter",b"filter",u"ids",b"ids",u"namespace",b"namespace"]) -> None: ...
type___DeleteRequest = DeleteRequest

class DeleteResponse(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    def __init__(self,
        ) -> None: ...
type___DeleteResponse = DeleteResponse

class FetchRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    ids: google___protobuf___internal___containers___RepeatedScalarFieldContainer[typing___Text] = ...
    namespace: typing___Text = ...

    def __init__(self,
        *,
        ids : typing___Optional[typing___Iterable[typing___Text]] = None,
        namespace : typing___Optional[typing___Text] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"ids",b"ids",u"namespace",b"namespace"]) -> None: ...
type___FetchRequest = FetchRequest

class FetchResponse(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    class VectorsEntry(google___protobuf___message___Message):
        DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
        key: typing___Text = ...

        @property
        def value(self) -> type___Vector: ...

        def __init__(self,
            *,
            key : typing___Optional[typing___Text] = None,
            value : typing___Optional[type___Vector] = None,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions___Literal[u"value",b"value"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"key",b"key",u"value",b"value"]) -> None: ...
    type___VectorsEntry = VectorsEntry

    namespace: typing___Text = ...

    @property
    def vectors(self) -> typing___MutableMapping[typing___Text, type___Vector]: ...

    def __init__(self,
        *,
        vectors : typing___Optional[typing___Mapping[typing___Text, type___Vector]] = None,
        namespace : typing___Optional[typing___Text] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"namespace",b"namespace",u"vectors",b"vectors"]) -> None: ...
type___FetchResponse = FetchResponse

class QueryVector(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    class SparseValuesEntry(google___protobuf___message___Message):
        DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
        key: builtin___int = ...
        value: builtin___float = ...

        def __init__(self,
            *,
            key : typing___Optional[builtin___int] = None,
            value : typing___Optional[builtin___float] = None,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"key",b"key",u"value",b"value"]) -> None: ...
    type___SparseValuesEntry = SparseValuesEntry

    values: google___protobuf___internal___containers___RepeatedScalarFieldContainer[builtin___float] = ...
    top_k: builtin___int = ...
    namespace: typing___Text = ...

    @property
    def sparse_values(self) -> typing___MutableMapping[builtin___int, builtin___float]: ...

    @property
    def filter(self) -> google___protobuf___struct_pb2___Struct: ...

    def __init__(self,
        *,
        values : typing___Optional[typing___Iterable[builtin___float]] = None,
        sparse_values : typing___Optional[typing___Mapping[builtin___int, builtin___float]] = None,
        top_k : typing___Optional[builtin___int] = None,
        namespace : typing___Optional[typing___Text] = None,
        filter : typing___Optional[google___protobuf___struct_pb2___Struct] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"filter",b"filter"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"filter",b"filter",u"namespace",b"namespace",u"sparse_values",b"sparse_values",u"top_k",b"top_k",u"values",b"values"]) -> None: ...
type___QueryVector = QueryVector

class QueryRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    class SparseVectorEntry(google___protobuf___message___Message):
        DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
        key: builtin___int = ...
        value: builtin___float = ...

        def __init__(self,
            *,
            key : typing___Optional[builtin___int] = None,
            value : typing___Optional[builtin___float] = None,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"key",b"key",u"value",b"value"]) -> None: ...
    type___SparseVectorEntry = SparseVectorEntry

    namespace: typing___Text = ...
    top_k: builtin___int = ...
    include_values: builtin___bool = ...
    include_metadata: builtin___bool = ...
    vector: google___protobuf___internal___containers___RepeatedScalarFieldContainer[builtin___float] = ...
    id: typing___Text = ...

    @property
    def filter(self) -> google___protobuf___struct_pb2___Struct: ...

    @property
    def queries(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[type___QueryVector]: ...

    @property
    def sparse_vector(self) -> typing___MutableMapping[builtin___int, builtin___float]: ...

    def __init__(self,
        *,
        namespace : typing___Optional[typing___Text] = None,
        top_k : typing___Optional[builtin___int] = None,
        filter : typing___Optional[google___protobuf___struct_pb2___Struct] = None,
        include_values : typing___Optional[builtin___bool] = None,
        include_metadata : typing___Optional[builtin___bool] = None,
        queries : typing___Optional[typing___Iterable[type___QueryVector]] = None,
        vector : typing___Optional[typing___Iterable[builtin___float]] = None,
        id : typing___Optional[typing___Text] = None,
        sparse_vector : typing___Optional[typing___Mapping[builtin___int, builtin___float]] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"filter",b"filter"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"filter",b"filter",u"id",b"id",u"include_metadata",b"include_metadata",u"include_values",b"include_values",u"namespace",b"namespace",u"queries",b"queries",u"sparse_vector",b"sparse_vector",u"top_k",b"top_k",u"vector",b"vector"]) -> None: ...
type___QueryRequest = QueryRequest

class SingleQueryResults(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    namespace: typing___Text = ...

    @property
    def matches(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[type___ScoredVector]: ...

    def __init__(self,
        *,
        matches : typing___Optional[typing___Iterable[type___ScoredVector]] = None,
        namespace : typing___Optional[typing___Text] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"matches",b"matches",u"namespace",b"namespace"]) -> None: ...
type___SingleQueryResults = SingleQueryResults

class QueryResponse(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    namespace: typing___Text = ...

    @property
    def results(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[type___SingleQueryResults]: ...

    @property
    def matches(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[type___ScoredVector]: ...

    def __init__(self,
        *,
        results : typing___Optional[typing___Iterable[type___SingleQueryResults]] = None,
        matches : typing___Optional[typing___Iterable[type___ScoredVector]] = None,
        namespace : typing___Optional[typing___Text] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"matches",b"matches",u"namespace",b"namespace",u"results",b"results"]) -> None: ...
type___QueryResponse = QueryResponse

class UpdateRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    id: typing___Text = ...
    values: google___protobuf___internal___containers___RepeatedScalarFieldContainer[builtin___float] = ...
    namespace: typing___Text = ...

    @property
    def set_metadata(self) -> google___protobuf___struct_pb2___Struct: ...

    def __init__(self,
        *,
        id : typing___Optional[typing___Text] = None,
        values : typing___Optional[typing___Iterable[builtin___float]] = None,
        set_metadata : typing___Optional[google___protobuf___struct_pb2___Struct] = None,
        namespace : typing___Optional[typing___Text] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"set_metadata",b"set_metadata"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"id",b"id",u"namespace",b"namespace",u"set_metadata",b"set_metadata",u"values",b"values"]) -> None: ...
type___UpdateRequest = UpdateRequest

class UpdateResponse(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    def __init__(self,
        ) -> None: ...
type___UpdateResponse = UpdateResponse

class DescribeIndexStatsRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def filter(self) -> google___protobuf___struct_pb2___Struct: ...

    def __init__(self,
        *,
        filter : typing___Optional[google___protobuf___struct_pb2___Struct] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"filter",b"filter"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"filter",b"filter"]) -> None: ...
type___DescribeIndexStatsRequest = DescribeIndexStatsRequest

class NamespaceSummary(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    vector_count: builtin___int = ...

    def __init__(self,
        *,
        vector_count : typing___Optional[builtin___int] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"vector_count",b"vector_count"]) -> None: ...
type___NamespaceSummary = NamespaceSummary

class DescribeIndexStatsResponse(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    class NamespacesEntry(google___protobuf___message___Message):
        DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
        key: typing___Text = ...

        @property
        def value(self) -> type___NamespaceSummary: ...

        def __init__(self,
            *,
            key : typing___Optional[typing___Text] = None,
            value : typing___Optional[type___NamespaceSummary] = None,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions___Literal[u"value",b"value"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"key",b"key",u"value",b"value"]) -> None: ...
    type___NamespacesEntry = NamespacesEntry

    dimension: builtin___int = ...
    index_fullness: builtin___float = ...
    total_vector_count: builtin___int = ...

    @property
    def namespaces(self) -> typing___MutableMapping[typing___Text, type___NamespaceSummary]: ...

    def __init__(self,
        *,
        namespaces : typing___Optional[typing___Mapping[typing___Text, type___NamespaceSummary]] = None,
        dimension : typing___Optional[builtin___int] = None,
        index_fullness : typing___Optional[builtin___float] = None,
        total_vector_count : typing___Optional[builtin___int] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"dimension",b"dimension",u"index_fullness",b"index_fullness",u"namespaces",b"namespaces",u"total_vector_count",b"total_vector_count"]) -> None: ...
type___DescribeIndexStatsResponse = DescribeIndexStatsResponse
