"""TEST FAKE of the classes a DCP `.metadata` pickle names (see `standins/standin.py`): the fields
`dcp-load/kit/params.py` reads, under the real classes' names (the real `StorageInfo` is
`filesystem._StorageInfo`)."""
import dataclasses


@dataclasses.dataclass(frozen=True)
class MetadataIndex:
    fqn: str
    offset: tuple


@dataclasses.dataclass
class StorageInfo:
    relative_path: str
    offset: int
    length: int


@dataclasses.dataclass
class ChunkStorageMetadata:
    offsets: tuple
    sizes: tuple


@dataclasses.dataclass
class TensorProperties:
    dtype: object


@dataclasses.dataclass
class TensorStorageMetadata:
    properties: TensorProperties
    chunks: list


@dataclasses.dataclass
class Metadata:
    state_dict_metadata: dict
    storage_data: dict
