import numpy as np
import orjson
from pydantic import BaseModel


def serialize(data) -> str:
    return orjson.dumps(
        data,
        option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS,
    )


deserialize = orjson.loads


# Custom Pydantic class that uses orjson for serialization
class ORJSONBaseModel(BaseModel):
    def model_dump_json(self, **kwargs) -> str:
        return serialize(self.model_dump(**kwargs))

    @classmethod
    def model_construct_json(cls, json_str: str, **kwargs):
        return cls.model_validate(deserialize(json_str))


# Define an input data model that uses orjson for serialization
class InputData(ORJSONBaseModel):
    data: dict


def deserialize_numpy(data):
    """Recursively convert lists back into NumPy arrays."""
    if isinstance(data, list):
        return np.array(data)
    elif isinstance(data, dict):
        return {k: deserialize_numpy(v) for k, v in data.items()}
    return data
