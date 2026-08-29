"""Minimal protobuf wire codec for the Cursor SDK bridge callback messages.

The bridge (`cursor-sdk-bridge`, see https://github.com/cursor/sdk-bridge)
calls back into Hermes over Connect unary POSTs when a custom tool executes
(`sdk.v1.SdkCustomToolCallbackService/CallCustomTool`).  Connect clients may
encode those requests as JSON (`application/json`) or binary protobuf
(`application/proto`).  Hermes speaks JSON for everything it initiates, but
the callback server must accept whichever encoding the bridge uses.

Rather than pulling in generated protobuf stubs (and a codegen pipeline) for
two small messages, this module hand-rolls the proto3 wire format for exactly
the types the callback needs:

    message CallCustomToolRequest {
      string tool_name = 1;
      google.protobuf.Struct args = 2;
      optional string tool_call_id = 3;
      string agent_id = 4;
    }

    message CallCustomToolResponse {
      google.protobuf.Struct result = 1;
    }

plus ``google.protobuf.Struct`` / ``Value`` / ``ListValue`` which map 1:1 to
JSON objects / values / arrays.
"""

from __future__ import annotations

from typing import Any

MAX_FRAME_BYTES = 1 * 1024 * 1024
MAX_CONTAINER_BYTES = 256 * 1024

# ── varint ────────────────────────────────────────────────────────────────


def _encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint must be non-negative")
    out = bytearray()
    while True:
        bits = value & 0x7F
        value >>= 7
        if value:
            out.append(bits | 0x80)
        else:
            out.append(bits)
            return bytes(out)


def _decode_varint(data: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        if pos >= len(data):
            raise ValueError("truncated varint")
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def _read_delimited(data: bytes, pos: int, *, what: str) -> tuple[bytes, int]:
    length, pos = _decode_varint(data, pos)
    if length > MAX_CONTAINER_BYTES:
        raise ValueError(f"{what} exceeds size limit")
    end = pos + length
    if end > len(data):
        raise ValueError(f"truncated {what}")
    return data[pos:end], end


def _tag(field_number: int, wire_type: int) -> bytes:
    return _encode_varint((field_number << 3) | wire_type)


def _len_delimited(field_number: int, payload: bytes) -> bytes:
    return _tag(field_number, 2) + _encode_varint(len(payload)) + payload


# ── google.protobuf.Struct <-> Python ─────────────────────────────────────
#
# Struct   { map<string, Value> fields = 1; }   (map entry: key=1, value=2)
# Value    { oneof kind { NullValue null_value = 1; double number_value = 2;
#            string string_value = 3; bool bool_value = 4;
#            Struct struct_value = 5; ListValue list_value = 6; } }
# ListValue{ repeated Value values = 1; }


def encode_struct(obj: dict[str, Any]) -> bytes:
    if not isinstance(obj, dict):
        raise TypeError("Struct payload must be a dict")
    out = bytearray()
    for key, value in obj.items():
        entry = _len_delimited(1, str(key).encode("utf-8")) + _len_delimited(
            2, _encode_value(value)
        )
        out += _len_delimited(1, entry)
    return bytes(out)


def _encode_value(value: Any) -> bytes:
    if value is None:
        return _tag(1, 0) + _encode_varint(0)  # NULL_VALUE
    if isinstance(value, bool):
        return _tag(4, 0) + _encode_varint(1 if value else 0)
    if isinstance(value, (int, float)):
        import struct as _struct

        return _tag(2, 1) + _struct.pack("<d", float(value))
    if isinstance(value, str):
        return _len_delimited(3, value.encode("utf-8"))
    if isinstance(value, dict):
        return _len_delimited(5, encode_struct(value))
    if isinstance(value, (list, tuple)):
        payload = bytearray()
        for item in value:
            payload += _len_delimited(1, _encode_value(item))
        return _len_delimited(6, bytes(payload))
    raise TypeError(f"Struct cannot encode value of type {type(value).__name__}")


def decode_struct(data: bytes) -> dict[str, Any]:
    result: dict[str, Any] = {}
    pos = 0
    while pos < len(data):
        tag, pos = _decode_varint(data, pos)
        field_number, wire_type = tag >> 3, tag & 0x07
        if field_number == 1 and wire_type == 2:
            entry, pos = _read_delimited(data, pos, what="map entry")
            key, value = _decode_map_entry(entry)
            result[key] = value
        else:
            pos = _skip_field(data, pos, wire_type)
    return result


def _decode_map_entry(entry: bytes) -> tuple[str, Any]:
    key = ""
    value: Any = None
    pos = 0
    while pos < len(entry):
        tag, pos = _decode_varint(entry, pos)
        field_number, wire_type = tag >> 3, tag & 0x07
        if field_number == 1 and wire_type == 2:
            raw, pos = _read_delimited(entry, pos, what="map key")
            key = raw.decode("utf-8")
        elif field_number == 2 and wire_type == 2:
            raw, pos = _read_delimited(entry, pos, what="map value")
            value = _decode_value(raw)
        else:
            pos = _skip_field(entry, pos, wire_type)
    return key, value


def _decode_value(data: bytes) -> Any:
    import struct as _struct

    value: Any = None
    pos = 0
    while pos < len(data):
        tag, pos = _decode_varint(data, pos)
        field_number, wire_type = tag >> 3, tag & 0x07
        if field_number == 1 and wire_type == 0:  # null_value
            _, pos = _decode_varint(data, pos)
            value = None
        elif field_number == 2 and wire_type == 1:  # number_value
            if pos + 8 > len(data):
                raise ValueError("truncated number value")
            value = _struct.unpack("<d", data[pos : pos + 8])[0]
            # Render integral doubles as ints for friendlier tool args.
            if isinstance(value, float) and value.is_integer() and abs(value) < 2**53:
                value = int(value)
            pos += 8
        elif field_number == 3 and wire_type == 2:  # string_value
            raw, pos = _read_delimited(data, pos, what="string value")
            value = raw.decode("utf-8")
        elif field_number == 4 and wire_type == 0:  # bool_value
            raw, pos = _decode_varint(data, pos)
            value = bool(raw)
        elif field_number == 5 and wire_type == 2:  # struct_value
            raw, pos = _read_delimited(data, pos, what="struct value")
            value = decode_struct(raw)
        elif field_number == 6 and wire_type == 2:  # list_value
            raw, pos = _read_delimited(data, pos, what="list value")
            value = _decode_list(raw)
        else:
            pos = _skip_field(data, pos, wire_type)
    return value


def _decode_list(data: bytes) -> list[Any]:
    items: list[Any] = []
    pos = 0
    while pos < len(data):
        tag, pos = _decode_varint(data, pos)
        field_number, wire_type = tag >> 3, tag & 0x07
        if field_number == 1 and wire_type == 2:
            raw, pos = _read_delimited(data, pos, what="list item")
            items.append(_decode_value(raw))
        else:
            pos = _skip_field(data, pos, wire_type)
    return items


def _skip_field(data: bytes, pos: int, wire_type: int) -> int:
    if wire_type == 0:  # varint
        _, pos = _decode_varint(data, pos)
        return pos
    if wire_type == 1:  # 64-bit
        end = pos + 8
        if end > len(data):
            raise ValueError("truncated fixed64 field")
        return end
    if wire_type == 2:  # length-delimited
        _, pos = _read_delimited(data, pos, what="field")
        return pos
    if wire_type == 5:  # 32-bit
        end = pos + 4
        if end > len(data):
            raise ValueError("truncated fixed32 field")
        return end
    raise ValueError(f"unsupported wire type {wire_type}")


# ── CallCustomTool messages ───────────────────────────────────────────────


def decode_call_custom_tool_request(data: bytes) -> dict[str, Any]:
    """Decode a binary ``CallCustomToolRequest`` into a plain dict.

    Returns ``{"toolName", "args", "toolCallId", "agentId"}`` using the same
    camelCase keys the JSON encoding produces, so callers handle one shape.
    """
    result: dict[str, Any] = {"toolName": "", "args": {}, "toolCallId": None, "agentId": ""}
    pos = 0
    while pos < len(data):
        tag, pos = _decode_varint(data, pos)
        field_number, wire_type = tag >> 3, tag & 0x07
        if field_number == 1 and wire_type == 2:
            raw, pos = _read_delimited(data, pos, what="tool name")
            result["toolName"] = raw.decode("utf-8")
        elif field_number == 2 and wire_type == 2:
            raw, pos = _read_delimited(data, pos, what="arguments")
            result["args"] = decode_struct(raw)
        elif field_number == 3 and wire_type == 2:
            raw, pos = _read_delimited(data, pos, what="tool call id")
            result["toolCallId"] = raw.decode("utf-8")
        elif field_number == 4 and wire_type == 2:
            raw, pos = _read_delimited(data, pos, what="agent id")
            result["agentId"] = raw.decode("utf-8")
        else:
            pos = _skip_field(data, pos, wire_type)
    return result


def encode_call_custom_tool_response(result: dict[str, Any]) -> bytes:
    """Encode ``CallCustomToolResponse{result: Struct}`` to binary protobuf."""
    return _len_delimited(1, encode_struct(result))
