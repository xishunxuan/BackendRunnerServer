import json
import logging
import pathlib
import traceback
from types import SimpleNamespace
from typing import Union

import torch
import io
import numpy as np
import base64
import safetensors
import safetensors.torch
from numpy.random import BitGenerator
import orjson


logger = logging.getLogger(__name__)
"""
该模块提供了将对象序列化和反序列化为JSON字符串的功能。Developed by Shijian Zheng 2023
它支持基本数据类型（int, float, str）、复杂数据结构（list, dict(键必须是字符串)）以及NumPy数组，PyTorch张量Tensor的序列化。NumPy数组、PyTorch张量Tensor的序列化避免了使用Pickle，从而提高了安全性与不同操作系统、包版本间兼容性
在序列化过程中，保留了每个属性的类型信息，以便能够进行准确的反序列化。
在反序列化过程中，利用Python的`types`模块中的`SimpleNamespace`进行动态对象创建。

功能函数:
- safe_type_convert: 使用预定义的转换器函数安全地将值转换为指定类型。
- serialize_to_json_with_types: 将对象的属性序列化为保留类型信息的JSON字符串。
- deserialize_from_json_with_types: 将JSON字符串反序列化回对象，恢复属性的原始数据类型。
- [类型转换器函数]: 一系列函数，用于将字符串表示转换为其相应的类型（例如，int_converter, float_converter）。

模块还包括一个测试用例，演示了包含各种数据类型和结构的示例对象的序列化和反序列化过程，包括嵌套列表、字典和NumPy数组。

注意事项: 
- dict数据: 由于json的限制，键必须是字符串
- PyTorch张量Tensor: 反序列化后默认放在cpu上
"""

"""
扩展数据类型支持:
若需支持新的数据类型的序列化和反序列化，请按照以下步骤进行：

1. 定义新的类型转换器函数:
   创建一个新的函数，它接受原始类型值并返回其字符串表示（对于序列化），或接受字符串表示并返回原始类型值（对于反序列化）。

   例如，若要支持日期类型（datetime）：
   def datetime_converter(value):
       return datetime.strptime(value, '%Y-%m-%dT%H:%M:%S')

2. 在 `serialize_recursively` 函数中添加支持:
   修改 `serialize_recursively` 函数，添加对新数据类型的支持。当遇到新数据类型时，函数应该返回一个字典，包含值的字符串表示及其类型名称。

   例如，对于datetime对象：
   elif isinstance(value, datetime):
       return {'value': value.strftime('%Y-%m-%dT%H:%M:%S'), 'type': 'datetime'}

3. 更新 `type_converters` 字典:
   在 `type_converters` 字典中添加一个新条目，其键为类型名称（如 'datetime'），值为步骤1中定义的转换器函数。

   例如：
   'datetime': datetime_converter

通过遵循这些步骤，可以轻松地将新的数据类型添加到序列化和反序列化过程中。
"""

"""
高级序列化与反序列化支持说明：

支持自定义的数据类型（尤其是那些可以被转换为bytes类型的）可以被序列化并储存为字符串形式，并在需要时可以从这些字符串中恢复原始数据类型。

实现步骤：

1. 数据类型到bytes的转换：
确保你需要支持的数据类型可以被转换为bytes类型。例如，对于numpy.ndarray，你可以使用numpy.save方法将其转换为bytes。

2. 序列化（数据类型到字符串的转换）：
一旦数据类型被转换为bytes，使用base64编码将这些bytes转换为字符串。这样可以确保bytes数据在JSON、XML等中可以安全传输。

3. 反序列化（字符串到数据类型的转换）：
接收到序列化的字符串后，先将其转换回bytes（通常是通过base64解码），然后使用这些bytes来重建原始数据类型。

安全注意事项：

- 当从bytes重建数据类型时，尤其是当这些bytes来自不可信的源时，这可能存在安全风险。例如，使用pickle或eval从不可信的源加载数据可能导致代码执行漏洞。因此，应该始终确保使用安全的方法来从bytes重建数据。

- 一些数据类型的序列化方式的优缺点对比可见https://github.com/huggingface/safetensors

- 对于任何需要添加支持的新数据类型，你需要为它实现相应的转换函数，并将这些函数添加到type_converters字典中，以便在序列化和反序列化时可以被调用。

示例：
在代码中已经实现了numpy.ndarray类型的序列化与反序列化支持，见numpy_converter，与serialize_to_json_with_types

"""

"""
拓展新类型但免注册的序列化方法
1. 实现的to_str, from_str方法
2. 调用本模块内的注册函数：register_new_type(class_T, class_T.to_str, class_T.from_str)
这样一来，序列化工具会识别上述新类型对象，并自动实现序列化与反序列化

"""
# todo 已知问题(待解决)
"""
1. 变量类型为np.float64时不能成功序列化

"""




# 定义各种类型转换函数
def int_converter(value):
    return int(value)


def float_converter(value):
    return float(value)


def str_converter(value):
    return str(value)


def list_converter(value):
    return [safe_type_convert(item['value'], item['type']) for item in value]


def tuple_converter(value):
    return tuple([safe_type_convert(item['value'], item['type']) for item in value])


def dict_converter(value):
    return {k: safe_type_convert(v['value'], v['type']) for k, v in value.items()}


def advanced_dict_converter(value):
    # 处理key类型不为str的字典。其数据结构为列表：[(key,value)]
    return {safe_type_convert(k['value'], k['type']): safe_type_convert(v['value'], v['type']) for k, v in value}


def numpy_converter(value: str):
    decode_value = base64.decodebytes(value.encode())
    f = io.BytesIO()
    f.write(decode_value)
    f.seek(0)
    return np.load(f, allow_pickle=False)


def tensor_converter(value: str):
    value = base64.decodebytes(value.encode())
    tensor_dict = safetensors.torch.load(value)
    return tensor_dict['value']


def ERROR_converter():
    return None


def pathlib_path_converter(value: str):
    value = pathlib.Path(value)
    return value


def none_type_converter(value: str):
    return None


def simple_namespace_converter(value: str):
    return deserialize_from_json_with_types(value)


# 转换函数映射
type_converters = {
    'int': int_converter,
    'float': float_converter,
    'int32': np.int32,
    'bool': bool,
    'int64': np.int64,
    'float32': np.float32,
    'float64': np.float64,
    'uint8': np.uint8,
    'str': str_converter,
    'list': list_converter,
    'dict': dict_converter,
    'advanced_dict': advanced_dict_converter,
    'ndarray': numpy_converter,
    'torch.Tensor': tensor_converter,
    'tuple': tuple_converter,
    'pathlib.Path': pathlib_path_converter,
    'NoneType': none_type_converter,
    'ERROR': ERROR_converter,
    'SimpleNamespace': simple_namespace_converter
    # 添加更多类型映射 ...
}
custom_type_converters = {}
custom_to_str_methods = {}


def register_new_type(cls_t, to_str_fun, from_str_fun, name=None):
    """
    动态地在序列化工具模块中注册一个新的可支持类型，使得SimpleNamespace中可以支持更多的自定义数据类型。
    注意：
    1. 注册的类型只能在SimpleNamespace内部放置，直接反序列化或者序列化新注册数据是不支持的。
    2. 线程、进程不安全。跨进程注册不同步。在单进程中跨模块全局生效。
    Args:
        cls_t: 对象的 type
        to_str_fun: Callable[[object],str] 输入对象，返回序列化后的字符串的函数
        from_str_fun: Callable[[str],object] 输入字符串，输出反序列化后对象的函数
        name: 对象的名称，为None时取cls_t的名称（ cls_t.__name__）

    Returns:

    """
    if name is None:
        name = cls_t.__name__
    custom_type_converters[name] = from_str_fun
    custom_to_str_methods[cls_t] = (to_str_fun, name)
    return True


def safe_type_convert(value, value_type):
    converter = type_converters.get(value_type)
    if converter:
        return converter(value)
    elif value_type in custom_type_converters:
        method = custom_type_converters[value_type]
        return method(value)
    else:
        raise ValueError(f"Unsupported type: {value_type}")


def serialize_to_json_with_types(obj, ignore_unsupported=False, pretty_json=False) -> str:
    """

    Args:
        obj:
        ignore_unsupported: 是否忽略错误。若是，将把不支持类型的值序列化为None
        pretty_json:

    Returns:

    """

    def serialize_recursively(value, ignore_unsupported=False):
        if isinstance(value, (int, float, str)):
            return {'value': value, 'type': type(value).__name__}
        if isinstance(value, (np.int32, np.int64, np.float32, np.float64, np.ubyte)):
            # 该功能依赖于np.dtypeXX(str(a))) = a
            return {'value': str(value), 'type': type(value).__name__}
        elif isinstance(value, list):
            return {'value': [serialize_recursively(item) for item in value], 'type': 'list'}
        elif isinstance(value, dict):
            return {'value': [(serialize_recursively(k), serialize_recursively(v)) for k, v in value.items()],
                    'type': 'advanced_dict'}
        elif isinstance(value, np.ndarray):
            bytes_io = io.BytesIO()
            np.save(bytes_io, value, allow_pickle=False)
            return {'value': base64.encodebytes(bytes_io.getvalue()).decode(), 'type': 'ndarray'}
        elif isinstance(value, torch.Tensor):
            bytes_io = safetensors.torch.save({"value": value})
            return {'value': base64.encodebytes(bytes_io).decode(), 'type': 'torch.Tensor'}
        elif isinstance(value, tuple):
            return {'value': [serialize_recursively(item) for item in value], 'type': 'tuple'}
        elif isinstance(value, pathlib.Path):
            return {'value': str(value), 'type': 'pathlib.Path'}
        elif value is None:
            return {'value': "None", 'type': 'NoneType'}
        elif isinstance(value, SimpleNamespace):
            return {'value': serialize_to_json_with_types(value, ignore_unsupported=ignore_unsupported,
                                                          pretty_json=pretty_json), 'type': 'SimpleNamespace'}
        elif type(value) in custom_to_str_methods.keys():
            method, name = custom_to_str_methods[type(value)]
            return {'value': method(value), 'type': name}
        # 添加更多你需要支持的类型转换
        else:
            if not ignore_unsupported:
                raise ValueError(f"Unsupported type: {type(value).__name__}")
            else:
                print(f"序列化工具:忽略不支持对象:{type(value).__name__}")
                return None

    if hasattr(obj, '__dict__'):
        attrs = vars(obj)
        serialized_data = {}
        for key, value in attrs.items():
            res = serialize_recursively(value, ignore_unsupported=ignore_unsupported)
            if res is not None:
                serialized_data[key] = res
    else:
        raise Exception("不支持对非对象类型调用序列化")
    if pretty_json:
        return json.dumps(serialized_data, indent=0)
    else:
        return orjson.dumps(serialized_data).decode("utf-8")


def deserialize_from_json_with_types(json_str: str) -> SimpleNamespace:
    try:
        # attrs_with_types = json.loads(json_str)
        attrs_with_types = orjson.loads(json_str)
        obj = SimpleNamespace()
        for key, value_dict in attrs_with_types.items():
            try:
                obj.__setattr__(key, safe_type_convert(value_dict['value'], value_dict['type']))
            except (ValueError, TypeError) as e:
                # 如果类型转换失败，保持原始值
                logger.info("转换失败")
                obj.__setattr__(key, value_dict['value'])
                raise e
        return obj
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON string")


def deserialize_from_file_with_types(path: Union[str, pathlib.Path]) -> SimpleNamespace:
    try:
        with open(path, 'r', encoding='utf-8') as file:
            data = file.read()
        return deserialize_from_json_with_types(data)
    except FileNotFoundError:
        raise Exception("文件未找到，请检查文件路径是否正确。")
    except IOError:
        raise Exception("文件读取出错。")
    except Exception as e:
        raise Exception("解析错误:" + str(e))


def serialize_to_file_with_types(path, data: Union[str], ignore_unsupported=False, pretty_json=False):
    try:
        with open(path, 'w+', encoding='utf-8') as file:
            data = serialize_to_json_with_types(data, ignore_unsupported, pretty_json)
            file.write(data)
        return
    except FileNotFoundError:
        raise Exception("文件未找到，请检查文件路径是否正确。")
    except IOError:
        raise Exception("文件写入出错。")
    except Exception as e:
        raise Exception("解析错误:" + str(e))


def check_serialization_consistency(original_obj, restored_obj):
    original_attrs = vars(original_obj)
    restored_attrs = vars(restored_obj)

    # 检查属性的数量是否一致
    if len(original_attrs) != len(restored_attrs):
        return False

    # 检查每个属性的键和值是否一致
    for key, original_value in original_attrs.items():
        restored_value = restored_attrs.get(key)
        if not check_value_equality(original_value, restored_value):
            return False

    return True


def check_value_equality(original_value, restored_value):
    if isinstance(original_value, np.ndarray):
        # 比较Numpy数组
        return np.array_equal(original_value, restored_value)
    elif isinstance(original_value, list):
        # 递归检查列表中的每个元素
        return all(check_value_equality(orig_item, rest_item)
                   for orig_item, rest_item in zip(original_value, restored_value))
    elif isinstance(original_value, dict):
        # 递归检查字典中的每个键值对
        return all(key in restored_value and check_value_equality(val, restored_value[key])
                   for key, val in original_value.items())
    elif isinstance(original_value, torch.Tensor):
        # 比较PyTorch张量Tensor，如果device不同，统一在original_value的device上比较
        if original_value.device != restored_value.device:
            restored_value = restored_value.to(original_value.device)
        return torch.equal(original_value, restored_value)
    elif isinstance(original_value, tuple):
        return all(check_value_equality(orig_item, rest_item)
                   for orig_item, rest_item in zip(original_value, restored_value))
    elif isinstance(original_value, SimpleNamespace):
        return check_serialization_consistency(original_value, restored_value)
    elif hasattr(original_value, '__dict__') and not isinstance(original_value, SimpleNamespace):
        if type(original_value) != type(restored_value):
            return False
        else:
            return check_serialization_consistency(original_value, restored_value)
    else:
        # 对于其他类型，使用标准比较
        return original_value == restored_value


#
#
# class BytesSerializableObject:
#     """
#     可序列化对象，要求拥有to_bytes() from_bytes()两种方法
#     to_bytes() 返回自身序列化的字节
#     from_bytes(input_bytes) 输入bytes，返回bytes对应的对象，要求为staticmethod
#     """
#
#     def __init__(self, *args, **kwargs):
#         pass
#
#     def to_bytes(self) -> bytes:
#         pass
#
#     @staticmethod
#     def from_bytes(cls, input_bytes: bytes):
#         pass
#
#     def to_str(self):
#         base64.encodebytes(self.to_bytes())


class StrSerializableObject:
    """
    可序列化对象，要求拥有to_str() from_str()两种方法
    to_str() 返回自身序列化的字符串
    from_str(input_str) 输入str，返回str对应的对象，要求为staticmethod
    """

    def __init__(self, *args, **kwargs):
        pass

    def to_bytes(self) -> bytes:
        pass

    @staticmethod
    def from_bytes(cls, input_bytes: bytes):
        pass
