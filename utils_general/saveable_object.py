
from abc import ABC, abstractmethod
import copy
import inspect
import json
import os
from typing import Any, Union
# Python 3.11+: can do "from typing import Self"
# Before Python 3.11: need to do the following:
from typing_extensions import Self
import warnings

from utils_general.io_utils import load_json, safe_issubclass, save_json


# Dictionary to keep track of subclasses of SaveableObject
_CLASSES: dict[str, type] = {}


def _info_dict_to_instance(info_dict) -> tuple[type, dict]:
    """Creates an instance of a class from a dictionary containing its class name
    and arguments."""
    if set(info_dict.keys()) != {"class_name", "kwargs"}:
        raise ValueError("info_dict should have keys 'class_name' and 'kwargs'")
    
    class_name = info_dict["class_name"]
    loaded_cls = _get_class_from_name(class_name)
    
    kwargs = info_dict["kwargs"]
    if not isinstance(kwargs, dict):
        raise TypeError("'kwargs' should be a dictionary")
    for k, v in kwargs.items():
        class_ = None
        if type(v) is str:
            try:
                class_ = _get_class_from_name(v)
            except KeyError:
                pass
        if class_ is not None:
            kwargs[k] = class_
        elif isinstance(v, dict) and set(v.keys()) == {"class_name", "kwargs"}:
            v_cls, v_kwargs = _info_dict_to_instance(v)
            kwargs[k] = v_cls(**v_kwargs)
    return loaded_cls, kwargs


def _get_class_from_name(class_name):
    try:
        loaded_cls = _CLASSES[class_name]
    except KeyError:
        try:
            # Backwards compatibility with old code
            if '.' not in class_name:
                raise KeyError
            new_class_name = class_name.split(".")[-1]
            warnings.warn(
                f"Subclass {class_name} of SaveableObject is not registered. "
                f"Must be old code. Trying to get it as {new_class_name}")
            loaded_cls = _CLASSES[new_class_name]
        except KeyError:
            raise KeyError(f"Subclass {class_name} of SaveableObject does not exist")
    return loaded_cls


def _default_json_SaveableObject(o):
    if o.__class__ is type:
        the_type = o.__name__
        tmp = f"the type {the_type}"
    else:
        the_type = o.__class__.__name__
        tmp = f"of type {the_type}"
    msg = "Unable to save the file because one of the values is not JSON " \
        "serializable. Only JSON serializable values and SaveableObject " \
        "subclasses and instances are supported to be saved by SaveableObject. " \
        f"If you need to save this object ({tmp}), consider making {the_type} " \
        "a subclass of SaveableObject."
    raise TypeError(msg)


def _get_class_name(c: type):
    # return f"{c.__module__}.{c.__name__}"  # Old code was this
    return c.__name__                        # This allows for moving files around

class SaveableObject(ABC):
    r"""An abstract base class for objects that can be saved and loaded with their
    initialization parameters.

    This class enables automatic serialization and deserialization of subclasses by 
    tracking their __init__ arguments. When a subclass is instantiated, its 
    constructor arguments are introspected and stored, allowing the object to be 
    reconstructed from saved data. Supports instances of `SaveableObject`
    as well as subclasses of `SaveableObject` as arguments to __init__.

    Subclass __init__ must not use `*args` (variadic positional arguments).
    
    Subclasses should subclass `SaveableObject` with `SaveableObject` as the last
    one, i.e., as in,
    ```class C(A, B, SaveableObject)
    ```
    
    *Note:* although internally, SaveableObject has an abstract method __init__, this is
    NOT for purpose of an interface/contract: since SaveableObject automatically assigns
    an explicit __init__ method whenever a new subclass is created, subclasses are NOT
    required to implement __init__.
    Rather, the purpose of this abstract __init__ method is to have at least one
    @abstractmethod in the abstract base class SaveableObject so that SaveableObject is
    prevented from being instantiated directly.
    """
    @classmethod
    def load_init(cls, folder: str) -> Self:
        r"""Loads an instance from a JSON file.

        This method reads the JSON file 'init.json' from the specified folder,
        reconstructs the initialization parameters stored therein, and returns a new
        instance of the appropriate class.

        Usage:
        - To load any `SaveableObject` instance using the base class:
                ```instance = SaveableObject.load_init("/path/to/folder")
                ```
        - To load an instance of a specific subclass `MySubClass`:
                ```instance = MySubClass.load_init("/path/to/folder")
                ```
        In the latter case, ensure that the JSON file contains data for a `MySubClass`
        instance; otherwise, a TypeError will be raised.

        Args:
            folder (str): The directory containing the 'init.json' file.

        Returns:
            The loaded instance.

        Raises:
            RuntimeError:
                If the JSON file could not be loaded (e.g., file not found,
                JSON decoding error, or Unicode decode error).
            TypeError:
                If the JSON file contains an instance of a different class than
                expected.
        """
        try:
            info_dict = load_json(os.path.join(folder, "init.json"))
        except (FileNotFoundError,
                json.decoder.JSONDecodeError, UnicodeDecodeError) as e:
            raise RuntimeError(f"Could not load {cls.__name__} instance") from e
        typ, kwargs = _info_dict_to_instance(info_dict)
        if not issubclass(typ, cls):
            raise TypeError(
                f"Trying to load a {cls.__name__} instance but the JSON file "
                f"contains a {typ.__name__} instance. Instead of calling "
                f"{cls.__name__}.load_init, you could call "
                f"SaveableObject.load_init or {typ.__name__}.load_init"
            )
        return typ(**kwargs)

    def save_init(self, folder: str):
        r"""Saves initialization parameters to a JSON file.
        Args:
            folder (str): The directory to save the JSON file.
        """
        os.makedirs(folder, exist_ok=True)
        save_json(self.get_info_dict(), os.path.join(folder, "init.json"),
                  default=_default_json_SaveableObject)
    
    def get_init_kwargs(self) -> dict[str, Any]:
        r"""Returns a dictionary of initialization parameters of the SaveableObject."""
        return {**self._init_kwargs}

    def get_info_dict(self) -> dict[str, Union[str, dict[str, Any]]]:
        r"""Returns a dictionary representation of initialization parameters
        and class name."""
        return {
            "class_name": _get_class_name(self.__class__),
            "kwargs": self.get_init_kwargs()
        }

    @abstractmethod
    def __init__(self, *args, **kwargs):
        # Default __init__ for SaveableObject that does nothing.
        # Prevents SaveableObject from being instantiated directly.
        # Although this is an abstractmethod, subclasses do NOT need to
        # implement __init__.
        pass

    def __init_subclass__(cls, **kwargs):
        # Preserve the original __init__ method.
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            # If we are constructing THIS exact class (and not a subclass further down
            # the hierarchy). i.e., check if if we are calling the same __init__ as the
            # __init__ defined directly in the object that is being constructed,
            # not a superclass' __init__.
            is_same_class = self.__class__ is cls

            if is_same_class:
                # Copy the original args and kwargs, just in case original_init modifies
                # them
                original_args = copy.deepcopy(args)
                original_kwargs = copy.deepcopy(kwargs)

            original_init(self, *args, **kwargs) # Call the original __init__ method

            if is_same_class:
                # Convert args to kwargs
                sig = inspect.signature(original_init)
                
                # Can use either bind or bind_partial;
                # we already ensured that all required arguments are passed
                # because we already called the original __init__ method.
                # Need to remember to put 'self' in the arguments.
                bound_args = sig.bind(self, *original_args, **original_kwargs)
                bound_args.apply_defaults()
                all_kwargs = bound_args.arguments

                # Detect VAR_KEYWORD parameters (i.e. **kwargs) and flatten them
                # to fix the problem that all_kwargs = {..., 'kwargs': {...}}
                # arises due to apply_defaults when there's ** kind of parameters.
                # Also, make sure that there are no *args type of parameters
                # because we don't support that (makes it too complicated).
                for p_name, param in sig.parameters.items():
                    if param.kind == param.VAR_KEYWORD:
                        if p_name in all_kwargs:
                            # all_kwargs[p_name] is the dict that ended up in **whatever
                            var_kw_dict = all_kwargs.pop(p_name) # remove from top-level
                            # Flatten all items inside that dict:
                            all_kwargs.update(var_kw_dict)
                    elif param.kind == param.VAR_POSITIONAL:
                        value = all_kwargs[p_name]
                        assert type(value) is tuple
                        if len(value) > 0:
                            raise ValueError(
                                "SaveableObject does not support *args in __init__")
                        all_kwargs.pop(p_name)

                # Remove 'self' from the kwargs
                all_kwargs.pop('self', None)

                for k, v in all_kwargs.items():
                    # If there are any subclasses of SaveableObject in the
                    # kwargs, replace them with their name
                    if safe_issubclass(v, SaveableObject):
                        all_kwargs[k] = _get_class_name(v)
                    # If there are any instances of SaveableObject in the
                    # kwargs, replace them with their info_dict
                    elif isinstance(v, SaveableObject):
                        all_kwargs[k] = v.get_info_dict()

                self._init_kwargs = all_kwargs

        # Replace the __init__ method with the new one
        cls.__init__ = new_init

        # Register the class in the _CLASSES dictionary
        _CLASSES[_get_class_name(cls)] = cls

        # Call the original __init_subclass__ method
        super().__init_subclass__(**kwargs)
