from typing import Any


class TypedDict(dict):
    """
    A dictionary that only allows values of value_class, 
    keys of type str, and no overriding of values. All keys are cast to 
    lowercase.
    """
    def __init__(self, value_klass: Any):
        """Constructor of TypedDict
        
        :param value_klass: [description]
        :type value_klass: Any
        """
        self._value_klass = value_klass

    def __getitem__(self, key: str) -> Any:
        """Gets the value with key
        
        :param key: The key of the value
        :type key: str
        :return: The value at key
        :rtype: Any (value_klass)
        """
        return super().__getitem__(key.lower())

    def __setitem__(self, key: str, value: Any):
        """Set the key value pair. Value must be of 
        
        :param key: The key of the value
        :type key: str
        :param value: The value for the key
        :type value: Any (value_klass)
        """
        if not isinstance(key, str):
            raise TypeError("Key must be of type str for TypedDict")
        if not isinstance(value, self._value_klass):
            raise TypeError(
                f"value must be an instance of {self._value_klass} for TypedDict"
            )
        return super().__setitem__(key.lower(), value)

    def __delitem__(self, key: str) -> bool:
        """Deletes the entry at key
        
        :param key: The key of the subject to be deleted
        :type key: str
        :return: whether or not item is deleted
        :rtype: bool
        """
        return super().__delitem__(key.lower())

    def __contains__(self, item: str) -> bool:
        """Checks if the dictionary contains the item

        :param item: key of the item we're looking for
        :type item: str
        :return: whether or not item is in the dictionary keys
        :rtype: bool
        """
        if not isinstance(item, str):
            raise TypeError(
                f"{self.__class__}'s keys can only contain type({str})")
        return item.lower() in self.keys()