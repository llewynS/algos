import inspect
from abc import abstractmethod, ABCMeta

class InterfaceGeneratorMeta(ABCMeta):
    def __new__(mcl, name, bases, class_dict):
        """
        Dynamically create a class interface based on the method names defined in that class

        Incorporating a new method of defining IO so that it becomes extensible and generalisable. Name alteration may be needed.
        
        This is to be used with caution as ABCMeta recommends using update_abstractmethods decorator
        
        This could be done as a class factory if that is preferred. e.g:
        def metafactory(cls):
            assert(hasattr(cls, "method_names"))
            assert(type(cls)==ABCMeta)
            [setattr(cls, x, abstractmethod(lambda self: None)) for x in cls.method_names]
            return update_abstractmethods(cls)
        @metafactory
        class YourClass(ABC):
            method_names = ["method_a", "method_b"]
        """
        #First interface level
        if "abstract_methods" in class_dict:
            cls = super().__new__(mcl, name, bases, {**class_dict, \
                                                    **{method_name:abstractmethod(lambda self, **kwargs:None) \
                                                        for method_name in class_dict["abstract_methods"]}})
        else:
            cls = super().__new__(mcl, name, bases, class_dict)
        return cls
