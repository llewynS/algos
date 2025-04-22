
class RegisterMeta(type):
    """
    The register metaclass adds a dictionary to class that uses the metaclass. 
    Automates registering classes at class definition, which is typically done
    manually. (e.g gym when creating custom environments)

    Metaclasses that inherit from register need to redefine the _register 
    variable if separate registers should be maintained.

    This is typically useful for factory classes/functions so that the desired
    class can be specified with a string
    """

    def __new__(mcl, name, bases, class_dict):
        """
        Adds any new classes to _register when the class is defined and
        uses the register metaclass or any of it's children
        """
        #Hopefully this will remove the need to add a register to each new metaclass.
        if (not hasattr(mcl,"_register")):
            mcl._register = {}
        cls = type.__new__(mcl, name, bases, class_dict)
        if (not cls.__dict__.get("__abstractmethods__")):
            if cls.__name__ in mcl._register:
                raise ValueError(f"Class with this name: {cls.__name__} already exists in this registry")
            else:
                mcl._register[cls.__name__] = cls
        return cls