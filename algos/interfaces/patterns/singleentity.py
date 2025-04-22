class DuplicateError(Exception):
    pass


class SingleEntity:
    """
    SingleEntity allows only one instance of the class to be created with a 
    name. This restricts the namespace of all entities to be unique.

    All classes that extend SingleEntity must make their own _entities variable
    or they will share a common dictionary with SingleEntity
    """
    _entities = {}

    def __new__(cls, name: str, owner: object, return_instance: bool = False, *args,
                **kwargs):
        """
        Singleton pattern that either errors when a duplicate is 
        created, or it returns the existing instance. Singletons are
        purely created based on their name. 

        :param name: the name of the entity
        :type name: str
        :param return_instance: whether to raise an error if the entity exists
                                or simply return the entity, defaults to False
        :type return_instance: bool, optional
        :raises DuplicateError: if return_instance is false raises an error if the
                                entity already exists
        :return: The entity
        :rtype: SingleEntity
        """
        if name in cls._entities:
            #raise a duplicate error
            raise DuplicateError(f"{cls} already exists")
        else:
            #create the entity
            new = super().__new__(cls)
            #add entity to the entities dictionary
            cls._entities[name] = new
            return new


class HybridSingleEntity:
    """
    SingleEntity allows only one instance of the class to be created with a 
    name. This restricts the namespace of all entities to be unique.

    All classes that extend SingleEntity must make their own _entities variable
    or they will share a common dictionary with SingleEntity
    """
    _entities = {}

    def __new__(cls, name: str, owner: object, *args,
                **kwargs):
        """
        Hybrid singleton pattern that either errors when a duplicate is 
        created, or it returns the existing instance. Singletons are
        purely created based on their name. 

        :param name: the name of the entity
        :type name: str
        :param return_instance: whether to raise an error if the entity exists
                                or simply return the entity, defaults to False
        :type return_instance: bool, optional
        :raises DuplicateError: if return_instance is false raises an error if the
                                entity already exists
        :return: The entity
        :rtype: SingleEntity
        """
        key = (name, owner)
        if key in cls._entities:
            #entity has been created
            return cls._entities[key]
        else:
            #create the entity
            new = super().__new__(cls)
            #add entity to the entities dictionary
            cls._entities[key] = new
            return new