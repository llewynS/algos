from typing import List
import networkx as nx
import matplotlib.pyplot as plt

from ..abstractbaseclasses import AbstractComponent
from ..patterns import DuplicateError
from ..utility import TypedDict


class ComponentCollection:
    """
    A collection of Components. Acts similarly to a dictionary containing keys. 

    """

    def __init__(self, **kwargs):
        """
        Constructor for ComponentCollection
        """
        self.keys = list(kwargs.keys())
        self.dict = TypedDict(AbstractComponent)
        for key in self.keys:
            self.dict[key] = kwargs[key]
        self._master_component = None

    def run(self, **kwargs):
        """
        Starts and joins all components.

        Terminating conditions and all control is now the responsibility of the components. 
        """
        [self[k]._process_io() for k in self.keys]
        [self[k].start() for k in self.keys]
        AbstractComponent._all_instances_started.set()
        [self[k].join() for k in self.keys]

    def __setitem__(self, key: str, value: AbstractComponent):
        """
        Allows ComponentCollection to have values added 
        using the same syntax as a dict. It is also in a case
        invariant.

        :param key: The key for the value
        :type key: str
        :param value: The value for the key
        :type value: AbstractComponent
        :raises TypeError: key must be a string
        :raises TypeError: value must be a AbstractComponent
        """

        if key not in self.keys:
            self.keys.append(key)
            try:
                return self.dict.__setitem__(key.lower(), value)
            except TypeError as e:
                self.keys.pop(-1)
                raise e
        else:
            raise DuplicateError(f"{key} already in {self.keys}")

    def __getitem__(self, key: str) -> AbstractComponent:
        """
        Allows ComponentCollection to get values 
        using the same syntax as a dict. It is also
        case invariant. 
        
        :param key: The key of the component
        :type key: str
        :return: The component corresponding to the key
        :rtype: AbstractComponent
        """
        return self.dict.__getitem__(key.lower())

    def __repr__(self) -> str:
        """repr method of ComponentCollection
        
        :return: the repr of ComponentCollection
        :rtype: str
        """
        dictrepr = dict.__repr__(self.dict)
        return '%s(%s)' % (type(self).__name__, dictrepr)
    
    def drawme(self):
        # Create directed graph
        G = nx.DiGraph()
        for key in self.keys:
            multi_io = self[key].io
            for io in multi_io.io:
                for inp in io.inputs:
                    G.add_node(inp)
                for out in io.outputs:
                    G.add_node(out)
                for inp in io.inputs:
                    for out in io.outputs:
                        G.add_edge(inp, out, method=io.name)

        # Draw graph
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, font_weight='bold')
        edge_labels = nx.get_edge_attributes(G, 'method')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()
