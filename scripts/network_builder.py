import torch.nn
import collections 

# Credit : https://gist.github.com/ferrine/89d739e80712f5549e44b2c2435979ef

class NetworkBuilder(object):
    def __init__(self, *namespaces):

        self._namespace = collections.ChainMap(*namespaces)


    def __call__(self, name, *args, **kwargs):
        try:
            return self._namespace[name](*args, **kwargs)
        except Exception as e:
            raise e.__class__(str(e), name, args, kwargs) from e

    def add_namespace(self, namespace, index=-1):
        if index >= 0:
            namespaces = self._namespace.maps
            namespaces.insert(index, namespace)
            self._namespace = collections.ChainMap(*namespaces)
        else:
            self._namespace = self._namespace.new_child(namespace)


def build_network(config, builder=NetworkBuilder(torch.nn.__dict__)):
    """
    Function to build the network

    Parameters:
    -----------
    config : dict
        Represents the configuration data (data which is obtained from the configuration file)

    builder : class, default : NetworkBuilder

    Returns:
    --------
    layers : array
        Represents the complete network that was decided
    
    """

    arch = config.get('architecture', None)
    assert type(arch) == dict, "Sorry you didn't explicitly define an architecture in the yaml file."

    layers = list()
    for block in arch:
        for l in arch[block].items():
            name, kwargs = l
            
            if kwargs is None:
                kwargs = {}

            args = kwargs.pop("args", [])
            layers.append(builder(name, *args, **kwargs))
   
    return layers