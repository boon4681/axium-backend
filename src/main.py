import sys_setup # setup things
import axium

axium.register.load_nodes()

axium.register
func = getattr(axium.register.NODE_CLASS_MAPPINGS['sum'], 'sum')
print(func([0, 1]))
axium.register.get_node('sum')