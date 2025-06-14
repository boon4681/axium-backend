import sys_setup
import axium

axium.register.load_nodes()
func = getattr(axium.register.NODE_CLASS_MAPPINGS['sum'], 'sum')
print(func([0, 1]))
