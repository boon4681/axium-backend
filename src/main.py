import sys_setup  # setup things
import axium

axium.register.load_nodes()

axium.register
node_cls = axium.register.get_node('sum')
func = getattr(node_cls, node_cls.__FUNCTION__)
print(func([5, 1]))


node_cls = axium.register.get_node("abs")
func = getattr(node_cls, node_cls.__FUNCTION__)
print(func([-5]))

node_cls = axium.register.get_node("n")
print(node_cls)