import torch
class SharedVariableClass:
    shared_variable =  torch.zeros(1,100)

    def __init__(self):
        self.variable = SharedVariableClass.shared_variable
        self.variable.requires_grad_(True)

    def update_shared_variable(self, new_value):
        SharedVariableClass.shared_variable = new_value  # Update shared variable
        self.variable = SharedVariableClass.shared_variable  # Sync instance reference

variable = torch.zeros(1, 100)
class VariableClass:
    def __init__(self, variable):
        self.variable = variable
        self.variable.requires_grad_(True)

    def add(self):
        self.variable = self.variable + torch.ones(1, 100)


# obj = SharedVariableClass()
# # obj.variable = obj.variable + torch.ones(1, 100)  # This does NOT change SharedVariableClass.shared_variable
# obj.update_shared_variable(obj.variable + torch.ones(1, 100))
# print(SharedVariableClass.shared_variable)  # Output: 42
# print(obj.variable)

obj1 = VariableClass(variable)
obj1.add()
print(variable, obj1.variable)
obj2 = VariableClass(variable)
obj2.add()
print(variable, obj2.variable)