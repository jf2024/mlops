"""below is for data flow part"""
# from metaflow import FlowSpec, step

# class Counterflow(FlowSpec):
#     @step
#     def start(self):
#         self.count = 0
#         self.next(self.add)

"""below is for parameter part"""
from metaflow import FlowSpec, step, Parameter

class Counterflow(FlowSpec):

    begin_count = Parameter('ct', default = 20, type = int, required = True)

    @step
    def start(self):
        self.count = self.begin_count
        self.next(self.add)

    @step
    def add(self):
        print("The count is", self.count, "before incrementing")
        self.count += 1
        self.next(self.end)

    @step
    def end(self):
        self.count += 1
        print("Final count is", self.count)

if __name__ == '__main__':
    Counterflow()