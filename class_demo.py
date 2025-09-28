class Demo:
    def __init__(self):
        self.a = 10  # Instance variable — stored in the object
        self.name = "Trading Workbench"
        self.display()  # Call display during initialization

    def display(self):
        print(f"Welcome to {self.name}!")

    def show_difference(self):
        a = 5  # Local variable — exists only inside this method
        print("Local variable a:", a)
        print("Instance variable self.a:", self.a)

# Create an object
obj = Demo()

# Call the method
obj.show_difference()
obj.display()