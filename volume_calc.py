class Cuboid:
    def __init__(self):
        # Initialize with default values
        self.length = 0
        self.breadth = 0
        self.height = 0

    def calculate_volume(self):
        volume = self.length * self.breadth * self.height
        print(f"Volume of the cuboid is: {volume}")

# Create an object
cuboid = Cuboid()

# Ask user for dimensions
cuboid.length = float(input("Enter length: "))
cuboid.breadth = float(input("Enter breadth: "))
cuboid.height = float(input("Enter height: "))

# Calculate and print volume
cuboid.calculate_volume()

