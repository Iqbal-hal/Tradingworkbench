class Cuboid:
    def __init__(self):
        # Initialize dimensions
        self.length = 0
        self.breadth = 0
        self.height = 0

    def calculate_volume(self):
        volume = self.length * self.breadth * self.height
        print(f"Volume of the cuboid is: {volume}")


class SurfaceAreaCalculator(Cuboid):
    def __init__(self):
        super().__init__()  # Call Cuboid's constructor to initialize dimensions

    def calculate_surface_area(self):
        # Surface Area = 2(lb + bh + hl)
        area = 2 * (self.length * self.breadth +
                    self.breadth * self.height +
                    self.height * self.length)
        print(f"Surface area of the cuboid is: {area}")


# Create an object of the child class
cuboid_calc = SurfaceAreaCalculator()

# Ask user for dimensions
cuboid_calc.length = float(input("Enter length: "))
cuboid_calc.breadth = float(input("Enter breadth: "))
cuboid_calc.height = float(input("Enter height: "))

# Call both methods
cuboid_calc.calculate_volume()
cuboid_calc.calculate_surface_area()