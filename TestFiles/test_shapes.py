#!/usr/bin/env python3

from ClassFiles.ShapeGenerator import ShapeGenerator

shapes = ShapeGenerator(128, 128)
shapes.add_polygon(times=4)
shapes.add_ellipse(times=5)

shapes.add_noise()
shapes.image.save("images/dirty.BMP")

# shapes.image.show()