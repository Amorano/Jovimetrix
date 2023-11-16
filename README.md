# jovimetrix
My personal collection of custom ComfyUI nodes that have a nunber of useful functions.

CREATION
--------

* 2D Shape Node. Shapes include:
    Circle
    Square
    Rectangle
    Ellipse
    Polygon (3-100 sides)
* Per Pixel Shader. Allows user function to generate a per pixel result of the supplied Width x Height. It injects $x, $y, $tu, $tv, $w and $h as variables the user can use in defining said functions.
* Constant Node. Set a single RGB value. Useful for masks, overlays and general filtering.

TRANSFORMATION
--------------

* Transform. Translate, Rotate, Scale, Tile and Invert an Image. All options allow for CROP or WRAPing of the edges.
* Invert. Alpha blend an Image's inverted version. with the original.
* Mirror. Flip an Image across the X axis, the Y Axis or both, with independant centers.
* HSV Adjustment. Tweak the Hue, Saturation and Value for an Image.
* Lumen. Contrast, Gamma and Exposure controls.

FILTERING
---------

A single node with multiple operations:

* Blur
* Sharpen
* Emboss
* Find Edges

BLENDING
--------

* Blending Node. Takes 2 Image inputs and an apha and performs a linear blend (alpha) between both images based on the selected operations. Operations include:

    Linear Interoplation
    Add
    Minimum
    Maxium
    Multiply
    Soft Light
    Hard Light
    Overlay
    Screen
    Subtract
    Logical AND
    Logical OR
    Logical XOR

MAPPING
---------

* 1
* 2