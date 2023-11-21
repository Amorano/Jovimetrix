
<img
    style="display: block;
           margin-left: auto;
           margin-right: auto;
           width: 100%;"
    src="res/logo-jovimetrix.png"
    alt="JOVIMETRIX">
</img>

<div style="text-align: center;">
<a href="https://github.com/comfyanonymous/ComfyUI">COMFYUI</a> Nodes for procedural masking, live composition and video manipulation
<br>
ALL IMAGES CONTAIN THE EMBEDDED WORKFLOW DEPICTED
</div>

---

<div style="text-align: center;">GENERAL NODE OVERVIEW</div>
<img
    style="display: block;
           margin-left: auto;
           margin-right: auto;
           width: 100%;"
    src="flow/overview.png"
    alt="JOVIMETRIX">
</img>


WEBCAM
------
<div style="text-align: center;">GENERAL WEBCAM MANIPULATION</div>
<img
    style="display: block;
           margin-left: auto;
           margin-right: auto;
           width: 100%;"
    src="flow/webcam_manipulation.png"
    alt="JOVIMETRIX">
</img>

<div style="text-align: center;">GENERAL NODE OVERVIEW</div>
<img
    style="display: block;
           margin-left: auto;
           margin-right: auto;
           width: 100%;"
    src="flow/webcam+mask.png"
    alt="JOVIMETRIX">
</img>

<div style="text-align: center;">**WIP**</div>
<img
    style="display: block;
           margin-left: auto;
           margin-right: auto;
           width: 100%;"
    src="flow/webcam_face_detect.png"
    alt="JOVIMETRIX">
</img>


CREATION
--------

* 2D SHAPE
  * CIRCLE
  * SQUARE
  * RECTANGLE
  * ELLIPSE
  * POLYGON (3+ sides)

<img
    style="display: block;
           margin-left: auto;
           margin-right: auto;
           width: 100%;"
    src="flow/per_pixel_shader_scaling.png"
    alt="JOVIMETRIX">
</img>

* PER PIXEL SHADER. Two nodes, one with source Image support. Allows user function to generate a per pixel result into an image of (Width x Height). variables include:
  * $x, $y: Current image (x, y)
  * $u, $v: Normalized texuture coordinates [0..1]
  * $w, $h: WIDTH and HEIGHT of the target output
  * $ir, $ig, $ib: The RED, GREEN and BLUE values for the current image input ($x, $y).


* CONSTANT. Set a single RGB value. Useful for masks, overlays and general filtering.

MANIPULATION
--------------

* TRANSFORM. Translate, Rotate, and Scale an input. Options allow for CROP or WRAPing of the edges.
* MIRROR. Flip an input across the X axis, the Y Axis or both, with independant centers.
* TILE. Repeat an input along the X, Y or XY at irregular intervals
* EXTEND. Combine two inputs into a new image, side by side or top-down.

ADJUSTMENT
---------

* HSV. Tweak the Hue, Saturation and Value for an input.

* ADJUST
  * EMBOSS
  * FIND EDGES

  Take Radius:
    * BLUR
    * SHARPEN

  Take Scalar:
    * CONTRAST
    * GAMMA
    * EXPOSURE
    * INVERT

BLENDING
--------

* BLEND. Takes 2 inputs with an apha and performs a linear blend (alpha) between both inputs based on the selected operation. Operations include:
  * LINEAR INTEROPLATION
  * ADD
  * MINIMUM
  * MAXIUM
  * MULTIPLY
  * SOFT LIGHT
  * HARD LIGHT
  * OVERLAY
  * SCREEN
  * SUBTRACT
  * LOGICAL AND
  * LOGICAL OR
  * LOGICAL XOR
