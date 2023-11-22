> [!CAUTION]
> <h3><p align="center">⚠️ SUBJECT TO CHANGE PRIOR TO VERSION 1.0. USE AT YOUR OWN RISK ⚠️</p></h3>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="res/logo-jovimetrix.png">
  <source media="(prefers-color-scheme: light)" srcset="res/logo-jovimetrix-light.png">
  <img alt="ComfyUI Nodes for procedural masking, live composition and video manipulation">
</picture>

<h3><p align="center">
<a href="https://github.com/comfyanonymous/ComfyUI">COMFYUI</a> Nodes for procedural masking, live composition and video manipulation
</p></h3>

<!---------------------------------------------------------------------------->

# INSTALLATION

If you have [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) installed you can search for Jovimetrix and install from the manager's database.

## MANUAL INSTALL
To manually install, clone the repository into your ComfyUI custom_nodes directory. You can clone the repository with the command:
```
git clone https://github.com/Amorano/Jovimetrix.git
```
You can then install the requirements by using the command:
```
.\python_embed\python.exe -s -m pip install -r requirements.txt
```
If you are using a <code>virtual environment</code> (<code><i>venv</i></code>), make sure it is activated before installation. Then install the requirements with the command:
```
pip install -r requirements.txt
```

<!---------------------------------------------------------------------------->

# NODE REFERENCE

<img
    style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
    src="flow/overview.png"
    alt="GENERAL NODE OVERVIEW">
</img>

<b><code>COPY</code> [EXAMPLE IMAGES](https://github.com/Amorano/Jovimetrix/tree/main/flow) <code>INTO COMFYUI TO LOAD SHOWN WORKFLOW</code></b>

<details>
  <summary><b>CAPTURE</b></summary>
  <ul>
    <li>Connect system devices directly into ComfyUI workflows</li>

NODE | OVERVIEW | COMFY UI
---|---|---
📷 WEBCAM|Connect system webcams directly into ComfyUI workflows|![CAPTURE](flow/node_webcam.png "Connect system webcams directly into ComfyUI workflows")
 </ul>
  <ul>
    <details>
      <summary><b>WEBCAM EXAMPLE</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="flow/node_webcam-simple.png"
        alt="Simple webcam capture setup">
      </img>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="flow/node_webcam-mask.png"
        alt="Webcam with a simple shape mask for realtime overlay">
      </img>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="flow/node_webcam-multiple.png"
        alt="Complex filtering with webcams">
      </img>
    </details>
  </ul>
</details>

<!---------------------------------------------------------------------------->

<details>
  <summary><b>CREATE</b></summary>
  <ul>
    <li>Constant color node for when you need a block of color</li>
    <li>Nodes that generate images and masks in various polygonal shapes
      <ul>
        <li>Ellipse & Circle</li>
        <li>Rectangle & Square</li>
        <li>Polygon of 3 or more sides</li>
      </ul>
    </li>
    * Per Pixel Shader with input support. Slow but works until the ![GLSL]("GLSL Node") is available. Variables pre-defined for use in the loop include:
      <ul>
        <li><code>$x</code>, <code>$y</code>: Current image (x, y)</li>
        <li><code>$u</code>, <code>$v</code>: Normalized texuture coordinates [0..1]</li>
        <li><code>$w</code>, <code>$h</code>: Width and Height of the target output</li>
        <li><code>$ir</code>, <code>$ig</code>, <code>$ib</code>: Red, Green & Blue values for current image input (x, y)</li>
      </ul>
    </li>

NODE | OVERVIEW | COMFY UI
---|---|---
🟪 CONSTANT|Set a single RGB value. Useful for masks, overlays and general filtering|![CONSTANT](flow/node_constant.png "constant color node for when you need a block of color")
✨ SHAPE GENERATOR|Generate polyhedra for masking or texture work|![SHAPE](flow/node_shape.png "Nodes that generate images and masks in various polygonal shapes")
🔆 PER PIXEL SHADER|Per Pixel user function for each R, G, B channel|![PPSHADER](flow/node_pixel.png "Per Pixel shader")
🔆 PER PIXEL SHADER IMAGE|Per Pixel Shader with input support|![PPSHADER](flow/node_pixelimage.png "Per Pixel Shader with input support")
🍩 GLSL|GLSL Shader support|![GLSL](flow/node_glsl.png)
  </ul>
  <ul>
    <details>
      <summary><b>CONSTANT EXAMPLE</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="flow/node_constant.png"
        alt="CONSTANT">
      </img>
    </details>
    <details>
      <summary><b>SHAPE GENERATOR EXAMPLE</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="flow/node_shape.png"
        alt="GENERATOR">
      </img>
    </details>
    <details>
      <summary><b>PER PIXEL SHADER EXAMPLE</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="flow/node_pixel.png"
        alt="PPSHADER">
      </img>
    </details>
    <details>
      <summary><b>PER PIXEL + IMAGE SHADER EXAMPLE</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="flow/node_pixel-scaling.png" alt="PPIMAGESHADER">
      </img>
    </details>
      <details>
      <summary><b>GLSL EXAMPLE</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="flow/node_glsl.png" alt="GLSL">
      </img>
    </details>
  </ul>
</details>

<!---------------------------------------------------------------------------->

<details>
  <summary><b>TRANSFORM</b></summary>
  <ul>
    <li>Manipulate inputs with affine transformations</li>
    <li>Duplicate and Stack inputs</li>
  </ul>

NODE | OVERVIEW | COMFY UI
--|---|---
🌱 TRANSFORM|Translate, Rotate, and Scale an input. Options allow for CROP or WRAPing of the edges|![TRANSFORM](flow/node_transform.png "Translate, Rotate, and Scale an input. Options allow for CROP or WRAPing of the edges")
🔳 TILE|Repeat an input along the X, Y or XY at irregular intervals|![TILE](flow/node_tile.png "Repeat an input along the X, Y or XY at irregular intervals")
🔰 MIRROR|Flip an input across the X axis, the Y Axis or both, with independant centers|![MIRROR](flow/node_mirror.png "Flip an input across the X axis, the Y Axis or both, with independant centers")
🎇 EXTEND|Combine two inputs into a new image, side by side or top-down|![EXTEND](flow/node_extend.png "Combine two inputs into a new image, side by side or top-down")
🗺️ PROJECTION|Convert inputs to cartesian, polar, ?|![PROJECTION](flow/node_projection.png "Convert inputs to cartesian, polar, ?")

  <ul>
    <details>
      <summary><b>TRANSFORM EXAMPLE</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="flow/node_transform.png" alt="TRANSFORM">
      </img>
    </details>
    <details>
      <summary><b>TILE EXAMPLE</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="flow/node_tile.png" alt="TILE">
      </img>
    </details>
        <details>
      <summary><b>MIRROR EXAMPLE</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="flow/node_mirror.png" alt="MIRROR">
      </img>
    </details>
        <details>
      <summary><b>EXTEND EXAMPLE</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="flow/node_extend.png" alt="EXTEND">
      </img>
    </details>
        <details>
      <summary><b>PROJECTION EXAMPLE</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="flow/node_projection.png"
        alt="PROJECTION">
      </img>
    </details>
  </ul>
</details>

<!---------------------------------------------------------------------------->

<details>
  <summary><b>ADJUST</b></summary>
  <ul>
    <li>Manipulate lighting and color conditions of an input</li>
    <li>Apply matrix operations to images and masks</li>

NODE | OVERVIEW | COMFY UI
--|---|---
🌈 HSV|Adjust Hue, Saturation, Value, Gamma, Contrast and Exposure of an input|![HSV](flow/node_hsv.png "Adjust Hue, Saturation, Value, Gamma, Contrast and Exposure of an input")
🕸️ ADJUST|Find Edges, Blur, Sharpen and Emobss an input|![ADJUST](flow/node_adjust.png "Find Edges, Blur, Sharpen and Emobss an input")
📉 THRESHOLD|Clip an input based on a mid point value|![ADJUST](flow/node_threshold.png "Clip an input based on a mid point value")
  </ul>
  <ul>
    <details>
      <summary><b>HSV EXAMPLE</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="flow/node_hsv.png" alt="HSV">
      </img>
    </details>
    <details>
      <summary><b>ADJUST EXAMPLE</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="flow/node_adjust.png" alt="ADJUST">
      </img>
    </details>
        <details>
      <summary><b>THRESHOLD EXAMPLE</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="flow/node_threshold.png" alt="THRESHOLD">
      </img>
    </details>
  </ul>
</details>

<!---------------------------------------------------------------------------->

<details>
  <summary><b>COMPOSE</b></summary>
  <ul>
    <li>Composite Images and Masks together with optional alpha blending.</li>
    <li>Supports many operations; </li>

[feel free to submit new ideas for compositing modes](https://github.com/Amorano/Jovimetrix/issues)

NODE | OVERVIEW | COMFY UI
--|---|---
⚗️ BLEND MASK|Compose 2 inputs together an alpha mask and linear blend scalar|![BLENDMASK](flow/node_blendmask.png "Compose 2 inputs together an alpha mask and linear blend scalar")
⚗️ BLEND|Compose 2 inputs together with linear blend scalar|![BLEND](flow/node_blend.png "Compose 2 inputs together with linear blend scalar")
  </ul>
  <ul>
    <details>
      <summary><b>BLEND WITH MASK EXAMPLE</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="flow/node_blendmask-simple.png" alt="Compose 2 inputs together an alpha mask and linear blend scalar">
      </img>
    </details>
    <details>
      <summary><b>BLEND EXAMPLE</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="flow/node_blend-simple.png" alt="Compose 2 inputs together with linear blend scalar">
      </img>
    </details>
  </ul>
</details>

<!---------------------------------------------------------------------------->

<details>
  <summary><b>UTILITY</b></summary>
  <ul>

NODE | OVERVIEW | COMFY UI
--|---|---
🚌 ROUTE|Send outputs of any type to any input|![ROUTE](flow/node_route.png "Send outputs of any type to any input")
  </ul>
  <ul>
    <details>
      <summary><b>ROUTE EXAMPLE</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="flow/node_route.png" alt="Send outputs of any type to any input">
      </img>
    </details>
  </ul>
</details>

---

## TODO:
- [✔️] create readme
- [✔️] embed workflows in example images
- [⭕] specific single examples for all nodes
- [⭕] hook GLSL context in litegraph

