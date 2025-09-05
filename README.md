<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Amorano/Jovimetrix-examples/blob/master/res/logo-jovimetrix.png">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/Amorano/Jovimetrix-examples/blob/master/res/logo-jovimetrix-light.png">
  <img alt="ComfyUI Nodes for procedural masking, live composition and video manipulation">
</picture>

<h2><div align="center">
<a href="https://github.com/comfyanonymous/ComfyUI">COMFYUI</a> Nodes for procedural masking, live composition and video manipulation
</div></h2>

<h3><div align="center">
JOVIMETRIX IS ONLY GUARANTEED TO SUPPORT <a href="https://github.com/comfyanonymous/ComfyUI">COMFYUI 0.1.3+</a> and <a href="https://github.com/Comfy-Org/ComfyUI_frontend">FRONTEND 1.2.40+</a><br>
IF YOU NEED AN OLDER VERSION, PLEASE DO NOT UPDATE.
</div></h3>

<h2><div align="center">

![KNIVES!](https://badgen.net/github/open-issues/amorano/jovimetrix)
![FORKS!](https://badgen.net/github/forks/amorano/jovimetrix)

</div></h2>

<!---------------------------------------------------------------------------->

# SPONSORSHIP

Please consider sponsoring me if you enjoy the results of my work, code or documentation or otherwise. A good way to keep code development open and free is through sponsorship.

<div align="center">

&nbsp;|&nbsp;|&nbsp;|&nbsp;
-|-|-|-
[![BE A GITHUB SPONSOR ❤️](https://img.shields.io/badge/sponsor-30363D?style=for-the-badge&logo=GitHub-Sponsors&logoColor=#EA4AAA)](https://github.com/sponsors/Amorano) | [![DIRECTLY SUPPORT ME VIA PAYPAL](https://img.shields.io/badge/PayPal-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://www.paypal.com/paypalme/onarom) | [![PATREON SUPPORTER](https://img.shields.io/badge/Patreon-F96854?style=for-the-badge&logo=patreon&logoColor=white)](https://www.patreon.com/joviex) | [![SUPPORT ME ON KO-FI!](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/alexandermorano)
</div>

## HIGHLIGHTS

* 30 function `BLEND` node -- subtract, multiply and overlay like the best
* Vector support for 2, 3, 4 size tuples of integer or float type
* Specific RGB/RGBA color vector support that provides a color picker
* All Image inputs support RGBA, RGB or pure MASK input
* Full Text generation support using installed system fonts
* Basic parametric shape (Circle, Square, Polygon) generator~~
* `COLOR BLIND` check support
* `COLOR MATCH` against existing images or create a custom LUT
* Generate `COLOR THEORY` spreads from an existing image
* `COLOR MEANS` to generate palettes for existing images to keep other images in the same tonal ranges
* `PIXEL SPLIT` separate the channels of an image to manipulate and `PIXEL MERGE` them back together
* `STACK` a series of images into a new single image vertically, horizontally or in a grid
* Or `FLATTEN` a batch of images into a single image with each image subsequently added on top (slap comp)
* `VALUE` Node has conversion support for all ComfyUI types and some 3rd party types (2DCoords, Mixlab Layers)
* `LERP` node to linear interpolate all ComfyUI and Jovimetrix value types
* Automatic conversion of Mixlab Layer types into Image types
* Generic `ARRAY` that can Merge, Split, Select, Slice or Randomize a list of ANY type
* `STRINGER` node to perform specific string manipulation operations: Split, Join, Replace, Slice.
* A `QUEUE` Node that supports recursing directories, filtering multiple file types and batch loading
* Use the `OP UNARY` and `OP BINARY` nodes to perform single and double type functions across all ComfyUI and Jovimetrix value types
* Manipulate vectors with the `SWIZZLE` node to swap their XYZW positions
* `DELAY` execution at certain parts in a workflow, with or without a timeout
* Generate curve data with the `TICK` and `WAVE GEN` nodes

<br>

<h1>AS OF VERSION 2.0.0, THESE NODES HAVE MIGRATED TO OTHER, SMALLER PACKAGES</h1>

Migrated to [Jovi_GLSL](https://github.com/Amorano/Jovi_GLSL)

~~* GLSL shader support~~
~~* * `GLSL Node`  provides raw access to Vertex and Fragment shaders~~
~~* * `Dynamic GLSL` dynamically convert existing GLSL scripts file into ComfyUI nodes at runtime~~
~~* * Over 20+ Hand written GLSL nodes to speed up specific tasks better done on the GPU (10x speedup in most cases)~~

Migrated to [Jovi_Capture](https://github.com/Amorano/Jovi_Capture)

~~* `STREAM READER` node to capture monitor, webcam or url media~~
~~* `STREAM WRITER` node to export media to a HTTP/HTTPS server for OBS or other 3rd party streaming software~~

Migrated to [Jovi_Spout](https://github.com/Amorano/Jovi_Spout)

~~* `SPOUT` streaming support *WINDOWS ONLY*~~

Migrated to [Jovi_MIDI](https://github.com/Amorano/Jovi_MIDI)

~~* `MIDI READER` Captures MIDI messages from an external MIDI device or controller~~
~~* `MIDI MESSAGE` Processes MIDI messages received from an external MIDI controller or device~~
~~* `MIDI FILTER` (advanced filter) to select messages from MIDI streams and devices~~
~~* `MIDI FILTER EZ` simpler interface to filter single messages from MIDI streams and devices~~

Migrated to [Jovi_Help](https://github.com/Amorano/Jovi_Help)

~~* Help System for *ALL NODES* that will auto-parse unknown knows for their type data and descriptions~~

Migrated to [Jovi_Colorizer](https://github.com/Amorano/Jovi_Colorizer)

~~* Colorization for *ALL NODES* using their own node settings, their node group or via regex pattern matching~~

## UPDATES

<h2>DO NOT UPDATE JOVIMETRIX PAST VERSION 1.7.48 IF YOU DONT WANT TO LOSE A BUNCH OF NODES</h2>

Nodes that have been removed are in various other packages now. You can install those specific packages to get the functionality back, but I have no way to migrate the actual connections -- you will need to do that manually. **

Nodes that have been migrated:

* ALL MIDI NODES:
* * MIDIMessageNode
* * MIDIReaderNode
* * MIDIFilterNode
* * MIDIFilterEZNode

[Migrated to Jovi_MIDI](https://github.com/Amorano/Jovi_MIDI)

* ALL STREAMING NODES:
* * StreamReaderNode
* * StreamWriterNode

[Migrated to Jovi_Capture](https://github.com/Amorano/Jovi_Capture)

* * SpoutWriterNode

[Migrated to Jovi_Spout](https://github.com/Amorano/Jovi_Spout)

* ALL GLSL NODES:
* * GLSL
* * GLSL BLEND LINEAR
* * GLSL COLOR CONVERSION
* * GLSL COLOR PALETTE
* * GLSL CONICAL GRADIENT
* * GLSL DIRECTIONAL WARP
* * GLSL FILTER RANGE
* * GLSL GRAYSCALE
* * GLSL HSV ADJUST
* * GLSL INVERT
* * GLSL NORMAL
* * GLSL NORMAL BLEND
* * GLSL POSTERIZE
* * GLSL TRANSFORM

[Migrated to Jovi_GLSL](https://github.com/Amorano/Jovi_GLSL)

**2025/09/04** @2.1.25:
* Auto-level for `LEVEL` node
* `HISTOGRAM` node
* new support for cozy_comfy (v3+ comfy node spec)

**2025/08/15** @2.1.23:
* fixed regression in `FLATTEN` node

**2025/08/12** @2.1.22:
* tick allows for float/int start

**2025/08/03** @2.1.21:
* fixed css for `DELAY` node
* delay node timer extended to 150+ days
* all tooltips checked to be TUPLE entries

**2025/07/31** @2.1.20:
* support for tensors in `OP UNARY` or `OP BINARY`

**2025/07/27** @2.1.19:
* added `BATCH TO LIST` node
* `VECTOR` node(s) default step changed to 0.1

**2025/07/13** @2.1.18:
* allow numpy>=1.25.0

**2025/07/07** @2.1.17:
* updated to cozy_comfyui 0.0.39

**2025/07/04** @2.1.16:
* Type hint updates

**2025/06/28** @2.1.15:
* `GRAPH NODE` updated to use new mechanism in cozy_comfyui 0.0.37 for list of list parse on dynamics

**2025/06/18** @2.1.14:
* fixed resize_matte mode to use full mask/alpha

**2025/06/18** @2.1.13:
* allow hex codes for vectors
* updated to cozy_comfyui 0.0.36

**2025/06/07** @2.1.11:
* cleaned up image_convert for grayscale/mask
* updated to cozy_comfyui 0.0.35

**2025/06/06** @2.1.10:
* updated to comfy_cozy 0.0.34
* default width and height to 1
* removed old debug string
* akashic try to parse unicode emoji strings

**2025/06/02** @2.1.9:
* fixed dynamic nodes that already start with inputs (dynamic input wouldnt show up)
* patched Queue node to work with new `COMBO` style of inputs

**2025/05/29** @2.1.8:
* updated to comfy_cozy 0.0.32

**2025/05/27** @2.1.7:
* re-ranged all FLOAT to their maximum representations
* clerical cleanup for JS callbacks
* added `SPLIT` node to break images into vertical or horizontal slices

**2025/05/25** @2.1.6:
* loosened restriction for python 3.11+ to allow for 3.10+
* * I make zero guarantee that will actually let 3.10 work and I will not support 3.10

**2025/05/16** @2.1.5:
* Full compatibility with [ComfyMath Vector](https://github.com/evanspearman/ComfyMath) nodes
* Masks can be inverted at inputs
* `EnumScaleInputMode` for `BLEND` node to adjust inputs prior to operation
* Allow images or mask inputs in `CONSTANT` node to fall through
* `VALUE` nodes return all items as list, not just >1
* Added explicit MASK option for `PIXEL SPLIT` node
* Split `ADJUST` node into `BLUR`, `EDGE`, `LIGHT`, `PIXEL`,
* Migrated most of image lib to cozy_comfyui
* widget_vector tweaked to disallow non-numerics
* widgetHookControl streamlined

**2025/05/08** @2.1.4:
* Support for NUMERICAL (bool, int, float, vecN) inputs on value inputs

**2025/05/08** @2.1.3:
* fixed for VEC* types using MIN/MAX

**2025/05/07** @2.1.2:
* `TICK` with normalization and new series generator

**2025/05/06** @2.1.1:
* fixed IS_CHANGED in graphnode
* updated `TICK SIMPLE` in situ of `TICK` to be inclusive of the end range
* migrated ease, normalization and wave functions to cozy_comfyui
* first pass preserving values in multi-type fields

**2025/05/05** @2.1.0:
* Cleaned up all node defaults
* Vector nodes aligned for list outputs
* Cleaned all emoji from input/output
* Clear all EnumConvertTypes and align with new comfy_cozy
* Lexicon defines come from Comfy_Cozy module
* `OP UNARY` fixed factorial
* Added fill array mode for `OP UNARY`
* removed `STEREOGRAM` and `STEROSCOPIC` -- they were designed poorly

**2025/05/01** @2.0.11:
* unified widget_vector.js
* new comfy_cozy support
* auto-convert all VEC*INT -> VEC* float types
* readability for node definitions

**2025/04/24** @2.0.10:
* `SHAPE NODE` fixed for transparency blends when using blurred masks

**2025/04/24** @2.0.9:
* removed inversion in pixel splitter

**2025/04/23** @2.0.8:
* categories aligned to new comfy-cozy support

**2025/04/19** @2.0.7:
* all JS messages fixed

**2025/04/19** @2.0.6:
* fixed reset message from JS

**2025/04/19** @2.0.5:
* patched new frontend input mechanism for dynamic inputs
* reduced requirements
* removed old vector conversions waiting for new frontend mechanism

**2025/04/17** @2.0.4:
* fixed bug in resize_matte `MODE` that would fail when the matte was smaller than the input image
* migrated to image_crop functions to cozy_comfyui

**2025/04/12** @2.0.0:
* REMOVED ALL STREAMING, MIDI and GLSL nodes for new packages, HELP System and Node Colorization system:

   [Jovi_Capture - Web camera, Monitor Capture, Window Capture](https://github.com/Amorano/Jovi_Capture)

   [Jovi_MIDI - MIDI capture and MIDI message parsing](https://github.com/Amorano/Jovi_MIDI)

   [Jovi_GLSL - GLSL Shaders](https://github.com/Amorano/Jovi_GLSL)

   [Jovi_Spout - SPOUT Streaming support](https://github.com/Amorano/Jovi_Spout)

   [Jovi_Colorizer - Node Colorization](https://github.com/Amorano/Jovi_Colorizer)

   [Jovi_Help - Node Help](https://github.com/Amorano/Jovi_Help)

* all nodes will accept `LIST` or `BATCH` and process as if all elements are in a list.
* patched constant node to work with `MATTE_RESIZE`
* patched import loader to work with old/new comfyui
* missing array web node partial
* removed array and no one even noticed.
* all inputs should be treated as a list even single elements []

<div align="center">
<img src="https://github.com/user-attachments/assets/8ed13e6a-218c-468a-a480-53ab55b04d21" alt="explicit vector node supports" width="640"/>
<img src="https://github.com/user-attachments/assets/4459855c-c4e6-4739-811e-a6c90aa5a90c" alt="TICK Node Batch Support Output" width="384"/>
</div>

# INSTALLATION

[Please see the wiki for advanced use of the environment variables used during startup](https://github.com/Amorano/Jovimetrix/wiki/B.-ASICS)

## COMFYUI MANAGER

If you have [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) installed, simply search for Jovimetrix and install from the manager's database.

## MANUAL INSTALL
Clone the repository into your ComfyUI custom_nodes directory. You can clone the repository with the command:
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
# WHERE TO FIND ME

You can find me on [![DISCORD](https://dcbadge.vercel.app/api/server/62TJaZ3Z5r?style=flat-square)](https://discord.gg/62TJaZ3Z5r).
