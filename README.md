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

**2025/04/19** @2.0.5:
* patched new frontend input mechanism for dynamic inputs
* reduced requirements
* removed old vector conversions waiting for new frontend mechanism

**2025/04/17** @2.0.4:
* fixed bug in resize_matte `MODE` that would fail when the matte was smaller than the input image
* migrated to image_crop functions to cozy_comfyui

**2025/04/15** @2.0.3:
* migrated to image_stack in cozy_comfyui

**2025/04/14** @2.0.2:
* migrated out old routes to cozy_comfyui

**2025/04/14** @2.0.1:
* numpy version set for < 2.0.0
* core supports switched to [cozy_comfyui](https://github.com/cozy-comfyui/cozy_comfyui)

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


**2025/03/28** @1.7.46:
* ~~updated requirements for numpy to only be >=1.26.4~~
* removed to keep numba working with numpy < 2.0

**2025/03/25** @1.7.45:
* no work around comfyui not doing type conversion past version 1.10.14

**2025/03/18** @1.7.42:
* updated numby jit to ignore python objects
* aligned float("NaN") constructs

**2025/03/18** @1.7.40:
* allow vectors to be inline inputs + widget cause waiting for ComfyUI team is painful
* changed precision default to 3
* merged new action trigger from ComfyUI team

**2025/03/06** @1.7.34:
* prep for Purz stream
* removed security scanner failures for in-line http links
* auto-size masks during mask add
* fix some limits on transform node
* better constraints for UNARY and BINARY OP nodes
* fix inversion in pixel merge
* defaults for regex colorizer entries
* fall through for constant node input

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
