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
* GLSL shader support
* * `GLSL Node`  provides raw access to Vertex and Fragment shaders
* * `Dynamic GLSL` dynamically convert existing GLSL scripts file into ComfyUI nodes at runtime
* * Over 20+ Hand written GLSL nodes to speed up specific tasks better done on the GPU (10x speedup in most cases)
* `STREAM READER` node to capture monitor, webcam or url media
* `STREAM WRITER` node to export media to a HTTP/HTTPS server for OBS or other 3rd party streaming software
* `SPOUT` streaming support *WINDOWS ONLY*
* `MIDI READER` Captures MIDI messages from an external MIDI device or controller
* `MIDI MESSAGE` Processes MIDI messages received from an external MIDI controller or device
* `MIDI FILTER` (advanced filter) to select messages from MIDI streams and devices
* `MIDI FILTER EZ` simpler interface to filter single messages from MIDI streams and devices
* Full Text generation support using installed system fonts
* Basic parametric shape (Circle, Square, Polygon) generator
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
* Help System for *ALL NODES* that will auto-parse unknown knows for their type data and descriptions
* Colorization for *ALL NODES* using their own node settings, their node group or via regex pattern matching

### HELP SYSTEM

The built-in help system will dynamically parse any nodes found at the ComfyUI API endpoint:

`<comfyui_web_root>/object_info`

If those nodes have descriptions written in HTML or Markdown, they will be converted into HTML for presentation in the panel.

<div align="center">
<img src="https://github.com/user-attachments/assets/075f6f9b-b27b-4a6f-84da-a6db486446ff" alt="Clicking Nodes for Help" width="512"/>
</div>

### NODE COLORS

You can colorize nodes via their `title background`, `node body` or `title text`. This can be done to the node's `category` as well, so you can quickly color groups of nodes.

<div align="center">
<img src="https://github.com/user-attachments/assets/8de9561a-b231-4a49-a63a-4fd5fdba64d9" alt="node color panel" width="512"/>
</div>

## UPDATES

**2024/02/04** @1.6.5:
* adjusted new default color config to old defaults

**2024/02/04** @1.6.0:
* added colorizer option for `title text`
* settings deprecated and removed from ComfyUI settings panel
* colorizer setting migrated to ComfyUI per user settings
* automatic contrast option for non-colored nodes
* all tooltips migrated to ComfyUI core tooltips

**2024/02/02** @1.2.67:
* fixed `GLSL` nodes missing from parse due to category sort

**2024/02/01** @1.2.66:
* patched `MIDI FILTER EZ` to work for all filters not just the first found
* cleanup pyproject for registry
* proper logo for registry

**2024/02/01** @1.2.63:
* Fixed `MIDI FILTER` to parse all filters for trigger
* Better `MIDI FILTER` ranges:
* * Single numbers: "1, 2" (equals)
* * Closed ranges: "5-10" (between inclusive)
* * Open ranges: "-100" (less than or equal to 100)
* * Open ranges: "50-" (greater than or equal to 50)
* * 1, 5-10, 2
* * * would check == 1, == 2 and 5 <= x <= 10

**2024/01/30** @1.2.62:
* `QUEUE` nodes can do URL loading of images

**2024/01/08** @1.2.61:
* living in the stupidest timeline where case-sensitivty matters for URLS in chrome...
* fixed stylesheet reference in help

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
