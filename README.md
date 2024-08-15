<h2><p align="center">THIS ENTIRE PROJECT IS DONATIONWARE.<br>PLEASE FEEL FREE TO CONTRIBUTE IN ANYWAY YOU THINK YOU CAN</p></h2>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Amorano/Jovimetrix-examples/blob/master/res/logo-jovimetrix.png">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/Amorano/Jovimetrix-examples/blob/master/res/logo-jovimetrix-light.png">
  <img alt="ComfyUI Nodes for procedural masking, live composition and video manipulation">
</picture>

<h3><p align="center">
<a href="https://github.com/comfyanonymous/ComfyUI">COMFYUI</a> Nodes for procedural masking, live composition and video manipulation
</p></h3>

<!---------------------------------------------------------------------------->

# QUICK REFERENCE

* [INSTALLATION](#installation)
* * [ENVIRONMENT](#environment-variables)
* [GLSL SHADERS](#glsl-shaders)
* * [CUSTOM SHADERS](#custom-shaders)
* [WORKFLOWS](#workflow-examples)
* [YT VIDEOS](#video-tutorials)
* [NODE REFERENCE](#node-reference)
* [COMMUNITY](#community)
* [ACKNOWLEDGEMENTS](#acknowledgements)

<!---------------------------------------------------------------------------->

# INSTALLATION

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

## ENVIRONMENT VARIABLES

There are a number of environment variables that can be used to affect how `Jovimetrix` itself functions within the ComfyUI environment:

CONTROL | VARIABLE(S)
---|---
[LOGGER](#logger) | JOV_LOG_LEVEL
[IGNORING NODES](#ignore-nodes) | JOV_IGNORE_NODE
[DEVICE SCANNER](#device-scan) | JOV_SCAN_DEVICES
[HELP SYSTEM](#help-system) | JOV_DOC
[GLSL SHADERS](#glsl-shaders) | JOV_GLSL
[GIFSKI EXPORT](#gifski-export) | JOV_GIFSKI
[SPOUT](#spout) | JOV_SPOUT

### LOGGER

The logger can be controlled via the JOV_LOG_LEVEL variable. It can be set to one of the following, by name or value:

* TRACE (5)
* DEBUG (10)
* INFO (20)
* SUCCESS (25)
* WARNING (30)
* ERROR (40)
* CRITICAL (50)

The default is WARNING (30); i.e.:

`SET JOV_LOG_LEVEL=WARNING`

### IGNORE NODES

Because there are a number of nodes that have overlapping functionality with other node packages, I have provided a mechanism to ignore loading of specific nodes.

If you create a file called `ignore.txt` inside the Jovimetrix root folder \(`.../ComfyUI/custom_nodes/Jovimetrix`\), it will skip loading any nodes included.

#### CUSTOM IGNORE FILE

If the `JOV_IGNORE_NODE` environment variable points to a valid text file, it will parse the file rows for `<Node Class>` names and attempt to skip loading those specific node class(es) at initialization.

This should be reflected in your ComfyUI log. Verify with the log any nodes ignored this way.

#### USAGE

Each entry should be on a separate line using the full node class name (the default name of the node). For example, in ignore.txt:

`CONSTANT (JOV) 🟪`

Will ignore the Constant node for use in ComfyUI.

This will *NOT* prevent the module from loading the imports, but this can help reduce your visual space while working within ComfyUI if you do not require looking at an additional 60+ nodes.


### SYSTEM DEVICE SCAN

Allows the system to auto-scan for any devices, so that it can populate the device list in the Stream Reader Node.

The [STREAM READER📺](https://github.com/Amorano/Jovimetrix/wiki/DEVICE#-stream-reader) is able to load media from local media, web media dna also system devices such as (virtual) web cameras and monitors. By default, the scan for web cameras is off.

If you wish to engage the auto-scan on ComfyUI start-up, set the JOV_SCAN_DEVICES variable to 1 or True.

`SET JOV_SCAN_DEVICES=1`

### HELP SYSTEM

The main help system is made possible by [Mel Massadian](https://github.com/melMass). It is located on the top right of each node (?). This will present a window which is loaded from the [main help repository for Jovimetrix](https://github.com/Amorano/Jovimetrix-examples/)

You can build all the help locally by loading Comfy and surfing to the local URL:

`http://127.0.0.1:8188/jovimetrix/doc`

This will build all the help stub files (.md) inside the main Jovimetrix folder under a folder named `_md`

If you wish to re-direct those outputs, you can set the ENV variable `JOV_DOC`.

For example:
`SET JOV_DOC=C:/dev/jvx/help`

You can also use the token: {name} in the path and it will be replaced with the name of the node, in case you wish to further categorize the output:

`SET JOV_DOC=C:/dev/jvx/help/{name}`

### GIFSKI EXPORT

If you have [GIFSKI](https://gif.ski/) installed you can enable the option for the Export Node to use GIFSKI when outputting frames.

You will need to add an environment var so it knows you have it installed and where:

`SET JOV_GIFSKI=[path to gifski]`

Once set the GIFSKI option should appear in the Export Node drop down list of output target formats.

### SPOUT
(WINDOWS ONLY)

If you are on Linux or Mac, `Spout` will not be installed from the requirements.txt.

By default, [Spout](https://leadedge.github.io/), a system for GPU accelerated sharing
of graphics between applications, is on for all platforms.

If you are on Mac or Linux, this will only amount to a message at startup about Spout not being used. When Spout is not found, the SpoutWriter node will not showup. In addition, the StreamReader node will not have Spout as an option from which to read stream data.

If you want to fully turn off the initial startup attempt to import Spout, you can use the environment variable:

`SET JOV_SPOUT=0`

# GLSL SHADERS

There are two mechanisms in Jovimetrix that allow the creation of GLSL shaders to support image creation or manipulation at ComfyUI runtime.

The main `GLSL (JOV) 🍩` node allows for the manipulating of shaders in the UI of ComfyUI. Once the shader is established, it can be left as is within the network and will produce the same content output as if it was a `Dynamic GLSL` node.

The secondary way is via the `Dynamic GLSL` system. This will search a filepath for shader files and pre-load them as nodes for ComfyUI. They will register as normal nodes and work in any API calls.

The benefit of the dynamic nodes are the reduced footprint of the node itself. Since the Dynamic nodes load their scripts statically, the node only contains the inputs and widgets for that specific script.

![`GLSL (JOV) 🍩` vs `Dynamic Node`](res/wiki/glsl_custom.png)
<p align="center">example of `GLSL (JOV) 🍩` footprint vs `Dynamic Node` footprint</p>

## DYNAMIC SHADERS

### CORE

All shaders have two parts: the vertex shader (.vert) and the fragment shader (.frag).

The default location for the included shaders is:

`<ComfyUI>/custom_nodes/Jovimetrix/res/glsl`

The basic shaders for each of these programs is included in the default location and are named:

`_.frag` and `_.vert`

There are several other shaders included and when they are used in ComfyUI they come denoted with a 🧙🏽 wizard icon at the end of their name.

These shader examples can be used to help jump-start custom shader work.

### VERTEX SHADER

```
#version 330 core

precision highp float;

void main()
{
    vec2 verts[3] = vec2[](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));
    gl_Position = vec4(verts[gl_VertexID], 0, 1);
}
```

The default vertex shader is simply a quad with the UV mapped from 0..1.

### CUSTOM SHADERS

You are able to add your own shaders such that they compile into nodes at ComfyUI load time. Custom shaders that are local to your machine will have a 🧙🏽‍♀️ wizard icon at the end of their name. The default location for local shaders is to search a folder in the root of Jovimetrix:

`<ComfyUI>/custom_nodes/Jovimetrix/glsl`

If you want to change the search location, you can set the environment variable:

`SET JOV_GLSL=<location to glsl shader files>`

### HEADER

```
uniform vec3    iResolution;
uniform float   iTime;
uniform float   iFrameRate;
uniform int     iFrame;

#define texture2D texture
```

All the shaders first have a header file that contains some pre-set variables for shader program usage. These are mostly mirrored from ShaderToy variables, with a few changes:

NAME | TYPE | USAGE
---|---|---
iResolution | vec3 | Dimensions of the GL canvas
iTime | float | current time in shader's lifetime
iFrameRate | float | the desired FPS
iFrame | int | the current frame based on the `iTime` and `iFrameRate`

### ENTRY POINT

The fragment shader's entry point is defined as:

```
void mainImage(out vec4 fragColor, vec2 fragCoord)
```

such that setting fragColor will output the final RGBA value for the pixel.

### SHADER META

Shaders are 100% fully GLSL compatible; however, there can be additional information (meta data) added to the shaders via comments. This expands the usefulness of the shaders since they can be pre-parsed and turned into nodes made at runtime (Dynamic Nodes).

```
// name: GRAYSCALE
// desc: Convert input to grayscale
// category: COLOR
```

The meta data breakdown of this shader header:

KEY | USAGE | EXPLANATION
---|---|---
name | GRAYSCALE | title of the node with an added 🧙🏽 for internal and 🧙🏽‍♀️ for custom nodes
desc | Convert input to grayscale | text that shows up for preview nodes and the Jovimetrix help panel
category | COLOR | ComfyUI menu placement. Added to the end of `JOVIMETRIX 🔺🟩🔵/GLSL`

`UNIFORM fields` also have metadata about usage(clipping for number fields) and their tooltips:

```
// default grayscale using NTSC conversion weights
uniform sampler2D image; | MASK, RGB or RGBA
uniform vec3 convert; // 0.299, 0.587, 0.114; 0; 1; 0.01 | Scalar for each channel
```

`<default value> ; <minimum> ; <maximum>; <step> | <tooltip>`

For the convert uniform this means a vector3 field with a default value of `<0.299, 0.587, 0.114>` clipped in the range `0..1` with a `0.01` step when the user interacts with the mouse and the tooltip will read: `Scalar for each channel`

If you need to omit fields, like a minimum, just place the token separator (;) by itself, for example:

uniform float num; // 0.5; ; 10

This would clip the upper-bound to 10 and allow the number to go to -system maximum.


# [NODE REFERENCE](https://github.com/Amorano/Jovimetrix/wiki)

[CREATE](https://github.com/Amorano/Jovimetrix/wiki/CREATE#create) | &nbsp;
---|---
[CONSTANT 🟪](https://github.com/Amorano/Jovimetrix/wiki/CREATE#-constant)|Create a single RGBA block of color. Useful for masks, overlays and general filtering.
[SHAPE GEN ✨](https://github.com/Amorano/Jovimetrix/wiki/CREATE#-shape-generator)|Generate polyhedra for masking or texture work.
[TEXT GEN 📝](https://github.com/Amorano/Jovimetrix/wiki/CREATE#-text-generator)|Uses any system font with auto-fit or manual placement.
[STEREOGRAM 📻](https://github.com/Amorano/Jovimetrix/wiki/CREATE#-stereogram)|Make a magic eye stereograms.
[STEREOSCOPIC 🕶️](https://github.com/Amorano/Jovimetrix/wiki/CREATE#-stereoscopic)|Simulate depth perception in images by generating stereoscopic views.
[GLSL 🍩](https://github.com/Amorano/Jovimetrix/wiki/CREATE#-glsl)|GLSL Shader support
[GRAPH WAVE▶ ılıılı](https://github.com/Amorano/Jovimetrix/wiki/CREATE#-graph-wave)|Import and display audio linear waveform data.
<img width=225/>|<img width=800/>

[COMPOSE](https://github.com/Amorano/Jovimetrix/wiki/COMPOSE) | &nbsp;
---|---
[ADJUST 🕸️](https://github.com/Amorano/Jovimetrix/wiki/COMPOSE#-adjust)|Blur, Sharpen, Emboss, Levels, HSV, Edge detection.
[BLEND ⚗️](https://github.com/Amorano/Jovimetrix/wiki/COMPOSE#-blend)|Applies selected operation to 2 inputs with optional mask using a linear blend (alpha)
[COLOR BLIND 👁‍🗨](https://github.com/Amorano/Jovimetrix/wiki/COMPOSE#-color-blind)|Simulate color blindness effects on images.
[COLOR MATCH 💞](https://github.com/Amorano/Jovimetrix/wiki/COMPOSE#-color-match)|Project the colors of one image  onto another or use a pre-defined color target.
[COLOR THEORY 🛞](https://github.com/Amorano/Jovimetrix/wiki/COMPOSE#-color-theory)|Generate Complimentary, Triadic and Tetradic color sets
[CROP ✂️](https://github.com/Amorano/Jovimetrix/wiki/COMPOSE#-crop)|Clip away sections of an image and backfill with optional color matte
[FILTER MASK 🤿](https://github.com/Amorano/Jovimetrix/wiki/COMPOSE#-filter-mask)|Create masks based on specific color ranges within an image.
[FLATTEN ⬇️](https://github.com/Amorano/Jovimetrix/wiki/COMPOSE#-flatten)|Combine multiple input images into a single image by summing their pixel values
[PIXEL MERGE 🫂](https://github.com/Amorano/Jovimetrix/wiki/COMPOSE#-pixel-merge)|Combine 3 or 4 inputs into a single image
[PIXEL SPLIT 💔](https://github.com/Amorano/Jovimetrix/wiki/COMPOSE#-pixel-split)|Splits images into constituent R, G and B and A channels
[PIXEL SWAP 🔃](https://github.com/Amorano/Jovimetrix/wiki/COMPOSE#-pixel-swap)|Swap pixel values between two input images based on specified channel swizzle operations
[STACK ➕](https://github.com/Amorano/Jovimetrix/wiki/COMPOSE#-stack)|Union multiple latents horizontal, vertical or in a grid
[THRESHOLD 📉](https://github.com/Amorano/Jovimetrix/wiki/COMPOSE#-threshold)|Clip an input based on a mid point value.
[TRANSFORM 🏝️](https://github.com/Amorano/Jovimetrix/wiki/COMPOSE#-transform)|Translate, Rotate, Scale, Tile, Mirror, Re-project and invert an input
<img width=225/>|<img width=800/>

[CALCULATE](https://github.com/Amorano/Jovimetrix/wiki/CALCULATE) | &nbsp;
---|---
[COMPARISON 🕵🏽](https://github.com/Amorano/Jovimetrix/wiki/CALCULATE#-comparison)|Compare two inputs: A=B, A!=B, A>B, A>=B, A<B, A<=B
[DELAY ✋🏽](https://github.com/Amorano/Jovimetrix/wiki/CALCULATE#-delay)|Delay traffic. Electrons on the data bus go round.
[LERP 🔰](https://github.com/Amorano/Jovimetrix/wiki/CALCULATE#-lerp)|Interpolate between two values with or without a smoothing
[OP UNARY 🎲](https://github.com/Amorano/Jovimetrix/wiki/CALCULATE#-calc-op-unary)|Perform a Unary Operation on an input.
[OP BINARY 🌟](https://github.com/Amorano/Jovimetrix/wiki/CALCULATE#-calc-op-binary)|Perform a Binary Operation on two inputs.
[TICK ⏱](https://github.com/Amorano/Jovimetrix/wiki/CALCULATE#-tick)|Periodic pulse exporting normalized, delta since last pulse and count.
[VALUE #️⃣](https://github.com/Amorano/Jovimetrix/wiki/CALCULATE-value)|Create a value for most types; also universal constants.
[WAVE GEN 🌊](https://github.com/Amorano/Jovimetrix/wiki/CALCULATE#-wave_generator)|Periodic and Non-Periodic Sinosodials.
<img width=225/>|<img width=800/>

[DEVICE](https://github.com/Amorano/Jovimetrix/wiki/DEVICE) | &nbsp;
---|---
[MIDI FILTER ✳️](https://github.com/Amorano/Jovimetrix/wiki/DEVICE#%EF%B8%8F-midi-filter)|Filter MIDI messages by channel, message type or value.
[MIDI FILTER EZ ❇️](https://github.com/Amorano/Jovimetrix/wiki/DEVICE#%EF%B8%8F-midi-filter-ez)|Filter MIDI messages by channel, message type or value.
[MIDI MESSAGE 🎛️](https://github.com/Amorano/Jovimetrix/wiki/DEVICE#%EF%B8%8F-midi-message)|Expands a MIDI message into its values.
[MIDI READER 🎹](https://github.com/Amorano/Jovimetrix/wiki/DEVICE#-midi-reader)|Capture MIDI devices and pass the data into Comfy.
[STREAM READER 📺](https://github.com/Amorano/Jovimetrix/wiki/DEVICE#-stream-reader)|Connect system media devices and remote streams into ComfyUI workflows.
[STREAM WRITER 🎞️](https://github.com/Amorano/Jovimetrix/wiki/DEVICE#%EF%B8%8F-stream-writer)|Broadcast ComfyUI Node outputs to custom webserver endpoint.
<img width=225/>|<img width=800/>

[UTILITY](https://github.com/Amorano/Jovimetrix/wiki/UTILITY) | &nbsp;
---|---
[AKASHIC 📓](https://github.com/Amorano/Jovimetrix/wiki/UTILITY#-akashic)|Display the top level attributes of an output
[EXPORT 📽](https://github.com/Amorano/Jovimetrix/wiki/UTILITY#-export)|Take your frames out static or animated (GIF)
[GRAPH 📈](https://github.com/Amorano/Jovimetrix/wiki/UTILITY#-value-graph)|Graphs historical execution run values
[QUEUE 🗃](https://github.com/Amorano/Jovimetrix/wiki/UTILITY#-queue)|Cycle lists of images files or strings for node inputs.
[ROUTE 🚌](https://github.com/Amorano/Jovimetrix/wiki/UTILITY#-route)|Pass all data because the default is broken on connection
[SAVE OUTPUT 💾](https://github.com/Amorano/Jovimetrix/wiki/UTILITY#-save-output)|Select an item from a user explicit list of inputs.
<img width=225/>|<img width=800/>

<!---------------------------------------------------------------------------->

# EXAMPLES

## WORKFLOW EXAMPLES

TBD

## VIDEO TUTORIALS

[![YouTube](./res/wiki/YouTube.svg)](https://www.youtube.com/channel/UCseaPIn-a2ji3LzVmnEF0Xw)

<!---------------------------------------------------------------------------->

# COMMUNITY

Everything here is made because I wanted to make it.
Everything you are looking for here that you cant find doesn't exist because I didn't make it.
If you feel like helping with text or code contributions, please pull and send me any PRs.

## CONTRIBUTIONS

Feel free to contribute to this project by reporting issues or suggesting improvements. Open an issue or submit a pull request on the GitHub repository.

## DONATIONS

[![If you feel like donating money resource instead, you can always use my ko-fi ❤️](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/alexandermorano)

## DISCORD
There are a few places you can chat about Jovimetrix nodes.

Directly on the `#jovimetrix` channel at the Banodoco discord:
[![](https://dcbadge.vercel.app/api/server/fbpScsxF4f?style=flat-square)](https://discord.gg/fbpScsxF4f)

On Purz discord (Creative Exploration):
[![](https://dcbadge.vercel.app/api/server/AxjQCRCnxn?style=flat-square)](https://discord.gg/AxjQCRCnxn)

## ACKNOWLEDGEMENTS
Thank you to all the creators and developers who make available their talent everyday to help everyone else.

## PROJECT SOURCES

This project uses code concepts from the following projects:

PROJECT | URL | LICENSE
---|---|---
MTB Nodes project | https://github.com/melMass/comfy_mtb | MIT
ComfyUI-Custom-Scripts project | https://github.com/pythongosssss/ComfyUI-Custom-Scripts | MIT
WAS Node Suite project | https://github.com/WASasquatch/was-node-suite-comfyui | MIT
rgthree-comfy project | https://github.com/rgthree/rgthree-comfy | MIT
FizzNodes project |  https://github.com/FizzleDorf/ComfyUI_FizzNodes | MIT

## COMFY DEVELOPERS & NODE MAKERS & COMMUNITY BUILDERS & ADDITIONAL THANKS!

Everyone mentioned here has been tireless in helping me, at some point, construct all the material in Jovimetrix.

THANK | YOU! | 💕
---|---|---
[comfy](https://github.com/comfyanonymous)|[Chris Goringe](https://github.com/chrisgoringe)|[Purz](https://github.com/purzbeats)
[pythongosssss](https://github.com/pythongosssss)|[melmass](https://github.com/melMass)|[Fizzledorf](https://github.com/Fizzledorf)
[Dr. Lt. Data](https://github.com/ltdrdata)|[Trung0246](https://github.com/Trung0246)|[Fannovel16](https://github.com/Fannovel16)
[Kijai](https://github.com/Kijai)|[WASasquatch](https://github.com/WASasquatch)|[MatisseTec](https://github.com/MatissesProjects)
[rgthree](https://github.com/rgthree)|[Suzue1](https://github.com/Suzie1)
<img width=250/>|<img width=250/>|<img width=250/>

<!---------------------------------------------------------------------------->

# WHY BUILD THESE NODES?

There are many ways to do composition and it is apparent that is a large portion of what Computer Vision - aka contemporaneous "A.I" - is invented to resolve.

While diffusers and latent hallucinations can make many amazing things at the moment, there is still a need to resolve final "frames" in something else, like one of the many free tools:
* [Krita](https://krita.org/en/) (2D)
* [Calvary](https://cavalry.scenegroup.co/) (motion graphics)
* [Davinci Resolve](https://www.blackmagicdesign.com/products/davinciresolve) (movie editing)

The main goal of Jovimetrix is to supplement those external workflows before the need to use them.

## TARGETS

* animation / motion graphics
* traditional image blending
* support masks as an image channel
* improved UX
** custom node colorization
** node favorites

<!---------------------------------------------------------------------------->

# LICENSE

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.