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

### LOGGER

The logger can be controlled via the JOV_LOG_LEVEL variable. It can be set to one of the following, by name or value:

* TRACE (5)
* DEBUG (10)
* INFO (20)
* SUCCESS (25)
* WARNING (30)
* ERROR (40)
* CRITICAL (50)

The default is WARNING (30); i.e. SET JOV_LOG_LEVEL=WARNING

### SYSTEM DEVICE SCAN

Allows the system to auto-scan for any devices, so that it can populate the device list in the Stream Reader Node.

The [STREAM READER📺](https://github.com/Amorano/Jovimetrix/wiki/DEVICE#-stream-reader) is able to load media from local media, web media dna also system devices such as (virtual) web cameras and monitors. By default, the scan for web cameras is off.

If you wish to engage the auto-scan on ComfyUI statrt-up, set the JOV_SCAN_DEVICES variable to 1 or True.

JOV_SCAN_DEVICES=1

### GIFSKI SUPPORT

If you have [GIFSKI](https://gif.ski/) installed you can enable the option for the Export Node to use GIFSKI when outputting frames.

You will need to add an environment var so it knows you have it installed and where:

set JOV_GIFSKI=[path to gifski]

Once set the GIFSKI option should appear in the Export Node drop down list of output target formats.

## FFMEPG

The audio nodes require FFMPEG. You can find the official [FFMPEG](https://ffmpeg.org "official FFMPEG binaries") here. Follow it's installation instructions for your specific operating system.

## PYAUDIO

By default, pyaudio is installed for all platforms; however, it may be nessicary to run a specific platform package manager to obtain all the correct platform dependencies. [You can refer to the non-complicated specific platform instructions for help.](https://people.csail.mit.edu/hubert/pyaudio).

In short:
* For MacOS you need the extra brew package of portaudio. (brew install portaudio)
* For Linux you need the extra apt package of python3-pyaudio. (sudo apt-get install python3-pyaudio)

## SPOUT (WINDOWS ONLY)

If you are on Linux or Mac, Spout will not be installed from the requirements.txt.

By default, [Spout](https://leadedge.github.io/), a system for GPU accelerated sharing
of graphics between applications, is on for all platforms.

If you are on Mac or Linux, this will only amount to a message at startup about Spout not being used. When Spout is not found, the SpoutWriter node will not showup. In addition, the StreamReader node will not have Spout as an option from which to read stream data.

If you want to fully turn off the initial startup attempt to import Spout, you can use the environment variable:

JOV_SPOUT=0

<!---------------------------------------------------------------------------->

# [NODE REFERENCE](https://github.com/Amorano/Jovimetrix/wiki)

[CREATE](https://github.com/Amorano/Jovimetrix/wiki/CREATE#create) | &nbsp;
---|---
[CONSTANT 🟪](https://github.com/Amorano/Jovimetrix/wiki/CREATE#-constant)|Create a single RGBA block of color. Useful for masks, overlays and general filtering.
[SHAPE GENERATOR ✨](https://github.com/Amorano/Jovimetrix/wiki/CREATE#-shape-generator)|Generate polyhedra for masking or texture work.
[TEXT GENERATOR 📝](https://github.com/Amorano/Jovimetrix/wiki/CREATE#-text-generator)|Uses any system font with auto-fit or manual placement.
[STEREOGRAM 📻](https://github.com/Amorano/Jovimetrix/wiki/CREATE#-stereogram)|Make a magic eye stereograms.
[GLSL 🍩](https://github.com/Amorano/Jovimetrix/wiki/CREATE#-glsl)|GLSL Shader support
<img width=225/>|<img width=800/>

[ADJUST](https://github.com/Amorano/Jovimetrix/wiki/ADJUST#adjust) | &nbsp;
---|---
[ADJUST 🕸️](https://github.com/Amorano/Jovimetrix/wiki/ADJUST#%EF%B8%8F-adjust)|Blur, Sharpen, Emboss, Levels, HSV, Edge detection.
[COLOR MATCH 💞](https://github.com/Amorano/Jovimetrix/wiki/ADJUST#-color-match)|Project the colors of one image  onto another or use a pre-defined color target.
[THRESHOLD 📉](https://github.com/Amorano/Jovimetrix/wiki/ADJUST#-threshold)|Clip an input based on a mid point value.
<img width=225/>|<img width=800/>

[COMPOSE](https://github.com/Amorano/Jovimetrix/wiki/COMPOSE) | &nbsp;
---|---
[TRANSFORM 🏝️](https://github.com/Amorano/Jovimetrix/wiki/COMPOSE#-transform)|Translate, Rotate, Scale, Tile, Mirror, Re-project and invert an input.
[BLEND ⚗️](https://github.com/Amorano/Jovimetrix/wiki/COMPOSE#%EF%B8%8F-blend)|Applies selected operation to 2 inputs with optional mask using a linear blend (alpha).
[PIXEL SPLIT 💔](https://github.com/Amorano/Jovimetrix/wiki/COMPOSE#-pixel-split)|Splits images into constituent R, G and B and A channels.
[PIXEL MERGE 🫂](https://github.com/Amorano/Jovimetrix/wiki/COMPOSE#-pixel-merge)|Combine 3 or 4 inputs into a single image
[STACK ➕](https://github.com/Amorano/Jovimetrix/wiki/COMPOSE#-stack)|Union multiple latents horizontal, vertical or in a grid.
[CROP ✂️](https://github.com/Amorano/Jovimetrix/wiki/COMPOSE#%EF%B8%8F-crop)|Clip away sections of an image and backfill with optional color matte.
[COLOR THEORY 🛞](https://github.com/Amorano/Jovimetrix/wiki/COMPOSE#-color-theory)|Generate Complimentary, Triadic and Tetradic color sets.
<img width=225/>|<img width=800/>

[CALC](https://github.com/Amorano/Jovimetrix/wiki/CALC) | &nbsp;
---|---
[VALUE #️⃣](https://github.com/Amorano/Jovimetrix/wiki/CALC#%EF%B8%8F%E2%83%A3-value)|Create a value for most types; also universal constants.
[CONVERT🧬](https://github.com/Amorano/Jovimetrix/wiki/CALC#-convert)|Convert INT, FLOAT, VEC*, STRING and BOOL.
[CALC OP UNARY 🎲](https://github.com/Amorano/Jovimetrix/wiki/CALC#-calc-op-unary)|Perform a Unary Operation on an input.
[CALC OP BINARY 🌟](https://github.com/Amorano/Jovimetrix/wiki/CALC#-calc-op-binary)|Perform a Binary Operation on two inputs.
[LERP 🔰](https://github.com/Amorano/Jovimetrix/wiki/CALC#-lerp)|Interpolate between two values with or without a smoothing.
<img width=225/>|<img width=800/>

[ANIMATE](https://github.com/Amorano/Jovimetrix/wiki/ANIMATE) | &nbsp;
---|---
[TICK ⏱](https://github.com/Amorano/Jovimetrix/wiki/ANIMATE#-tick)|Periodic pulse exporting normalized, delta since last pulse and count.
[WAVE GENERATOR 🌊](https://github.com/Amorano/Jovimetrix/wiki/ANIMATE#-wave_generator)|Periodic and Non-Periodic Sinosodials.
<img width=225/>|<img width=800/>

[FLOW](https://github.com/Amorano/Jovimetrix/wiki/FLOW) | &nbsp;
---|---
[COMPARISON 🕵🏽](https://github.com/Amorano/Jovimetrix/wiki/FLOW#-comparison)|Compare two inputs: A=B, A!=B, A>B, A>=B, A<B, A<=B
[DELAY ✋🏽](https://github.com/Amorano/Jovimetrix/wiki/FLOW#-delay)|Delay traffic. Electrons on the data bus go round.
[HOLD VALUE 🫴🏽](https://github.com/Amorano/Jovimetrix/wiki/FLOW#-hold-value)|When engaged will send the last value it had even with new values arriving.
<img width=225/>|<img width=800/>

[DEVICE](https://github.com/Amorano/Jovimetrix/wiki/DEVICE) | &nbsp;
---|---
[MIDI READER🎹](https://github.com/Amorano/Jovimetrix/wiki/DEVICE#-midi-reader)|Capture MIDI devices and pass the data into Comfy.
[MIDI MESSAGE🎛️](https://github.com/Amorano/Jovimetrix/wiki/DEVICE#%EF%B8%8F-midi-message)|Expands a MIDI message into its values.
[MIDI FILTER ✳️](https://github.com/Amorano/Jovimetrix/wiki/DEVICE#%EF%B8%8F-midi-filter)|Filter MIDI messages by channel, message type or value.
[MIDI FILTER EZ ❇️](https://github.com/Amorano/Jovimetrix/wiki/DEVICE#%EF%B8%8F-midi-filter-ez)|Filter MIDI messages by channel, message type or value.
[STREAM READER📺](https://github.com/Amorano/Jovimetrix/wiki/DEVICE#-stream-reader)|Connect system media devices and remote streams into ComfyUI workflows.
[STREAM WRITER🎞️](https://github.com/Amorano/Jovimetrix/wiki/DEVICE#%EF%B8%8F-stream-writer)|Broadcast ComfyUI Node outputs to custom webserver endpoint.
<img width=225/>|<img width=800/>

[AUDIO](https://github.com/Amorano/Jovimetrix/wiki/AUDIO) | &nbsp;
---|---
[GRAPH WAVE▶ ılıılı](https://github.com/Amorano/Jovimetrix/wiki/AUDIO#-graph-wave)|Import and display audio linear waveform data.
<img width=225/>|<img width=800/>

[UTILITY](https://github.com/Amorano/Jovimetrix/wiki/UTILITY) | &nbsp;
---|---
[VALUE GRAPH📈](https://github.com/Amorano/Jovimetrix/wiki/UTILITY#-value-graph)|Graphs historical execution run values
[AKASHIC📓](https://github.com/Amorano/Jovimetrix/wiki/UTILITY#-akashic)|Display the top level attributes of an output
[QUEUE🗃](https://github.com/Amorano/Jovimetrix/wiki/UTILITY#-queue)|Cycle lists of images files or strings for node inputs.
[SELECT🤏🏽](https://github.com/Amorano/Jovimetrix/wiki/UTILITY#-select)|Select an item from a user explicit list of inputs.
[ROUTE🚌](https://github.com/Amorano/Jovimetrix/wiki/UTILITY#-route)|Pass all data because the default is broken on connection
[EXPORT 📽](https://github.com/Amorano/Jovimetrix/wiki/UTILITY#-export)|Take your frames out static or animated (GIF)
[IMAGE DIFF 📏](https://github.com/Amorano/Jovimetrix/wiki/UTILITY#-image-diff)|Explicitly show the differences between two images via self-similarity index
<img width=225/>|<img width=800/>

[GLSL](https://github.com/Amorano/Jovimetrix/wiki/GLSL) | &nbsp;
---|---
[GLSL - GRAYSCALE](https://github.com/Amorano/Jovimetrix/wiki/GLSL#-glsl-grayscale)|Converts input to grayscale.
[GLSL - NOISE](https://github.com/Amorano/Jovimetrix/wiki/GLSL#-glsl-noise)|Various noise functions
[GLSL - PATTERN](https://github.com/Amorano/Jovimetrix/wiki/GLSL#-glsl-pattern)|Pattern factory wip.
[GLSL - POLYGON](https://github.com/Amorano/Jovimetrix/wiki/GLSL#-glsl-polygon)|Make convex polygons
[GLSL - MAP](https://github.com/Amorano/Jovimetrix/wiki/GLSL#-glsl-map)|Re-project an image
[GLSL - SELECT RANGE](https://github.com/Amorano/Jovimetrix/wiki/GLSL#-glsl-select-range)|Select a value range from an image for masking.
[GLSL - MIRROR](https://github.com/Amorano/Jovimetrix/wiki/GLSL#-glsl-mirror)|Mirror an input with an optional custom pivot.
[GLSL - ROTATE](https://github.com/Amorano/Jovimetrix/wiki/GLSL#-glsl-rotate)|Rotate an input.
[GLSL - TILER](https://github.com/Amorano/Jovimetrix/wiki/GLSL#-glsl-tiler)|A tiling utility wip.
[GLSL - VFX](https://github.com/Amorano/Jovimetrix/wiki/GLSL#-glsl-vfx)|Various Video Effects.
<img width=225/>|<img width=800/>

<!---------------------------------------------------------------------------->

# ACKNOWLEDGEMENTS

Everyone mentioned here has been tireless in helping me, at some point, construct all the material in Jovimetrix.

&nbsp; | &nbsp;
---|---
[Stable Diffusion](https://stability.ai/stable-diffusion/)|without this, we would all still only be using pixel pushing tools
[ComfyUI](https://github.com/comfyanonymous/ComfyUI)|Thank You! for existing
<img width=225/>|<img width=auto/>

## COMFY DEVELOPERS & NODE MAKERS & COMMUNITY BUILDERS

&nbsp; | &nbsp; | &nbsp;
---|---|---
[comfy](https://github.com/comfyanonymous)|[Chris Goringe](https://github.com/chrisgoringe)|[Purz](https://github.com/purzbeats)
[pythongosssss](https://github.com/pythongosssss)|[melmass](https://github.com/melMass)|[Fizzledorf](https://github.com/Fizzledorf)
[Dr. Lt. Data](https://github.com/ltdrdata)|[Trung0246](https://github.com/Trung0246)|[Fannovel16](https://github.com/Fannovel16)
[Kijai](https://github.com/Kijai)|[WASasquatch](https://github.com/WASasquatch)|[MatisseTec](https://github.com/MatissesProjects)
[rgthree](https://github.com/rgthree)|[Suzue1](https://github.com/Suzie1)
<img width=250/>|<img width=250/>|<img width=250/>

<!---------------------------------------------------------------------------->

# HELP & DONATIONS FOR DONATION WARE

Everything here is made because I wanted to make it.
Everything you are looking for here that you cant find doesnt exsit because I didnt make it.
If you feel like helping with text or code contributions, please pull and send me any PRs.
[If you feel like donating money resource instead, you can always use my ko-fi](https://ko-fi.com/alexandermorano).
