> [!CAUTION]
> <h3><p align="center">⚠️ SUBJECT TO CHANGE PRIOR TO VERSION 1.0. USE AT YOUR OWN RISK ⚠️</p></h3>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Amorano/Jovimetrix-examples/blob/master/res/logo-jovimetrix.png">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/Amorano/Jovimetrix-examples/blob/master/res/logo-jovimetrix-light.png">
  <img alt="ComfyUI Nodes for procedural masking, live composition and video manipulation">
</picture>

<h3><p align="center">
<a href="https://github.com/comfyanonymous/ComfyUI">COMFYUI</a> Nodes for procedural masking, live composition and video manipulation
</p></h3>

<!---------------------------------------------------------------------------->

# INSTALLATION

If you have [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) installed you can search for Jovimetrix and install from the manager's database.

## GIFSKI SUPPORT

If you have [GIFSKI](https://gif.ski/) installed you can enable the option for the Export Node to use GIFSKI when outputting frames.

You will need to add an environment var so it knows you have it installed and where:

set JOV_GIFSKI=[path to gifski]

Once set the GIFSKI option should appear in the Export Node drop down list of output target formats.


## FFMEPG

The audio nodes require FFMPEG. You can find the official [FFMPEG](https://ffmpeg.org "official FFMPEG binaries") here. Follow it's installation instructions for your specific operating system.

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

CREATE | WHAT
---|---
[CONSTANT 🟪](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_constant.png)|Create a single RGBA block of color. Useful for masks, overlays and general filtering.
[SHAPE GENERATOR ✨](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_shape_generator.png)|Generate polyhedra for masking or texture work.
[TEXT GENERATOR 📝](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_text_generator.png)|Uses any system font with auto-fit or manual placement.
[STEREOGRAM 📻](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_stereogram.png)|Make a magic eye stereograms.
[GLSL 🍩](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_glsl.png)|GLSL Shader support

ADJUST | WHAT
---|---
[ADJUST 🕸️](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_adjust.png)|Blur, Sharpen, Emboss, Levels, HSV, Edge detection.
[COLOR MATCH 💞](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_color_match.png)|Project the colors of one image  onto another or use a pre-defined color target.
[THRESHOLD 📉](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_threshold.png)|Clip an input based on a mid point value.

COMPOSE | WHAT
---|---
[BLEND ⚗️](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_blend.png)|Applies selected operation to 2 inputs with optional mask using a linear blend (alpha).
[PIXEL SPLIT 💔](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_pixel_split.png)|Splits images into constituent R, G and B and A channels.
[PIXEL MERGE 🫂](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_pixel_merge.png)|Combine 3 or 4 inputs into a single image
[TRANSFORM 🏝️](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_transform.png)|Translate, Rotate, Scale, Tile, Mirror, Re-project and invert an input.
[STACK ➕](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_stack.png)|Union multiple latents horizontal, vertical or in a grid.
[CROP ✂️](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_crop.png)|Clip away sections of an image and backfill with optional color matte.
[COLOR THEORY 🛞](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_color_theory.png)|Generate Complimentary, Triadic and Tetradic color sets.

IMAGE | WHAT
---|---
[EXPORT 📽](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_export.png)|
[IMAGE DIFF 📏](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_image_diff.png)|

CALC | WHAT
---|---
[VALUE #️⃣](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_value.png)|Create a value for most types; also universal constants.
[CONVERT🧬](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_convert.png)|Convert INT, FLOAT, VEC*, STRING and BOOL.
[CALC OP UNARY 🎲](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_calc_unary.png)|Perform a Unary Operation on an input.
[CALC OP BINARY 🌟](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_calc_binary.png)|Perform a Binary Operation on two inputs.
[LERP 🔰](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_lerp.png)|Interpolate between two values with or without a smoothing.

ANIMATE | WHAT
---|---
[TICK ⏱](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_tick.png)|Periodic pulse exporting normalized, delta since last pulse and count.
[WAVE GENERATOR 🌊](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_wave_generator.png)|Periodic and Non-Periodic Sinosodials.

FLOW | WHAT
---|---
[COMPARISON 🕵🏽](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_comparison.png)|Compare two inputs: A=B, A!=B, A>B, A>=B, A<B, A<=B
[DELAY ✋🏽](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_delay.png)|Delay traffic. Electrons on the data bus go round.
[HOLD VALUE 🫴🏽](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_hold_value.png)|When engaged will send the last value it had even with new values arriving.
[IF-THEN-ELSE 🔀](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_if-then-else.png)|IF <valid> then A else B

DEVICE | WHAT
---|---
[MIDI READER🎹](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_midi_reader.png)|Capture MIDI devices and pass the data into Comfy.
[MIDI MESSAGE🎛️](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_midi_message.png)|Expands a MIDI message into its values.
[MIDI FILTER ✳️](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_midi_filter.png)|Filter MIDI messages by channel, message type or value.
[MIDI FILTER EZ ❇️](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_midi_filter_ez.png)|Filter MIDI messages by channel, message type or value.
[STREAM READER📺](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_stream_reader.png)|Connect system media devices and remote streams into ComfyUI workflows.
[STREAM WRITER🎞️](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_stream_writer.png)|Broadcast ComfyUI Node outputs to custom webserver endpoint.

AUDIO | WHAT
---|---
[GRAPH WAVE▶ ılıılı](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_graph_wave.png)|Import and display audio linear waveform data.

UTILITY | WHAT
---|---
[VALUE GRAPH📈](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_value_graph.png)|Graphs historical execution run values
[AKASHIC📓](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_akashic.png)|Display the top level attributes of an output
[QUEUE🗃](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_queue.png)|Cycle lists of images files or strings for node inputs.
[SELECT🤏🏽](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_select.png)|Select an item from a user explicit list of inputs.
[RE-ROUTE🚌](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/node_reroute.png)|Pass all data because the default is broken on connection

GLSL | WHAT
---|---
[GLSL - GRAYSCALE](https://github.com/Amorano/Jovimetrix-examples/blob/master/glsl/glsl-grayscale.png)|Converts input to grayscale.
[GLSL - NOISE](https://github.com/Amorano/Jovimetrix-examples/blob/master/glsl/glsl-noise.png)|Various noise functions
[GLSL - PATTERN](https://github.com/Amorano/Jovimetrix-examples/blob/master/glsl/glsl-pattern.png)|Pattern factory wip.
[GLSL - POLYGON](https://github.com/Amorano/Jovimetrix-examples/blob/master/glsl/glsl-polygon.png)|Make convex polygons
[GLSL - MAP](https://github.com/Amorano/Jovimetrix-examples/blob/master/glsl/glsl-map.png)|Re-project an image
[GLSL - SELECT RANGE](https://github.com/Amorano/Jovimetrix-examples/blob/master/glsl/glsl-select_range.png)|Select a value range from an image for masking.
[GLSL - MIRROR](https://github.com/Amorano/Jovimetrix-examples/blob/master/glsl/glsl-mirror.png)|Mirror an input with an optional custom pivot.
[GLSL - ROTATE](https://github.com/Amorano/Jovimetrix-examples/blob/master/glsl/glsl-rotate.png)|Rotate an input.
[GLSL - TILER](https://github.com/Amorano/Jovimetrix-examples/blob/master/glsl/glsl-tiler.png)|A tiling utility wip.
[GLSL - VFX](https://github.com/Amorano/Jovimetrix-examples/blob/master/glsl/glsl-vfx.png)|Various Video Effects.

<!---------------------------------------------------------------------------->

# ACKNOWLEDGEMENTS

Everyone mentioned here has been tireless in helping me, at some point, construct all the material in Jovimetrix.

&nbsp; | &nbsp;
---|---
[Stable Diffusion](https://stability.ai/stable-diffusion/)|without this, we would all still only be using pixel pushing tools
[ComfyUI](https://github.com/comfyanonymous/ComfyUI)|Thank You! for existing

## COMFY DEVELOPERS & NODE MAKERS & COMMUNITY BUILDERS

&nbsp; | &nbsp; | &nbsp;
 ---|---|---
[comfy](https://github.com/comfyanonymous)|[Chris Goringe](https://github.com/chrisgoringe)|[Purz](https://github.com/purzbeats)
[pythongosssss](https://github.com/pythongosssss)|[melmass](https://github.com/melMass)|[Fizzledorf](https://github.com/Fizzledorf)
[Dr. Lt. Data](https://github.com/ltdrdata)|[Trung0246](https://github.com/Trung0246)|[Fannovel16](https://github.com/Fannovel16)
[Kijai](https://github.com/Kijai)|[WASasquatch](https://github.com/WASasquatch)|[MatisseTec](https://github.com/MatissesProjects)
[rgthree](https://github.com/rgthree)|[Suzue1](https://github.com/Suzie1)




