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

<img
    style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
    src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/overview.png"
    alt="GENERAL NODE OVERVIEW">
</img>

<b><code>COPY</code> [EXAMPLE IMAGES](https://github.com/Amorano/Jovimetrix-examples/blob/master/node) <code>INTO COMFYUI TO LOAD SHOWN WORKFLOW</code></b>

<details>
  <summary><b>CREATE</b></summary>
  <ul>
    <li>Constant color node for when you need a block of color</li>
    <li>Nodes that generate images and masks in various polygonal shapes
      <ul>
        <li>🟢 Ellipse & Circle </li>
        <li>🟪 Rectangle & Square</li>
        <li>🦚 Polygon of 3 or more sides</li>
      </ul>
    </li>
    * Per Pixel Shader with input support. Slow but works until the ![GLSL]("GLSL Node") is available. Variables pre-defined for use in the loop include:
      <ul>
        <li><code>$x</code>, <code>$y</code>: Current image (x, y)</li>
        <li><code>$u</code>, <code>$v</code>: Normalized texture coordinates [0..1]</li>
        <li><code>$w</code>, <code>$h</code>: Width and Height of the target output</li>
        <li><code>$ir</code>, <code>$ig</code>, <code>$ib</code>: Red, Green & Blue values for current image input (x, y)</li>
      </ul>
    </li>

NODE | OVERVIEW | COMFY UI
---|---|---
🟪 CONSTANT|Set a single RGB value. Useful for masks, overlays and general filtering|![🟪 CONSTANT](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/create/constant/node_create-constant.png "constant color node for when you need a block of color")
✨ SHAPE GENERATOR|Generate polyhedra for masking or texture work|![✨ SHAPE GENERATOR](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/create/shape_generator/node_create-shape_generator.png "Generate images and masks in various polygonal shapes")
🔆 PER PIXEL SHADER|Per Pixel user function for each R, G, B channel|![🔆 PER PIXEL SHADER](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/create/pixel_shader/node_create-pixel.png "Per Pixel shader")
🍩 GLSL|GLSL Shader support NOT YET|![🍩 GLSL](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/create/glsl/node_create-glsl.png "GLSL Shader support NOT YET")
  </ul>
  <ul>
    <details>
      <summary><b>🟪 CONSTANT EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/create/constant/node_create-constant-simple.png"
        alt="constant color node for when you need a block of color">
      </img>
    </details>
    <details>
      <summary><b>✨ SHAPE GENERATOR EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/create/shape_generator/node_create-shape_generator-simple.png"
        alt="Generate images and masks in various polygonal shapes">
      </img>
    </details>
    <details>
      <summary><b>🔆 PER PIXEL SHADER EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/create/pixel_shader/node_create-pixel_shader-simple.png"
        alt="Per Pixel shader">
      </img>
    </details>
    <details>
      <summary><b>🍩 GLSL EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/create/glsl/node_create-glsl-simple.png" alt="GLSL Shader support NOT YET">
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
🕸️ ADJUST|Blur, Sharpen and Emboss an input|![🕸️ ADJUST](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/adjust/adjust/node_adjust-adjust.png "Blur, Sharpen and Emboss an input")
💞 COLOR MATCH|Color match based on a B input or template color maps|![💞 COLOR MATCH](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/adjust/color_match/node_adjust-color_match.png "Color match based on a B input or template color maps")
🔳 FIND EDGES|Find Edges of an input|![🔳 FIND EDGES](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/adjust/find_edges/node_adjust-find_edges.png "Find Edges of an input")
🌈 HSV|Adjust Hue, Saturation, Value, Gamma, Contrast and Exposure of an input|![🌈 HSV](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/adjust/hsv/node_adjust-hsv.png "Adjust Hue, Saturation, Value, Gamma, Contrast and Exposure of an input")
🛗 LEVELS|Low, Mid, High range clipping|![🛗 LEVELS](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/adjust/levels/node_adjust-levels.png "Low, Mid, High range clipping")
📉 THRESHOLD|Clip an input based on a mid point value|![📉 THRESHOLD](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/adjust/threshold/node_adjust-threshold.png "Clip an input based on a mid point value")
  </ul>
  <ul>
    <details>
      <summary><b>🕸️ ADJUST EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/adjust/adjust/node_adjust-adjust-simple.png" alt="Blur, Sharpen and Emboss an input">
      </img>
    </details>
    <details>
      <summary><b>💞 COLOR MATCH EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/adjust/color_match/node_adjust-color_match-simple.png" alt="Color match based on a B input or template color maps">
      </img>
    </details>
    <details>
      <summary><b>🔳 FIND EDGES EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/adjust/find_edges/node_adjust-find_edges-simple.png" alt="Find Edges of an input">
      </img>
    </details>
    <details>
      <summary><b>🌈 HSV EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/adjust/hsv/node_adjust-hsv-simple.png" alt="Adjust Hue, Saturation, Value, Gamma, Contrast and Exposure of an input">
      </img>
    </details>
    <details>
      <summary><b>🛗 LEVELS EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/adjust/levels/node_adjust-levels-simple.png" alt="Low, Mid, High range clipping">
      </img>
    </details>
    <details>
      <summary><b>📉 THRESHOLD EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/adjust/threshold/node_adjust-threshold-simple.png" alt="Clip an input based on a mid point value">
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
🌱 TRS|Translate, Rotate, and Scale without extra options|![🌱 TRS](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/trans/trs/node_trans-trs.png "Translate, Rotate, and Scale without extra options")
🌱 TRANSFORM|Translate, Rotate, and Scale an input. Options allow for CROP or WRAPing of the edges|![🌱 TRANSFORM](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/trans/transform/node_trans-transform.png "Translate, Rotate, and Scale an input. Options allow for CROP or WRAPing of the edges")
🔳 TILE|Repeat an input along the X, Y or XY at irregular intervals|![🔳 TILE](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/trans/tile/node_trans-tile.png "Repeat an input along the X, Y or XY at irregular intervals")
🔰 MIRROR|Flip an input across the X axis, the Y Axis or both, with independent centers|![🔰 MIRROR](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/trans/mirror/node_trans-mirror.png "Flip an input across the X axis, the Y Axis or both, with independant centers")
🗺️ PROJECTION|Project into various perspective transformations|![🗺️ PROJECTION](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/trans/projection/node_trans-projection.png "Project into various perspective transformations")

  <ul>
    <details>
      <summary><b>🌱 TRS EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/trans/trs/node_trans-trs-simple.png" alt="Translate, Rotate, and Scale without extra options">
      </img>
    </details>
    <details>
      <summary><b>🌱 TRANSFORM EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/trans/transform/node_trans-transform-simple.png" alt="Translate, Rotate, and Scale an input. Options allow for CROP or WRAPing of the edges">
      </img>
    </details>
    <details>
      <summary><b>🔳 TILE EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/trans/tile/node_trans-tile-simple.png" alt="Repeat an input along the X, Y or XY at irregular intervals">
      </img>
    </details>
    <details>
      <summary><b>🔰 MIRROR EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/trans/mirror/node_trans-mirror-simple.png" alt="Flip an input across the X axis, the Y Axis or both, with independent centers">
      </img>
    </details>
    <details>
      <summary><b>🗺️ PROJECTION EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/trans/projection/node_trans-projection.png"
        alt="Project into various perspective transformations">
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

NODE | OVERVIEW | COMFY UI
--|---|---
⚗️ BLEND|Compose 2 inputs together with optional alpha mask along a linear blend|![⚗️ BLEND](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/comp/blend/node_comp-blend.png "Compose 2 inputs together an alpha mask and linear blend scalar")
💔 PIXEL SPLIT|Splits pixel blocks into it's constituent R, G and B channels|![💔 PIXEL SPLIT](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/comp/pixel_split/node_comp-pixel_split.png "Splits pixel blocks into it's constituent R, G and B channels")
🫱🏿‍🫲🏼 Pixel Merge|Combines multiple inputs into a single block of pixels|![🫱🏿‍🫲🏼 Pixel Merge](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/comp/pixel_merge/node_comp-pixel_merge.png "Combines multiple inputs into a single block of pixels")
➕ MERGE|Combine two inputs into a new image, side by side, top-down or in a grid|![➕ MERGE](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/comp/merge/node_comp-merge.png "Combine two inputs into a new image, side by side, top-down or in a grid")
✂️ CROP|Clip away sections of an image and backfill the matte|![✂️ CROP](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/comp/crop/node_comp-crop.png "Clip away sections of an image and backfill the matte")
🛞 COLOR THEORY|Generate Complimentary, Triadic and Tetradic color sets|![🛞 COLOR THEORY](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/comp/color_theory/node_comp-color_theory.png "Generate Complimentary, Triadic and Tetradic color sets")
  </ul>
  <ul>
    <details>
      <summary><b>⚗️ BLEND EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/comp/blend/node_comp-blend-simple.png" alt="Compose 2 inputs together an alpha mask and linear blend scalar">
      </img>
    </details>
    <details>
      <summary><b>💔 PIXEL SPLIT EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/comp/pixel_split/node_comp-pixel_split-simple.png" alt="Splits pixel blocks into it's constituent R, G and B channels">
      </img>
    </details>
    <details>
      <summary><b>🫱🏿‍🫲🏼 PIXEL MERGE EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/comp/pixel_merge/node_comp-pixel_merge-simple.png" alt="Combines multiple inputs into a single block of pixels">
      </img>
    </details>
    <details>
      <summary><b>➕ MERGE EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/comp/merge/node_comp-merge-simple.png" alt="Combine two inputs into a new image, side by side, top-down or in a grid">
      </img>
    </details>
    <details>
      <summary><b>✂️ CROP EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/comp/crop/node_comp-crop-simple.png" alt="Flip an input across the X axis, the Y Axis or both, with independent centers">
      </img>
    </details>
    <details>
      <summary><b>🛞 COLOR THEORY EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/comp/color_theory/node_comp-color_theory-simple.png" alt="Generate Complimentary, Triadic and Tetradic color sets">
      </img>
    </details>
  </ul>
</details>

<!---------------------------------------------------------------------------->

<details>
  <summary><b>ANIMATE</b></summary>
  <ul>
    <li></li>

NODE | OVERVIEW | COMFY UI
--|---|---
🕛 TICK|Periodic pulse exporting normalized, delta since last pulse and count.|![🕛 TICK](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/anim/tick/node_anim-tick.png "Periodic pulse exporting normalized, delta since last pulse and count")
🌊 WAVE GENERATOR|Periodic and Non-Periodic Sinosodials|![🌊 WAVE GENERATOR](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/anim/wave_generator/node_anim-wave_generator.png "Periodic and Non-Periodic Sinosodials")
  </ul>
  <ul>
    <details>
      <summary><b>🕛 TICK EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/anim/tick/node_anim-tick-simple.png" alt="Periodic pulse exporting normalized, delta since last pulse and count">
      </img>
    </details>
    <details>
      <summary><b>🌊 WAVE GENERATOR EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/anim/wave_generator/node_anim-wave_generator-simple.png" alt="Periodic and Non-Periodic Sinosodials">
      </img>
    </details>
  </ul>
</details>

<!---------------------------------------------------------------------------->

<details>
  <summary><b>FLOW</b></summary>
  <ul>
    <li></li>

NODE | OVERVIEW | COMFY UI
--|---|---
✋ DELAY|Pause or hold outputs|![✋ ROUTE](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/flow/delay/delay_flow-route.png "Pause or hold outputs")
🕵🏽 COMPARISON|A=B, A!=B, A&gt;B, A>=B, A&lt;B, A<=B|![🕵🏽 COMPARISON](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/flow/comparison/node_flow-comparison.png "A=B, A!=B, A&gt;B, A>=B, A&lt;B, A<=B")
🔀 IF-THEN-ELSE|IF A THEN B ELSE C|![🔀 IF-THEN-ELSE](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/flow/if-then-else/node_flow-if-then-else.png "IF A THEN B ELSE C")
  </ul>
  <ul>
    <details>
      <summary><b>✋ DELAY EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/flow/route/node_flow-delay-simple.png" alt="Pause or hold outputs">
      </img>
    </details>
    <details>
      <summary><b>🕵🏽 COMPARISON EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/flow/comparison/node_flow-comparison-simple.png" alt="A=B, A!=B, A&gt;B, A>=B, A&lt;B, A<=B">
      </img>
    </details>
    <details>
      <summary><b>🔀 IF-THEN-ELSE EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/flow/if-then-else/node_flow-if-then-else-simple.png" alt="IF A THEN B ELSE C">
      </img>
    </details>
  </ul>
</details>

<!---------------------------------------------------------------------------->

<details>
  <summary><b>DEVICE</b></summary>
  <ul>
    <li>Connect system media directly into ComfyUI workflows</li>
    <li>Broadcast ComfyUI media to mjpeg reader endpoints</li>
    <li>Poll and react to MIDI events</li>

NODE | OVERVIEW | COMFY UI
---|---|---
📺 STREAM READER|Connect system media directly into ComfyUI workflows|![📺 STREAM READER](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/device/stream_reader/node_device-stream_reader.png "Connect system media directly into ComfyUI workflows")
🎞️ STREAM WRITER|Broadcast ComfyUI Node outputs to custom webserver endpoint|![🎞️ STREAM WRITER](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/device/stream_writer/node_device-stream_writer.png "Broadcast ComfyUI Node outputs to custom webserver endpoint")
🎹 MIDI PORT|Capture a MIDI port and pass controls through to Comfy|![🎹 MIDI PORT](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/device/midi_reader/node_device-midi_reader.png "Capture a MIDI port and pass controls through to Comfy")
 </ul>
  <ul>
    <details>
      <summary><b>📺 STREAM READER EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/device/stream_reader/node_device-stream_reader-simple.png"
        alt="Simple webcam capture setup">
      </img>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/device/stream_reader/node_device-stream_reader-mask.png"
        alt="Webcam with a simple shape mask for realtime overlay">
      </img>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/device/stream_reader/node_device-stream_reader-multiple.png"
        alt="Complex filtering with webcams">
      </img>
    </details>
    <details>
      <summary><b>🎞️ STREAM WRITER EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/device/stream_writer/node_device-stream_writer-simple.png"
        alt="Write to a local server port">
      </img>
    </details>
    <details>
      <summary><b>🎹 MIDI PORT EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/device/midi_reader/node_device-midi_reader-simple.png"
        alt="Read from a midi device">
      </img>
    </details>
  </ul>
</details>

<!---------------------------------------------------------------------------->


<details>
  <summary><b>AUDIO</b></summary>
  <ul>
    <li> </li>

NODE | OVERVIEW | COMFY UI
--|---|---
🎶 GRAPH AUDIO|Export an audio file as a linear waveform along with the linear wave data|![🎶 GRAPH AUDIO](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/audio/graph_wave/node_audio-graph_wave.png "Export an audio file as a linear waveform along with the linear wave data")
  </ul>
  <ul>
    <details>
      <summary><b>🎶 GRAPH AUDIO EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/audio/graph_wave/node_audio-graph_wave-simple.png" alt="Export an audio file as a linear waveform along with the linear wave data">
      </img>
    </details>
  </ul>
</details>

<!---------------------------------------------------------------------------->

<details>
  <summary><b>UTILITY</b></summary>
  <ul>

NODE | OVERVIEW | COMFY UI
---|---|---
⚙️ OPTIONS|Change Jovimetrix Global Options|![⚙️ OPTIONS](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/util/options/node_util-options.png "Change Jovimetrix Global Options")
⚙️ OPTIONS|Change Jovimetrix Global Options|![⚙️ OPTIONS](https://github.com/Amorano/Jovimetrix-examples/blob/master/node/util/options/node_util-options.png "Change Jovimetrix Global Options")
  </ul>
  <ul>
    <details>
      <summary><b>⚙️ OPTIONS EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/util/options/node_util-options-simple.png" alt="Change Jovimetrix Global Options">
      </img>
    </details>
    <details>
      <summary><b>⚙️ OPTIONS EXAMPLES</b></summary>
      <img
        style="display: block; margin-left: auto; margin-right: auto; width: 100%;"
        src="https://github.com/Amorano/Jovimetrix-examples/blob/master/node/util/options/node_util-options-simple.png" alt="Change Jovimetrix Global Options">
      </img>
    </details>
  </ul>
</details>

## ACKNOWLEDGEMENTS
WHO | WHAT
---|---
[Stable Diffusion](https://stability.ai/stable-diffusion/)|without this, we would all still only be using pixel pushing tools
[ComfyUI](https://github.com/comfyanonymous/ComfyUI)|THANK YOU for existing

### COMFY & NODE DEVELOPERS

All the developers I mention here are developer's developers. They have been endless in the help they provide, entirely free, through out many months of - sometimes - frustrating development work in this super nascent industry.

WHO | WHAT
---|---
[comfy](https://github.com/comfyanonymous)|for endless updates to Comfy and building it in the first place
[pythongosssss](https://github.com/pythongosssss)|...so many magic tricks
[Dr. Lt. Data](https://github.com/ltdrdata)|ComfyManager and workflows
[WASasquatch](https://github.com/WASasquatch)|excellent core nodes
[melmass](https://github.com/melMass)
[rgthree](https://github.com/rgthree)
[Fizzledorf](https://github.com/Fizzledorf)
[Fannovel16](https://github.com/Fannovel16)
[Kijai](https://github.com/Kijai)

### WORKFLOW MAGICIANS

WHO | WHAT
---|---
[Purz](https://github.com/purzbeats)
[searge](https://civitai.com/user/searge)
[Akatsuzi](https://civitai.com/user/Akatsuzi)
[LexChen](https://civitai.com/user/LexChen)

### COMMUNITY BUILDERS

WHO | WHAT
---|---
[MatisseTec](https://www.twitch.tv/matissetec)|constantly doing free work on cutting edge tech
[Purz](https://www.purz.xyz/)|mentioned twice cause he is that frigging good.

---

## TODO:

- [⭕] specific single examples for all nodes
- [⭕] hook GLSL context in litegraph
- [⭕] ~redo camera with stream reader~ / writer defaults
- [⭕] generalized section to explain common parameters (w/h/invert/mode)
- [⭕] add flip/alternate? to tile node

### TODO NODES:

- [ ] audio
- [ ] database
- [ ] shotgrid events
- [ ] rest points
- [ ] keyframe system or general event timeline
