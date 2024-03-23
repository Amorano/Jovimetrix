!NAME! || !CAT!

The Blend node merges two input sources using a defined Blend Mode and an optional Opacity mask.

The Blend Mode determines how attributes from each layer interact, influencing the resulting composite. Additionally, the Opacity mask allows for precise control over transparency, enhancing blending effects.

WIKI: !URL!

MODE|EXPECTED|DESCRIPTION
---|---|---
NORMAL | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/normal_expected.png)|
ADDITIVE | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/additive_expected.png)|
NEGATION | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/negation_expected.png)|
DIFFERENCE | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/difference_expected.png)|
MULTIPLY | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/multiply_expected.png)|
DIVIDE | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/divide_expected.png)|
LIGHTEN | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/lighten_expected.png)|
DARKEN | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/darken_expected.png)|
SCREEN | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/screen_expected.png)|
BURN | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/colourburn_expected.png)|
DODGE | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/colourdodge_expected.png)|
OVERLAY | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/overlay_expected.png)|
HUE | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/hue_expected.png)|
SATURATION | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/saturation_expected.png)|
LUMINOSITY | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/luminosity_expected.png)|
COLOR | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/colour_expected.png)|
SOFT | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/softlight_expected.png)|
HARD | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/hardlight_expected.png)|
PIN | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/pinlight_expected.png)|
VIVID | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/vividlight_expected.png)|
EXCLUSION | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/exclusion_expected.png)|
REFLECT | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/reflect_expected.png)|
GLOW | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/glow_expected.png)|
XOR | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/xor_expected.png)|
EXTRACT | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/grainextract_expected.png)|
MERGE | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/grainmerge_expected.png)|
DESTIN | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/additive_expected.png)|'clip' composite mode. All parts of 'layer above' which are alpha in 'layer below' will be made also alpha in 'layer above' (to whatever degree of alpha they were). Destination which overlaps the source, replaces the source.
DESTOUT | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/additive_expected.png) | Reverse 'Clip' composite mode.All parts of 'layer below' which are alpha in 'layer above' will be made also alpha in 'layer below' (to whatever degree of alpha they were).
SRCATOP | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/additive_expected.png) | Place the layer below above the 'layer above' in places where the 'layer above' exists.
DESTATOP | ![](https://raw.githubusercontent.com/FHPythonUtils/BlendModes/master/tests/data/additive_expected.png) | Place the layer below above the 'layer above' in places where the 'layer above' exists, where 'layer below' does not exist, but 'layer above' does, place 'layer-above'.
<img width=150/>|<img width=200/>|<img width=250/>

!URL_VID!