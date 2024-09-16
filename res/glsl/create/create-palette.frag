// The MIT License
// https://www.youtube.com/c/InigoQuilez
// https://iquilezles.org/
// Copyright © 2015 Inigo Quilez
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// See https://iquilezles.org/articles/palettes for more information
//
// name: COLOR PALETTE
// desc: Color palette creation using the formula: color(t) = a + b * cos[tau(c*t+d)]. See https://iquilezles.org/articles/palettes for more information.
// category: CREATE

#include .lib/const.lib

uniform vec3 bias;  // 0.5,0.5,0.5;0 | scale and bias (dc offset)
uniform vec3 amp;   // 0.5,0.5,0.5   | contrast and brightness (amplitude)
uniform vec3 freq;  // 1,1,1         | color cycle (R, G and B) (frequency)
uniform vec3 phase; // 0,0,0         | starting offset for the cycle

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 uv = fragCoord.xy / iResolution.xy;
    vec3 col = bias + amp * cos(M_PI * (freq * uv.x + phase));
	fragColor = vec4(col, 1.0);
}