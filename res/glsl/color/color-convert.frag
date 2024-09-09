// name: COLOR CONVERSION
// desc: Convert an image from one color space (RGB, HSV, LAB, XYZ) to another.
// category: COLOR

uniform sampler2D image; // | Image to convert
uniform int operator; // EnumGLSLColorConvert | conversion operation to perform.

const vec3 D65 = vec3(95.047, 100.0, 108.883);

// =============================================================================
// PROTOTYPES
// =============================================================================

vec3 rgb2xyz(vec3 rgb);
vec3 xyz2rgb(vec3 xyz);
vec3 xyz2lab(vec3 xyz);

// =============================================================================
// RGB
// =============================================================================

vec3 rgb2hsv(vec3 rgb) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(rgb.bg, K.wz), vec4(rgb.gb, K.xy), step(rgb.b, rgb.g));
    vec4 q = mix(vec4(p.xyw, rgb.r), vec4(rgb.r, p.yzx), step(p.x, rgb.r));
    float d = q.x - min(q.w, q.y);
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + M_EPSILON)), d / (q.x + M_EPSILON), q.x);
}

vec3 rgb2lab(vec3 rgb) {
    vec3 xyz = rgb2xyz(rgb);
    return xyz2lab(xyz);
}

vec3 rgb2xyz(vec3 rgb) {
    vec3 tmp;
    tmp.x = (rgb.r > 0.04045) ? pow((rgb.r + 0.055) / 1.055, 2.4) : rgb.r / 12.92;
    tmp.y = (rgb.g > 0.04045) ? pow((rgb.g + 0.055) / 1.055, 2.4) : rgb.g / 12.92;
    tmp.z = (rgb.b > 0.04045) ? pow((rgb.b + 0.055) / 1.055, 2.4) : rgb.b / 12.92;
    return 100.0 * tmp * mat3(
        0.4124, 0.3576, 0.1805,
        0.2126, 0.7152, 0.0722,
        0.0193, 0.1192, 0.9505
    );
}

// =============================================================================
// HSV
// =============================================================================

vec3 hsv2rgb(vec3 hsv) {
    hsv = vec3(hsv.x, clamp(hsv.yz, 0.0, 1.0));
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(hsv.xxx + K.xyz) * 6.0 - K.www);
    return hsv.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), hsv.y);
}

vec3 hsv2lab(vec3 hsv) {
    float H = hsv.x * 360.0;
    float S = hsv.y;
    float V = hsv.z;

    // Convert to LAB
    float L = V * 100.0;
    float C = S * L;

    float h = H * M_PI / 180.0;
    float a = C * cos(h);
    float b = C * sin(h);

    // Normalize LAB
    return vec3(L / 100.0, (a + 128.0) / 255.0, (b + 128.0) / 255.0);
}

vec3 hsv2xyz(vec3 hsv) {
    vec3 rgb = hsv2rgb(hsv);
    return rgb2xyz(rgb);
}

// =============================================================================
// LAB
// =============================================================================

vec3 lab2xyz(vec3 lab) {
    float fy = (lab.x + 16.0) / 116.0;
    float fx = lab.y / 500.0 + fy;
    float fz = fy - lab.z / 200.0;
    return vec3(
         95.047 * ((fx > 0.206897) ? fx * fx * fx : (fx - 16.0 / 116.0) / 7.787),
        100.000 * ((fy > 0.206897) ? fy * fy * fy : (fy - 16.0 / 116.0) / 7.787),
        108.883 * ((fz > 0.206897) ? fz * fz * fz : (fz - 16.0 / 116.0) / 7.787)
    );
}

vec3 lab2rgb(vec3 lab) {
    vec3 xyz = lab2xyz(lab);
    return xyz2rgb(xyz);
}

vec3 lab2hsv(vec3 lab) {
    vec3 rgb = lab2rgb(lab);
    return rgb2hsv(rgb);
}

// =============================================================================
// XYZ
// =============================================================================

vec3 xyz2rgb(vec3 xyz) {
    vec3 v =  xyz / 100.0 * mat3(
        3.2406, -1.5372, -0.4986,
        -0.9689, 1.8758, 0.0415,
        0.0557, -0.2040, 1.0570
    );
    vec3 r;
    r.x = ( v.r > 0.0031308 ) ? (( 1.055 * pow( v.r, ( 1.0 / 2.4 ))) - 0.055 ) : 12.92 * v.r;
    r.y = ( v.g > 0.0031308 ) ? (( 1.055 * pow( v.g, ( 1.0 / 2.4 ))) - 0.055 ) : 12.92 * v.g;
    r.z = ( v.b > 0.0031308 ) ? (( 1.055 * pow( v.b, ( 1.0 / 2.4 ))) - 0.055 ) : 12.92 * v.b;
    return r;
}

vec3 xyz2hsv(vec3 xyz) {
    vec3 rgb = xyz2rgb(xyz);
    return rgb2hsv(rgb);
}

vec3 xyz2lab(vec3 xyz) {
    vec3 n = xyz / D65;
    vec3 v;
    v.x = ( n.x > 0.008856 ) ? pow( n.x, 1.0 / 3.0 ) : ( 7.787 * n.x ) + ( 16.0 / 116.0 );
    v.y = ( n.y > 0.008856 ) ? pow( n.y, 1.0 / 3.0 ) : ( 7.787 * n.y ) + ( 16.0 / 116.0 );
    v.z = ( n.z > 0.008856 ) ? pow( n.z, 1.0 / 3.0 ) : ( 7.787 * n.z ) + ( 16.0 / 116.0 );
    return vec3(( 116.0 * v.y ) - 16.0, 500.0 * ( v.x - v.y ), 200.0 * ( v.y - v.z ));
}

// =============================================================================
// SELECTOR
// =============================================================================

vec3 convertColor(vec3 color, int operator) {
    // RGB
    if (operator == 0) {
        return rgb2hsv(color);
    } else if (operator == 1) {
        return rgb2lab(color);
    } else if (operator == 2) {
        return rgb2xyz(color);
    // HSV
    } else if (operator == 10) {
        return hsv2rgb(color);
    } else if (operator == 11) {
        return hsv2lab(color);
    } else if (operator == 12) {
        return hsv2xyz(color);
    // LAB
    } else if (operator == 20) {
        return lab2rgb(color);
    } else if (operator == 21) {
        return lab2hsv(color);
    } else if (operator == 22) {
        return lab2xyz(color);
    // XYZ
    } else if (operator == 30) {
        return xyz2rgb(color);
    } else if (operator == 31) {
        return xyz2hsv(color);
    } else if (operator == 32) {
        return xyz2lab(color);
    }
    return color;
}

void mainImage(out vec4 fragColor, vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec4 color = texture(image, uv);
    vec3 rgb = convertColor(color.rgb, operator);
    fragColor = vec4(rgb, color.a);
}