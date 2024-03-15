/**
 * File: util_color.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import * as util_config from './util_config.js'

// gets the CONFIG entry for name
export function node_color_get(node) {
    const find_me = node.type || node.name;
    if (find_me === undefined) {
        return
    }
    // First look to regex....
    for (const colors of util_config.CONFIG_REGEX) {
        if (colors.regex == "") {
            continue
        }
        const regex = new RegExp(colors.regex, 'i');
        const found = find_me.match(regex);
        if (found !== null && found[0].length > 0) {
            return colors;
        }
    }
    // now look to theme
    let color = util_config.CONFIG_THEME[find_me]
    if (color) {
        return color
    }
    color = util_config.NODE_LIST[find_me]
    // now look to category theme
    if (color && color.category) {
        const segments = color.category.split('/')
        let k = segments.join('/')
        while (k) {
            const found = util_config.CONFIG_THEME[k]
            if (found) {
                return found
            }
            const last = k.lastIndexOf('/')
            k = last !== -1 ? k.substring(0, last) : ''
        }
    }
    return null;
}

// refresh the color of a node
export function node_color_reset(node, refresh=true) {
    const data = node_color_get(node);
    if (data) {
        node.bgcolor = data.body;
        node.color = data.title;
        // console.info(node, data)
        if (refresh) {
            node.setDirtyCanvas(true, true);
        }
    }
}

export function node_color_list(nodes) {
    Object.entries(nodes).forEach((node) => {
        node_color_reset(node, false);
    })
    app.canvas.setDirty(true);
}

export function node_color_all() {
    app.graph._nodes.forEach((node) => {
        node_color_reset(node, false);
    })
    app.canvas.setDirty(true);
    // console.info("JOVI] all nodes color refreshed")
}

export function hex2rgb(hex) {
  hex = hex.replace(/^#/, '');
  const bigint = parseInt(hex, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return [r, g, b];
}

/*
* Parse a string "255,255,255,255" or a list[255,255,255,255] into hex
*/
export function rgb2hex(input) {
    const rgbArray = typeof input === 'string' ? input.match(/\d+/g) : input;
    if (rgbArray.length < 3) {
        throw new Error('input not 3 or 4 values');
    }
    const hexValues = rgbArray.map((value, index) => {
        if (index === 3 && !value) return 'ff';
        const hex = parseInt(value).toString(16);
        return hex.length === 1 ? '0' + hex : hex;
    });
    return '#' + hexValues.slice(0, 3).join('') + (hexValues[3] || '');
}

export function fade_lerp_color(colorStart, colorEnd, lerp) {
  // Parse color strings into RGB arrays
  const startRGB = hex2rgb(colorStart);
  const endRGB = hex2rgb(colorEnd);

  // Linearly interpolate each RGB component
  const lerpedRGB = startRGB.map((component, index) => {
      return Math.round(component + (endRGB[index] - component) * lerp);
  });

  // Convert the interpolated RGB values back to a hex color string
  return rgb2hex(lerpedRGB);
}

export function color_contrast(hexColor) {
    const rgb = hex2rgb(hexColor);
    const L = 0.2126 * rgb[0] / 255. + 0.7152 * rgb[1] / 255. + 0.0722 * rgb[2] / 255.;
    // console.info(L)
    return L > 0.790 ? "#000" : "#CCC";
}
