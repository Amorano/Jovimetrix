/**
 * File: util_color.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
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
            console.debug("node_color_get", colors, found, node)
            colors.jov_set_color = 1;
            colors.jov_set_bgcolor = 1;
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
    // if we made it here, we could have "temp" colors. reset.
    if (node?.jov_set_color == 1)
    {
        node.color = ""
    }
    if (node?.jov_set_bgcolor == 1)
    {
        node.bgcolor = ""
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
    app.graph.setDirtyCanvas(true, true);
}

export function node_color_all() {
    app.graph._nodes.forEach((node) => {
        this.node_color_reset(node, false);
    })
    app.graph.setDirtyCanvas(true, true);
    // console.info("JOVI] all nodes color refreshed")
}

export function convert_hex(color) {
    if (!color.HEX.includes("NAN")) {
        return '#' + color.HEX + ((color.alpha * 255) | 1 << 8).toString(16).slice(1).toUpperCase()
    }
    return "#353535FF"
}

export function hexToRgb(hex) {
  //console.info(hex)
  hex = hex.replace(/^#/, '');

  // Parse the hex value into RGB components
  const bigint = parseInt(hex, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;

  return [r, g, b];
}

export function rgbToHex(rgb) {
  // Convert RGB components to a hex color string
  const [r, g, b] = rgb;
  return `#${(1 << 24 | r << 16 | g << 8 | b).toString(16).slice(1)}`;
}

export function fade_lerp_color(colorStart, colorEnd, lerp) {
  // Parse color strings into RGB arrays
  const startRGB = hexToRgb(colorStart);
  const endRGB = hexToRgb(colorEnd);

  // Linearly interpolate each RGB component
  const lerpedRGB = startRGB.map((component, index) => {
      return Math.round(component + (endRGB[index] - component) * lerp);
  });

  // Convert the interpolated RGB values back to a hex color string
  return rgbToHex(lerpedRGB);
}

export function color_contrast(hexColor) {
    // Remove the # symbol if it exists
    if (hexColor.startsWith("#")) {
        hexColor = hexColor.slice(1);
    }

    // Expand short hex code to full hex code
    if (hexColor.length === 3) {
        hexColor = hexColor.split('').map(char => char + char).join('');
    }

    // Convert the hex values to decimal (base 10) integers
    const r = parseInt(hexColor.slice(0, 2), 16) / 255;
    const g = parseInt(hexColor.slice(2, 4), 16) / 255;
    const b = parseInt(hexColor.slice(4, 6), 16) / 255;

    // Calculate the relative luminance
    const L = 0.2126 * r + 0.7152 * g + 0.0722 * b;

    // Use the contrast ratio to determine the text color
    return L > 0.210 ? "#000000" : "#999999";
}
