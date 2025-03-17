/**/

import { app } from "../../../scripts/app.js"

function arrayToObject(values, length, parseFn) {
    const result = {};
    for (let i = 0; i < length; i++) {
        result[i] = parseFn(values[i]);
    }
    return result;
}

export function domRenderTemplate(template, data) {
    for (const key in data) {
        if (Object.prototype.hasOwnProperty.call(data, key)) {
            const regex = new RegExp(`{{\\s*${key}\\s*}}`, 'g')
            template = template.replace(regex, data[key])
        }
    }
    return template
}

export function domFoldableToggle(elementId, symbolId) {
    const content = document.getElementById(elementId)
    const symbol = document.getElementById(symbolId)
    if (content.style.display == 'none' || content.style.display == '') {
        content.style.display = 'flex'
        symbol.innerHTML = '&#9661' // Down arrow
    } else {
        content.style.display = 'none'
        symbol.innerHTML = '&#9655' // Right arrow
    }
}

export function domInnerValueChange(node, pos, widget, value, event=undefined) {
    const type = widget.type.includes("INT") ? Number : parseFloat
    widget.value = arrayToObject(value, Object.keys(value).length, type);
    if (
        widget.options &&
        widget.options.property &&
        node.properties[widget.options.property] !== undefined
        ) {
            node.setProperty(widget.options.property, widget.value)
        }
    if (widget.callback) {

        widget.callback(widget.value, app.canvas, node, pos, event)
    }
}

export function domWidgetOffset(
    widget,
    ctx,
    node,
    widgetWidth,
    widgetY,
    height
  ) {
    const margin = 10
    const elRect = ctx.canvas.getBoundingClientRect()
    const transform = new DOMMatrix()
      .scaleSelf(
        elRect.width / ctx.canvas.width,
        elRect.height / ctx.canvas.height
      )
      .multiplySelf(ctx.getTransform())
      .translateSelf(0, widgetY + margin)

    const scale = new DOMMatrix().scaleSelf(transform.a, transform.d)
    Object.assign(widget.inputEl.style, {
      transformOrigin: '0 0',
      transform: scale,
      left: `${transform.e}px`,
      top: `${transform.d + transform.f}px`,
      width: `${widgetWidth}px`,
      height: `${(height || widget.parent?.inputHeight || 32) - margin}px`,
      position: 'absolute',
      background: !node.color ? '' : node.color,
      color: !node.color ? '' : 'white',
      zIndex: 5, //app.graph._nodes.indexOf(node),
    })
}

export function domEscapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;')
}

export function domShowModal(innerHTML, eventCallback, timeout=null) {
    return new Promise((resolve, reject) => {
        const modal = document.createElement("div");
        modal.className = "modal";
        modal.innerHTML = innerHTML;
        document.body.appendChild(modal);

        // center
        const modalContent = modal.querySelector(".jov-modal-content");
        modalContent.style.position = "absolute";
        modalContent.style.left = "50%";
        modalContent.style.top = "50%";
        modalContent.style.transform = "translate(-50%, -50%)";

        let timeoutId;

        const handleEvent = (event) => {
            const targetId = event.target.id;
            const result = eventCallback(targetId);

            if (result != null) {
                if (timeoutId) {
                    clearTimeout(timeoutId);
                    timeoutId = null;
                }
                modal.remove();
                resolve(result);
            }
        };
        modalContent.addEventListener("click", handleEvent);
        modalContent.addEventListener("dblclick", handleEvent);

        if (timeout) {
            timeout *= 1000;
            timeoutId = setTimeout(() => {
                modal.remove();
                reject(new Error("TIMEOUT"));
            }, timeout);
        }

        //setTimeout(() => {
        //    modal.dispatchEvent(new Event('tick'));
        //}, 1000);
    });
}

/*
* Parse a string "255,255,255,255" or a List[255,255,255,255] into hex
*/

export function colorLerpHex(colorStart, colorEnd, lerp) {
  // Parse color strings into RGB arrays
  const startRGB = colorHex2RGB(colorStart);
  const endRGB = colorHex2RGB(colorEnd);

  // Linearly interpolate each RGB component
  const lerpedRGB = startRGB.map((component, index) => {
      return Math.round(component + (endRGB[index] - component) * lerp);
  });

  // Convert the interpolated RGB values back to a hex color string
  return colorRGB2Hex(lerpedRGB);
}


