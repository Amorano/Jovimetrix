/**
 * File: util_help.js
 * Project: Jovimetrix
 * Thanks to MelMass https://github.com/melMass
 * for all the support functions
 * Original code from: https://github.com/melMass/comfy_mtb/pull/160
 */

const dataCache = {};

const create_documentation_stylesheet = () => {
    const tag = 'mtb-documentation-stylesheet'

    let styleTag = document.head.querySelector(tag)

    if (!styleTag) {
    styleTag = document.createElement('style')
    styleTag.type = 'text/css'
    styleTag.id = tag

    styleTag.innerHTML = `
    .documentation-popup {
        background: var(--bg-color);
        position: absolute;
            color: var(--fg-color);
            font: 13px monospace;
            line-height: 1.25em;
            padding: 2px;
            border-radius: 3px;
            pointer-events: "inherit";
            z-index: 1111111111111111;
        overflow:scroll;
    }

        .documentation-popup img {
            max-width: 100%;
        }
        .documentation-popup table {
                border-collapse: collapse;
            border: 1px var(--border-color) solid;
        }
        .documentation-popup th,
        .documentation-popup td {
            border: 1px var(--border-color) solid;
        }
    .documentation-popup th {
        background-color: var(--comfy-input-bg);
        }
        `
    document.head.appendChild(styleTag)
    }
}

let documentationConverter

/** Add documentation widget to the selected node */
export const addDocumentation = (
    nodeData,
    nodeType,
    opts = { icon_size: 14, icon_margin: 3 },
) => {
    if (!documentationConverter) {
        documentationConverter = new showdown.Converter({
            tables: true,
            strikethrough: true,
            emoji: true,
            ghCodeBlocks: true,
            tasklists: true,
            ghMentions: true,
            smoothLivePreview: true,
            simplifiedAutoLink: true,
            parseImgDimensions: true,
            openLinksInNewWindow: true,
        })
    }

    opts = opts || {}
    const iconSize = opts.icon_size ? opts.icon_size : 14
    const iconMargin = opts.icon_margin ? opts.icon_margin : 3
    let docElement = null
    let offsetX = 0
    let offsetY = 0
    if (!nodeData.description) {
        return
    }

    const drawFg = nodeType.prototype.onDrawForeground
    nodeType.prototype.onDrawForeground = function (ctx, canvas) {
        const r = drawFg ? drawFg.apply(this, arguments) : undefined
        if (this.flags.collapsed) return r
        // icon position
        const x = this.size[0] - iconSize - iconMargin
        if (this.show_doc && docElement === null) {
            create_documentation_stylesheet();
            docElement = document.createElement('div');
            docElement.classList.add('documentation-popup');
            if (!(nodeData.name in dataCache)) {
                // Load data from URL asynchronously if it ends with .md
                if (nodeData.description.endsWith('.md')) {
                    // Check if data is already cached
                    // Fetch data from URL
                    fetch(nodeData.description)
                        .then(response => {
                            if (!response.ok) {
                                docElement.innerHTML = `Failed to load documentation\n\n${response}`
                            }
                            return response.text();
                        })
                        .then(data => {
                            // Cache the fetched data
                            dataCache[nodeData.name] = documentationConverter.makeHtml(data);
                            docElement.innerHTML = dataCache[nodeData.name];
                        })
                        .catch(error => {
                            docElement.innerHTML = `Failed to load documentation\n\n${error}`
                            console.error('Error:', error);
                        });
                } else {
                    // If description does not end with .md, set data directly
                    dataCache[nodeData.name] = documentationConverter.makeHtml(dataCache[nodeData.name]);
                    docElement.innerHTML = dataCache[nodeData.name];
                }
            } else {
                docElement.innerHTML = dataCache[nodeData.name];
            }

            // resize handle
            const resizeHandle = document.createElement('div');
            resizeHandle.style.width = '10px';
            resizeHandle.style.height = '10px';
            resizeHandle.style.background = 'gray';
            resizeHandle.style.position = 'absolute';
            resizeHandle.style.bottom = '0';
            resizeHandle.style.right = '0';
            resizeHandle.style.cursor = 'se-resize';

            // TODO: fix resize logic
            docElement.appendChild(resizeHandle)
            let isResizing = false;
            let startX, startY, startWidth, startHeight;

            resizeHandle.addEventListener('mousedown', function (e) {
                e.stopPropagation()
                isResizing = true;
                startX = e.clientX;
                startY = e.clientY;
                startWidth = parseInt(
                    document.defaultView.getComputedStyle(docElement).width,
                    10,
                );
                startHeight = parseInt(
                    document.defaultView.getComputedStyle(docElement).height,
                    10,
                );
            })

            document.addEventListener('mousemove', function (e) {
                if (!isResizing) return;
                const newWidth = startWidth + e.clientX - startX;
                const newHeight = startHeight + e.clientY - startY;
                offsetX += newWidth - startWidth;
                offsetY += newHeight - startHeight;
                startWidth = newWidth;
                startHeight = newHeight;
            })

            document.addEventListener('mouseup', function () {
                isResizing = false;
            })
            document.body.appendChild(docElement)
        } else if (!this.show_doc && docElement !== null) {
            docElement.parentNode.removeChild(docElement)
            docElement = null
        }

        if (this.show_doc && docElement !== null) {
            const rect = ctx.canvas.getBoundingClientRect()
            const scaleX = rect.width / ctx.canvas.width
            const scaleY = rect.height / ctx.canvas.height
            const transform = new DOMMatrix()
            .scaleSelf(scaleX, scaleY)
            .multiplySelf(ctx.getTransform())
            .translateSelf(this.size[0] * scaleX, 0)
            .translateSelf(10, -32)

            const scale = new DOMMatrix().scaleSelf(transform.a, transform.d);
            Object.assign(docElement.style, {
                transformOrigin: '0 0',
                transform: scale,
                left: `${transform.a + transform.e}px`,
                top: `${transform.d + transform.f}px`,
                width: `${this.size[0] * 2}px`,
                height: `${this.size[1] || this.parent?.inputHeight || 32}px`,
            })
        }

        ctx.save()
        ctx.translate(x, iconSize - 34) // Position the icon on the canvas
        ctx.scale(iconSize / 32, iconSize / 32) // Scale the icon to the desired size
        ctx.strokeStyle = 'rgba(255,255,255,0.3)'
        ctx.lineCap = 'round'
        ctx.lineJoin = 'round'
        ctx.lineWidth = 2.4
        // ctx.stroke(questionMark);
        ctx.font = '36px monospace'
        ctx.fillText('?', 0, 24)
        ctx.restore()
        return r
    }

    const mouseDown = nodeType.prototype.onMouseDown
    nodeType.prototype.onMouseDown = function (e, localPos, canvas) {
        const r = mouseDown ? mouseDown.apply(this, arguments) : undefined
        const iconX = this.size[0] - iconSize - iconMargin
        const iconY = iconSize - 34
        if (
            localPos[0] > iconX &&
            localPos[0] < iconX + iconSize &&
            localPos[1] > iconY &&
            localPos[1] < iconY + iconSize
        ) {
            // Pencil icon was clicked, open the editor
            // this.openEditorDialog();
            if (this.show_doc === undefined) {
                this.show_doc = true
            } else {
                this.show_doc = !this.show_doc
            }
            return true; // Return true to indicate the event was handled
        }
        return r;
    }
}
