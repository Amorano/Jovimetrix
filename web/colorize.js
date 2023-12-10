/**
 * File: colorize.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js";
import { $el } from "../../../scripts/ui.js";
import { jovimetrix } from "./jovimetrix.js";

const ext = {
    name: "jovimetrix.colorize",

    async init() {
		//let visible = true;
		const showButton = $el("button.comfy-settings-btn", {
			textContent: "🎨",
			style: {
				left: "16px",
				cursor: "pointer",
				display: "unset",
			},
		});

		showButton.onclick = () => {
            jovimetrix.settings.show();
		};
		document.querySelector(".comfy-settings-btn").after(showButton);
	},

    async beforeRegisterNodeDef(nodeType, nodeData) {
        // console.debug(jovimetrix.CONFIG)
        if (jovimetrix.CONFIG.color === undefined || jovimetrix.CONFIG.color.length == 0) {
            return;
        }

        let found = jovimetrix.CONFIG.color[nodeData.type || nodeData.name];
        if (found === undefined && nodeData.category) {
            // console.debug(nodeData.type || nodeData.name)
            const categorySegments = nodeData.category.split('/');
            let k = categorySegments.join('/');

            while (k) {
                found = jovimetrix.CONFIG.color[k];
                if (found) {
                    break;
                }
                const lastSlashIndex = k.lastIndexOf('/');
                k = lastSlashIndex !== -1 ? k.substring(0, lastSlashIndex) : '';
            }
        }

        if (found) {
            // console.debug(nodeData);
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                if (nodeData.color === undefined) {
                    this['color'] = (found.title || "#7F7F7FDD")
                }

                if (nodeData.bgcolor === undefined) {
                    this['bgcolor'] = (found.body || "#7F7F7FDD")
                }

                /*
                // default, box, round, card
                if (nodeData.shape === undefined || nodeData.shape == false) {
                    this['_shape'] = nodeData._shape ? nodeData._shape : found.shape ?
                    found.shape in ['default', 'box', 'round', 'card'] : 'round';
                }*/
                // console.debug("jovi-colorized", this.title, this.color, this.bgcolor, this._shape);
                // result.serialize_widgets = true
                return result;
            }
        }
    },
}

app.registerExtension(ext);
