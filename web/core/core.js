/**
 * File: core.js
 *
 * Project: Jovimetrix
 */

import { app } from "../../../scripts/app.js"

app.registerExtension({
    name: "jovimetrix",
    init() {
        const styleTagId = 'jovimetrix-stylesheet';
        let styleTag = document.getElementById(styleTagId);
        if (styleTag) {
            return;
        }
        document.head.appendChild(Object.assign(document.createElement('link'), {
            id: styleTagId,
            rel: 'stylesheet',
            type: 'text/css',
            href: 'extensions/Jovimetrix/jovimetrix.css'
        }));
	}
});