/**
 * File: core.js
 *
 * ASYNC
    init
    setup
    registerCustomNodes
    nodeCreated
    beforeRegisterNodeDef
    getCustomWidgets
    afterConfigureGraph
    refreshComboInNodes

    NON-ASYNC
    onNodeOutputsUpdated
    beforeRegisterVueAppNodeDefs
    loadedGraphNode
 * Project: Jovimetrix
 */

import { app } from "../../../scripts/app.js"

app.registerExtension({
    name: "jovimetrix",
    async init() {
        const styleTagId = 'jovimetrix-stylesheet';
        let styleTag = document.getElementById(styleTagId);
        if (styleTag) {
            return;
        }

        /*
        document.head.appendChild(Object.assign(document.createElement('link'), {
            id: styleTagId,
            rel: 'stylesheet',
            type: 'text/css',
            href: 'https://cdn.jsdelivr.net/npm/@simonwep/pickr/dist/themes/classic.min.css'
        }));
        document.head.appendChild(Object.assign(document.createElement('link'), {
            id: styleTagId,
            rel: 'stylesheet',
            type: 'text/css',
            href: 'https://cdn.jsdelivr.net/npm/@simonwep/pickr/dist/themes/monolith.min.css'
        }));
        document.head.appendChild(Object.assign(document.createElement('link'), {
            id: styleTagId,
            rel: 'stylesheet',
            type: 'text/css',
            href: 'https://cdn.jsdelivr.net/npm/@simonwep/pickr/dist/themes/nano.min.css'
        }));

        document.head.appendChild(Object.assign(document.createElement('script'), {
            src: "https://cdn.jsdelivr.net/npm/@simonwep/pickr/dist/pickr.min.js"
        }));

        document.head.appendChild(Object.assign(document.createElement('script'), {
            src: "https://unpkg.com/vanilla-picker@2.12.1/dist/vanilla-picker.min.js"
        }));
        */

        document.head.appendChild(Object.assign(document.createElement('script'), {
            src: "https://cdn.jsdelivr.net/npm/@jaames/iro@5"
        }));

        document.head.appendChild(Object.assign(document.createElement('link'), {
            id: styleTagId,
            rel: 'stylesheet',
            type: 'text/css',
            href: 'extensions/Jovimetrix/jovimetrix.css'
        }));
	}
});