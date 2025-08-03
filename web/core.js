/**
    ASYNC
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
  */

import { app } from "../../scripts/app.js"

app.registerExtension({
    name: "jovimetrix",
    async init() {
        const styleTagId = 'jovimetrix-stylesheet';
        let styleTag = document.getElementById(styleTagId);
        if (styleTag) {
            return;
        }

        document.head.appendChild(Object.assign(document.createElement('script'), {
            src: "https://cdn.jsdelivr.net/npm/@jaames/iro@5"
        }));

        document.head.appendChild(Object.assign(document.createElement('link'), {
            id: styleTagId,
            rel: 'stylesheet',
            type: 'text/css',
            href: 'extensions/jovimetrix/jovi_metrix.css'
        }));
	}
});