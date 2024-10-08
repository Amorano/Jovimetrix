/**
 * File: core_help.js
 * Project: Jovimetrix
 * The original code to show the help was based on Mel Massadian's mtb node help extension:
 *
 *    https://github.com/melMass/comfy_mtb/
 */

import { app } from '../../../scripts/app.js'

let PANEL;

// help now is 100% dynamically built as it is used, no more requests out
const CACHE_DOCUMENTATION = {};

if (!window.jovimetrixEvents) {
    window.jovimetrixEvents = new EventTarget();
}
const jovimetrixEvents = window.jovimetrixEvents;

const JOV_HELP_URL = "./api/jovimetrix/doc";
const JOV_HOME = "./api/jovimetrix";
const JOV_WEBWIKI_URL = "https://github.com/Amorano/Jovimetrix/wiki/Z.-REFERENCE#";

async function load_help(name, absolute=false) {
    let url = JOV_HOME;
    if (!absolute) {
        if (name in CACHE_DOCUMENTATION) {
            return CACHE_DOCUMENTATION[name];
        }
        url = `${JOV_HELP_URL}/${name}`;
    }

    // Check if data is already cached
    const result = fetch(url,
        { cache: "no-store" }
    )
        .then(response => {
            if (!response.ok) {
                return `Failed to load documentation: ${name}\n\n${response}`
            }
            return response.text();
        })
        .then(data => {
            // Cache the fetched data
            if (data.startsWith("unknown")) {
                data = `
                    <div align="center">
                        <h3>${data}</h3>
                        <h4>SELECT A NODE TO SEE HELP</h4>
                        <h4>JOVIMETRIX 🔺🟩🔵 NODES ALL HAVE HELP</h4>
                    </div>
                `;
            };
            CACHE_DOCUMENTATION[name] = data;
            return CACHE_DOCUMENTATION[name];
        })
        .catch(error => {
            console.error('Error:', error);
            return `Failed to load documentation: ${name}\n\n${error}`
        });
    return result;
}

app.extensionManager.registerSidebarTab({
    id: "jovimetrix.sidebar.help",
    icon: "pi pi-money-bill",
    title: "Jovimetrix Lore",
    tooltip: "In panel help for most ComfyUI extension packs.\nJOVIMETRIX 🔺🟩🔵",
    type: "custom",
    render: async (el) => {
        PANEL = el;
        PANEL.innerHTML = await load_help("home", true);
    }
});

// Listen for the custom event
jovimetrixEvents.addEventListener('jovimetrixHelpRequested', async (event) => {
    // const node = `${event.detail.class}/${event.detail.name}`;
    if (PANEL) {
        PANEL.innerHTML = await load_help(event.detail.name);
    }
});

app.registerExtension({
    name: "jovimetrix.help",
    async setup() {
        const onSelectionChange = app.canvas.onSelectionChange;
        app.canvas.onSelectionChange = function(selectedNodes) {
            const me = onSelectionChange?.apply(this);
            if (selectedNodes && Object.keys(selectedNodes).length > 0) {
                const firstNodeKey = Object.keys(selectedNodes)[0];
                const firstNode = selectedNodes[firstNodeKey];
                const data = {
                    class: firstNode?.getNickname?.() || "unknown",
                    name: firstNode.type
                }
                const event = new CustomEvent('jovimetrixHelpRequested', { detail: data });
                jovimetrixEvents.dispatchEvent(event);
            }
            return me;
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData?.category?.startsWith("JOVIMETRIX")) {
            return;
        }

        // MENU HELP!
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            const me = getExtraMenuOptions?.apply(this, arguments);
            if (this.title.endsWith('🧙🏽‍♀️')) {
                return me;
            }

            const widget_tooltip = (this.widgets || [])
                .find(widget => widget.type == 'JTOOLTIP');

            if (widget_tooltip) {
                const tips = widget_tooltip.options.default || {};
                const url = tips['_'];
                const help_menu = [{
                    content: `HELP: ${this.title}`,
                    callback: () => {
                        LiteGraph.closeAllContextMenus();
                        window.open(`${JOV_WEBWIKI_URL}${url}`, '_blank');
                        this.setDirtyCanvas(true, true);
                    }
                }];
                if (help_menu.length) {
                    options.push(...help_menu, null);
                }
            }
            return me;
        }
    }
});
