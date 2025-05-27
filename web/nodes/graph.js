/**/

import { app } from "../../../scripts/app.js"
import { apiJovimetrix, nodeAddDynamic } from "../util.js"

const _id = "GRAPH (JOV) 📈"
const _prefix = '❔'

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
    async init() {
        LGraphCanvas.link_type_colors['JOV_VG_0'] = "#A00";
        LGraphCanvas.link_type_colors['JOV_VG_1'] = "#0A0";
        LGraphCanvas.link_type_colors['JOV_VG_2'] = "#00A";
        LGraphCanvas.link_type_colors['JOV_VG_3'] = "#0AA";
        LGraphCanvas.link_type_colors['JOV_VG_4'] = "#AA0";
        LGraphCanvas.link_type_colors['JOV_VG_5'] = "#A0A";
        LGraphCanvas.link_type_colors['JOV_VG_6'] = "#000";
    },
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }

        await nodeAddDynamic(nodeType, _prefix);

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = await onNodeCreated?.apply(this, arguments);
            const self = this;
            const widget_reset = this.widgets.find(w => w.name == 'reset');
            widget_reset.callback = async() => {
                widget_reset.value = false;
                apiJovimetrix(self.id, "reset");
            }
            return me;
        }

        const onConnectionsChange = nodeType.prototype.onConnectionsChange
        nodeType.prototype.onConnectionsChange = function (slotType, slot, event, link_info) {
            const me = onConnectionsChange?.apply(this, arguments);
            if (!link_info || slot == this.inputs.length) {
                return;
            }
            let count = 0;
            for (let i = 0; i < this.inputs.length; i++) {
                const link_id = this.inputs[i].link;
                const link = app.graph.links[link_id];
                const nameParts = this.inputs[i].name.split('_');
                const isInteger = nameParts.length > 1 && !isNaN(nameParts[0]) && Number.isInteger(parseFloat(nameParts[0]));
                if (link && isInteger && nameParts[1].substring(0, _prefix.length) == _prefix) {
                //if(link && this.inputs[i].name.substring(0, _prefix.length) == _prefix) {
                    link.type = `JOV_VG_${count}`;
                    this.inputs[i].color_on = LGraphCanvas.link_type_colors[link.type];
                    count += 1;
                }
            }
            app.graph.setDirtyCanvas(true, true);
            return me;
        }
	}
})
