import { app } from "/scripts/app.js";
import * as util from '../core/util.js'

const NodeType = {
    SET: "jovimetrix.node.set",
    GET: "jovimetrix.node.get",
};

let Memory = {};
// "SET-GET (JOV) 🟰"

app.registerExtension({
	name: NodeType.SET,
	registerCustomNodes() {
		class SetNode {
			defaultVisibility = true;
			serialize_widgets = true;
			constructor() {
				if (!this.properties) {
					this.properties = {
						"previousName": ""
					};
				}
				this.properties.showOutputText = SetNode.defaultVisibility;
				const node = this;
				this.addWidget("text", "VARIABLE", '', (s, t, u, v, x) => {
                        this.update();
                        this.properties.previousName = this.widgets[0].value;
                    },
                    {})
				this.addInput("*", "*");

				this.onConnectionsChange = function(slotType, slot, isChangeConnect, link_info, output) {
					if (slotType === util.SlotType.Input && isChangeConnect === util.ChangeType.Disconnect) {
						this.inputs[slot].type = '*';
						this.inputs[slot].name = '*';
					}

					if (slotType === util.SlotType.Input && isChangeConnect === util.ChangeType.Connect) {
						const fromNode = node.graph._nodes.find((otherNode) => otherNode.id == link_info.origin_id);
						const type = fromNode.outputs[link_info.origin_slot].type;
						this.inputs[0].type = type;
						this.inputs[0].name = type;
					}
					this.update();
				}

				this.clone = function () {
					const cloned = SetNode.prototype.clone.apply(this);
					cloned.inputs[0].name = '*';
					cloned.inputs[0].type = '*';
					cloned.properties.previousName = '';
					cloned.size = cloned.computeSize();
					return cloned;
				};

				this.update = function() {
					if (node.graph) {
						this.findGetters(node.graph).forEach((getter) => {
							getter.setType(this.inputs[0].type);
						});
						if (this.widgets[0].value) {
							this.findGetters(node.graph, true).forEach((getter) => {
								getter.setName(this.widgets[0].value)
							});
						}

						const allGetters = node.graph._nodes.filter((otherNode) => otherNode.type == NodeType.GET);
						allGetters.forEach((otherNode) => {
							if (otherNode.setComboValues) {
								otherNode.setComboValues();
							}
						})
					}
				}

				this.findGetters = function(graph, checkForPreviousName) {
					const name = checkForPreviousName ? this.properties.previousName : this.widgets[0].value;
					return graph._nodes.filter((otherNode) => {
						if (otherNode.type == NodeType.GET && otherNode.widgets[0].value === name && name != '') {
							return true;
						}
						return false;
					})
				}
				this.isVirtualNode = true;
			}

			onRemoved() {
				const allGetters = this.graph._nodes.filter((otherNode) => otherNode.type == NodeType.GET);
				allGetters.forEach((otherNode) => {
					if (otherNode.setComboValues) {
						otherNode.setComboValues([this]);
					}
				})
			}
		}

		LiteGraph.registerNodeType(
			NodeType.SET,
			Object.assign(SetNode, {
				title: "Set",
			})
		);
		SetNode.category = "JOVIMETRIX 🔺🟩🔵/FLOW";
	},
});


app.registerExtension({
	name: NodeType.GET,
	registerCustomNodes() {
		class GetNode {

			defaultVisibility = true;
			serialize_widgets = true;

			constructor() {
				if (!this.properties) {
					this.properties = {};
				}
				this.properties.showOutputText = GetNode.defaultVisibility;

				const node = this;
				this.addWidget("combo", "VARIABLE", "", (e) => {
						this.onRename();
					},
					{
						values: () => {
                            const setterNodes = graph._nodes.filter((otherNode) => otherNode.type == NodeType.SET);
                            return setterNodes.map((otherNode) => otherNode.widgets[0].value).sort();
                        }
					}
				)
				this.addOutput("*", '*');

				this.onConnectionsChange = function(slotType, slot, isChangeConnect, link_info, output) {
					this.validateLinks();
				}

				this.setName = function(name) {
					node.widgets[0].value = name;
					node.onRename();
					node.serialize();
				}

				this.onRename = function() {
					const setter = this.findSetter(node.graph);
					if (setter) {
						this.setType(setter.inputs[0].type);
					} else {
						this.setType('*');
					}
				}

				this.clone = function () {
					const cloned = GetNode.prototype.clone.apply(this);
					cloned.size = cloned.computeSize();
					//this.update();
					return cloned;
				};

				this.validateLinks = function() {
					if (this.outputs[0].type != '*' && this.outputs[0].links) {
						this.outputs[0].links.forEach((linkId) => {
							const link = node.graph.links[linkId];
							if (link && link.type != this.outputs[0].type && link.type != '*') {
								node.graph.removeLink(linkId)
							}
						})
					}
				}

				this.setType = function(type) {
					this.outputs[0].name = type;
					this.outputs[0].type = type;
					this.validateLinks();
				}

				this.findSetter = function(graph) {
					const name = this.widgets[0].value;
					return graph._nodes.find((otherNode) => {
						if (otherNode.type == NodeType.SET && otherNode.widgets[0].value === name && name != '') {
							return true;
						}
						return false;
					})
				}
				this.isVirtualNode = true;
			}

			getInputLink(slot) {
				const setter = this.findSetter(this.graph);
				if (setter) {
					const slot_info = setter.inputs[slot];
                    const link = this.graph.links[ slot_info.link ];
                    return link;
				} else {
					throw new Error("No setter found for " + this.widgets[0].value + "(" + this.type + ")");
				}

			}

			onAdded(graph) {

			}

		}

		LiteGraph.registerNodeType(
			NodeType.GET,
			Object.assign(GetNode, {
				title: "Get",
			})
		);

		GetNode.category = "JOVIMETRIX 🔺🟩🔵/FLOW";
	},
});