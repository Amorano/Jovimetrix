import { api } from "../../../../scripts/api.js";

function col_color(name, data, attr) {
    var col = document.createElement('td');
    var box = document.createElement('input');
    box.classList.add('color');
    if (data === undefined) {
        data = '#7F7F7F';
    }
    box.setAttribute("value", data)
    box.setAttribute("jovi", name)
    box.setAttribute("part", attr)
    col.appendChild(box);
    return col;
}

function box_color(root, name, title, body,) {
    var row = document.createElement('tr');
    var col = document.createElement('td');
    col.innerHTML = name;
    row.appendChild(col);
    row.appendChild(col_color(name, title, 'title'));
    row.appendChild(col_color(name, body, 'body'));
    root.appendChild(row);
}

export class Jovimetrics {
    async initialize() {

        var response = await api.fetchApi("/config/raw", { cache: "no-store" });
        const JOV_CONFIG = await response.json();

        response = await api.fetchApi("./../object_info", { cache: "no-store" });
        const NODE_LIST = await response.json();

        const div = document.getElementById('configColor');
        if (JOV_CONFIG.color === undefined) {
            return;
        }

        var existing = [];
        const COLORS = Object.entries(JOV_CONFIG.color)
        COLORS.forEach(entry => {
            box_color(div, entry[0], entry[1].title, entry[1].body);
            existing.push(entry[0])
        });

        // now the rest which are untracked....
        var nodes = Object.entries(NODE_LIST);

        var categories = [];
        nodes.forEach(entry => {
            var name = entry[0];
            var cat = entry[1].category;
            console.log(cat)
            if (existing.includes(name) == false) {
                box_color(div, entry[0], '#7F7F7F', '#7F7F7F');
            }
            if (categories.includes(cat) == false) {
                // console.log(cat, categories)
                categories.push(cat);
            }
        });

        categories.sort(function (a, b) {
            return a.toLowerCase().localeCompare(b.toLowerCase());
        });

        Object.entries(categories).forEach(entry => {
            if (existing.includes(entry[1]) == false) {
                box_color(div, entry[1], '#3F3F3F', '#3F3F3F');
            }
        });

        jsColorPicker('input.color', {
            customBG: '#FFF',
            readOnly: true,
            size: 2,
            multipleInstances: false,
            //mode: 'HEX',
            init: function(elm, rgb) {
              elm.style.backgroundColor = elm.value;
              elm.style.color = rgb.rgbaMixCustom.luminance > 0.22 ? '#222' : '#ddd';
            },
            convertCallback: function(data, type) {
                const name = this.patch.attributes.jovi.value;
                const part = this.patch.attributes.part.value;
                var body = {}
                body[name] = {"part": part, "color": data.HEX}
                const res = api.fetchApi("/config", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify(body),
                });
                COLORS[name] = data.HEX;
            },
        });
    }

	constructor() {
        (async () => {
            await this.initialize();
        })();
    }
}
