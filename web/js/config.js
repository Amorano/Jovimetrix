import { api } from "../../../../scripts/api.js";

function col_color(data) {
    var col = document.createElement('td');
    var box = document.createElement('input');
    box.classList.add('color');
    box.setAttribute("value", data)
    col.appendChild(box);
    return col;
}

function box_color(root, name, title, body,) {
    var row = document.createElement('tr');
    var col = document.createElement('td');
    col.innerHTML = name;
    row.appendChild(col);
    row.appendChild(col_color(title));
    row.appendChild(col_color(body));
    root.appendChild(row);
}

export class Jovimetrics {
    async initialize() {
        var response = await api.fetchApi("/config/raw", { cache: "no-store" });
	    const data = await response.json();

        const div = document.getElementById('configColor');
        const colors = Object.entries(data.color)
        colors.forEach(entry => {
            box_color(div, entry[0], entry[1].title, entry[1].body);
        });

        // now the rest which are untracked....
        var categories = [];
        response = await api.fetchApi("./../object_info", { cache: "no-store" });
	    const untracked = await response.json();

        Object.entries(untracked).forEach(entry => {
            var name = entry[0];
            if (colors.includes(name) == false) {
                categories = [new Set([categories, entry[1].category])];
                box_color(div, entry[0], '#7F7F7F', '#7F7F7F');
            }
        });

        Object.entries(untracked).forEach(entry => {
            var name = entry[0];
            if (colors.includes(name) == false) {
                categories = [new Set([categories, entry[1].category])];
                box_color(div, entry[0], '#7F7F7F', '#7F7F7F');
            }
        });

        jsColorPicker('input.color', {
            customBG: '#FFF',
            readOnly: true,
            size: 2,
            multipleInstances: false,
            //mode: 'HEX',
            init: function(elm, colors) {
              elm.style.backgroundColor = elm.value;
              elm.style.color = colors.rgbaMixCustom.luminance > 0.22 ? '#222' : '#ddd';
            },
            onchange: function(elm, val) {
                console.log("asd")
            },
        });
    }

	constructor() {
        (async () => {
            await this.initialize();
        })();

        /*
        document.addEventListener("dragover", (e) => {
            e.preventDefault();
        }, false);
        document.addEventListener("drop", (e) => {
            this.onDrop(e);
        });
        this.btnFix.addEventListener("click", (e) => {
            this.onFixClick(e);
        });*/
    }
}
