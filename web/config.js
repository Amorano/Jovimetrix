import { api } from "../../../scripts/api.js";

function colorBox(data) {
    const box = document.createElement('text');
    box.value = data;
    box.setAttribute("data-coloris", "")
    // box.classList.add('color-box');
    box.style.width = '270px';
    box.style.height = '20px';
    box.style.backgroundColor = data;
    return box;
}

// Function to create a color box
function createColorBox(color) {

    const outerBox = document.createElement('div');
    outerBox.classList.add('outer-box');
    const page = document.createElement('p');
    outerBox.appendChild(page);

    var textBox = document.createTextNode(color[0]);
    page.appendChild(textBox)

    var box = colorBox(color[1].title)
    box.classList.add('color-box');
    box.innerHTML = 'TITLE'
    page.appendChild(box)

    box = colorBox(color[1].body)
    box.innerHTML = 'BODY'
    box.classList.add('color-box');
    page.appendChild(box);
    return outerBox;
}

export class Jovimetrics {
    async initialize() {
        const response = await api.fetchApi("/config/color", { cache: "no-store" });
	    const data = await response.json();
        const div = document.getElementById('colorContainer');

        Object.entries(data).forEach(entry => {
            const colorBox = createColorBox(entry);
            div.appendChild(colorBox);
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
