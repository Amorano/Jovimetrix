
export const template_color_block = `
<tr>
    <td style="color: white; text-align: right">{{ name }}</td>
    <td><input class="jov-color" type="text" name="{{ name }}" value="title" color="{{title}}" part="title"></input></td>
    <td><input class="jov-color" type="text" name="{{ name }}" value="body" color="{{body}}" part="body"></input></td>
    <td><button type="button" onclick="color_clear('{{name}}')"></button></td>
</tr>
`
