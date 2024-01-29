// Will return a value of 1 if the 'x' is < 'value'
float Less(float x, float value)
{
	return 1.0 - step(value, x);
}

// Will return a value of 1 if the 'x' is >= 'lower' && < 'upper'
float Between(float x, float  lower, float upper)
{
    return step(lower, x) * (1.0 - step(upper, x));
}

//	Will return a value of 1 if 'x' is >= value
float GEqual(float x, float value)
{
    return step(value, x);
}

void main()
{
    float brightness = 1.25;
	vec2 uv = fragCoord;
    //uv.y = uv.y;
    //uv = uv * 0.05;

    vec2 uvStep;
    uvStep.x = uv.x / (1.0 / iResolution.x);
    uvStep.x = mod(uvStep.x, 3.0);
    uvStep.y = uv.y / (1.0 / iResolution.y);
    uvStep.y = mod(uvStep.y, 3.0);

    vec4 newColour = texture(iChannel0, uv);

    if (iUser1 > 0 && iUser1 < 2.0) {
        newColour.r = newColour.r * Less(uvStep.x, 1.0);
        newColour.g = newColour.g * Between(uvStep.x, 1.0, 2.0);
        newColour.b = newColour.b * GEqual(uvStep.x, 2.0);
    }

    if (iUser1 == 2.0) {
        newColour.r = newColour.r * Less(uvStep.y, 1.0);
        newColour.g = newColour.g * Between(uvStep.y, 1.0, 2.0);
        newColour.b = newColour.b * GEqual(uvStep.y, 2.0);
    }

    if (iUser1 == 0) {
        newColour.r = newColour.r * step(1.0, (Less(uvStep.x, 1.0) + Less(uvStep.y, 1.0)));
        newColour.g = newColour.g * step(1.0, (Between(uvStep.x, 1.0, 2.0) + Between(uvStep.y, 1.0, 2.0)));
        newColour.b = newColour.b * step(1.0, (GEqual(uvStep.x, 2.0) + GEqual(uvStep.y, 2.0)));
    }

	fragColor = newColour * brightness;
}