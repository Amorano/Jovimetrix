
float time = 0.1;
float sdBox( vec3 p, vec3 b )
{
  vec3 d = abs(p) - b;
  return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

float opS( float d1, float d2 )
{
    return max(-d1,d2);
}

 mat3 rotationMatrix(vec3 axis, float angle)
{
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat3(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c);
}

float length8(vec3 x)
{
 	return pow(dot(vec3(1.0, 2.0, 1.0), pow(abs(x), vec3(8.0))), 1.0/8.0)  ;
}

float length8(vec2 x)
{
 	return pow(dot(vec2(1.0, 1.0), pow(abs(x), vec2(8.0))), 1.0/8.0)  ;
}

float sdTorus88( vec3 p, vec2 t )
{
  vec2 q = vec2(length8(p.xy)-t.x,p.z);
  return length8(q)-t.y;
}

float nsin(float x)
{
    return sin(x) * 0.5 + 0.5;
}

float map(vec3 p)
{
    vec3 q = p;
    float rep = 0.01;
    vec3 c = vec3(rep);
    p.z = mod(p.z,c.z)-0.5*c.z;
    vec3 p_s;

    float bars = 1000.0;
    float inner = 1000.0;
    int sides = 6;
    float angle = 3.1415 * 0.33;
    float blockID = floor(q.z / rep) + nsin(iTime);

    for ( int i = 0; i < sides; i ++)
    {
        p_s = p * rotationMatrix(vec3(0.0, 0.0, 1.0), angle * float(i));
        float cutout = 10000.;
        vec2 line = vec2( nsin(q.z*33.0), 0.005 * nsin(10.0 * q.z));

        p_s = p_s + vec3(
            sin(blockID * 11.0)* 0.1 + 0.3 ,
            sin(q.z * sin(q.z+ iTime* 0.01)) * sin(p.z* 4.0) ,
            0.0);

        p_s = p_s * rotationMatrix(vec3(0.0, 0.0, 1.0), 1.1 *  iTime * 0.5  );
     	p_s = p_s * vec3(5.0, 1.0, 1.0);

        cutout = sdTorus88(p_s, line);

        inner = min(inner, cutout);

    }
    float result = inner;
    return result;
}


void getCamPos(inout vec3 ro, inout vec3 rd)
{
    ro.z = time;
}

 vec3 gradient(vec3 p, float t) {
			vec2 e = vec2(0., t);

			return normalize(
				vec3(
					map(p+e.yxx) - map(p-e.yxx),
					map(p+e.xyx) - map(p-e.xyx),
					map(p+e.xxy) - map(p-e.xxy)
				)
			);
		}


void main() {
	time = iTime * 0.5;
    vec2 _p = (fragCoord.xy - 0.5) * 2;
    vec3 ray = normalize(vec3(_p, 1.0));
    vec3 cam = vec3(0.0, 0.0, 0.0);
    bool hit = false;
    getCamPos(cam, ray);

    float depth = 0.1, d = 0.0, iter = 0.0;
    vec3 p;

    for( int i = 0; i < 80; i ++)
    {
    	p = depth * ray + cam;
        d = map(p);

        if (d < 0.0001) {
			hit = true;
            break;
        }
        if ( depth > 20.0)
            break;

        float ratio =  nsin(iTime * 10.1) * 0.01 + 0.03 + nsin(iTime)* 0.02;
		depth += d * ratio  ;
		iter++;

    }
    vec3 col = vec3(0.0);

    if(hit)
    	col = vec3(1.0 - iter / 40.0);

    col = pow(col, vec3(
        cos(floor(p.z * 300.0 )) * 0.15 + 0.15,
        0.95,
        sin(floor(p.z / 0.1)) * 0.15 + 0.15 ));

    fragColor = vec4(( col), 1.0);

}