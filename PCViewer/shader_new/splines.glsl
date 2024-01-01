const float alpha = .5;
float get_t(float t, in vec2 p0, in vec2 p1){
    if(alpha == 0)
        return 1 + t;
    float a = pow((p1.x-p0.x), 2.0f) + pow((p1.y-p0.y), 2.0f);
    float b = pow(a, .5f);
    float c = pow(b,alpha);
    return c+t;
}

vec2 get_spline_pos(vec2 p0, vec2 p1, vec2 p2, vec2 p3, float t){
    float t0 = 0;
    float t1 = get_t(t0, p0, p1);
    float t2 = get_t(t1, p1, p2);
    float t3 = get_t(t2, p2, p3);

    t = mix(t1, t2, t);
    vec2 a1 = ( t1-t )/( t1-t0 )*p0 + ( t-t0 )/( t1-t0 )*p1;
    vec2 a2 = ( t2-t )/( t2-t1 )*p1 + ( t-t1 )/( t2-t1 )*p2;
    vec2 a3 = ( t3-t )/( t3-t2 )*p2 + ( t-t2 )/( t3-t2 )*p3;
    vec2 b1 = ( t2-t )/( t2-t0 )*a1 + ( t-t0 )/( t2-t0 )*a2;
    vec2 b2 = ( t3-t )/( t3-t1 )*a2 + ( t-t1 )/( t3-t1 )*a3;
    return ( t2-t )/( t2-t1 )*b1 + ( t-t1 )/( t2-t1 )*b2;
}