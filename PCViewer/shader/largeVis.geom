#version 450
#extension GL_ARB_separate_shader_objects : enable

//resolution of the resulting spline

const float alpha = .5f;

// 4 vertices per-primitive -- 2 for the line (1,2) and 2 for adjacency (0,3)
layout (points) in;
layout (location = 0)in vec4 col[];
layout (location = 1)in vec4 aPosBPos[];

// Standard fare for drawing lines
layout (line_strip, max_vertices = 2) out;
layout (location = 0)out vec4 color;

float getT(in float t,in vec2 p0,in vec2 p1);

void main (void) {
  //// The two vertices adjacent to the line that you are currently processing
  //vec4 prev_vtx = gl_in [0].gl_Position;
  //vec4 next_vtx = gl_in [3].gl_Position;
  //vec4 a = gl_in[1].gl_Position;
  //vec4 b = gl_in[2].gl_Position;
  //mat4 points = transpose(mat4(prev_vtx,a,b,next_vtx));
//
  ////calculating the x position of prev and next_vtx new to avoid that two points lie on the same spot at the beginning of the line
  //prev_vtx.x = 2*a.x - b.x;
  //next_vtx.x = 2*b.x - a.x;
//
  //float t0 = 0;
  //float t1 = getT(t0,prev_vtx.xy,a.xy);
  //float t2 = getT(t1,a.xy,b.xy);
  //float t3 = getT(t2,b.xy,next_vtx.xy);
//
  //for(float t = t1; t<t2+.00001f; t += (t2-t1)/res){
    //vec2 A1 = (t1-t)/(t1-t0)*prev_vtx.xy + (t-t0)/(t1-t0)*a.xy;
    //vec2 A2 = (t2-t)/(t2-t1)*a.xy + (t-t1)/(t2-t1)*b.xy;
    //vec2 A3 = (t3-t)/(t3-t2)*b.xy + (t-t2)/(t3-t2)*next_vtx.xy;
    //
    //vec2 B1 = (t2-t)/(t2-t0)*A1 + (t-t0)/(t2-t0)*A2;
    //vec2 B2 = (t3-t)/(t3-t1)*A2 + (t-t1)/(t3-t1)*A3;
    //
    //gl_Position = vec4((t2-t)/(t2-t1)*B1 + (t-t1)/(t2-t1)*B2,0,1);
    //color = col[0];
//
    //EmitVertex();
  //}
  if(col[0].a == 0)
    return;           //skip lines which can not be seen and dont emit a vertex

  color = col[0];
  gl_Position = vec4(aPosBPos[0].xy, 0, 1);
  EmitVertex();
  color = col[0];
  gl_Position = vec4(aPosBPos[0].zw, 0, 1);
  EmitVertex();

  //EndPrimitive();
}

float getT(in float t,in vec2 p0,in vec2 p1) {
   float a = pow((p1.x-p0.x), 2.0f) + pow((p1.y-p0.y), 2.0f);
   float b = pow(a, .5f);
   float c = pow(b,alpha);
   return c+t;
}