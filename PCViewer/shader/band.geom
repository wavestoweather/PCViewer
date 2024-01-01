#version 450
#extension GL_ARB_separate_shader_objects : enable

//resolution of the resulting spline
const int res = 28;
const int reductionRes = 10;
const float alpha = .5f;
const float centerWeight = .9;
const float innerPadding = .1;

layout(binding = 0) buffer StorageBuffer{
    float[] d;  //dicrete structure of values in LineBundles.hpp
}data;

// 4 vertices per-primitive -- 2 for the line (1,2) and 2 for adjacency (0,3)
layout (lines_adjacency) in;
layout (location = 0)in vec4 col[];
layout (location = 1)in vec4 pos[];
layout (location = 2)in uvec2 ids[];

// Standard fare for drawing lines
layout (triangle_strip, max_vertices = 2 * (res)) out;
layout (location = 0)out vec4 color;
layout (location = 1)out vec4 side; //has a 1 in x if top, 1 in y if bot, z component is band width
layout (location = 2)out vec4 haloColor;

float getT(in float t,in vec2 p0,in vec2 p1);

void addBand(vec2 lefTop, vec2 leftBot, vec2 rightTop, vec2 rightBot, vec2 prevLeftTop, vec2 prevLeftBot, vec2 nextRightTop, vec2 nextRightBot, int count, bool finish);
void addMainBand(vec2 lefTop, vec2 leftBot, vec2 rightTop, vec2 rightBot, vec2 prevLeftTop, vec2 prevLeftBot, vec2 nextRightTop, vec2 nextRightBot, int count);

void main (void) {
  vec2 centerA = vec2(gl_in[1].gl_Position.x, pos[1].y);
  vec2 topA = vec2(gl_in[1].gl_Position.x, pos[1].z);
  vec2 botA = vec2(gl_in[1].gl_Position.x, pos[1].x);
  vec2 centerB = vec2(gl_in[2].gl_Position.x, pos[2].y);
  vec2 topB = vec2(gl_in[2].gl_Position.x, pos[2].z);
  vec2 botB = vec2(gl_in[2].gl_Position.x, pos[2].x);
  vec2 innerX = vec2(mix(topA.x, topB.x, innerPadding), mix(topB.x, topA.x, innerPadding));
  vec2 innerATop = centerWeight * centerA + (1 - centerWeight) * topA;
  vec2 innerABot = centerWeight * centerA + (1 - centerWeight) * botA;
  vec2 innerBTop = centerWeight * centerB + (1 - centerWeight) * topB;
  vec2 innerBBot = centerWeight * centerB + (1 - centerWeight) * botB;
  innerATop.x = innerX.x;
  innerABot.x = innerX.x;
  innerBTop.x = innerX.y;
  innerBBot.x = innerX.y;
  vec2 prevTop = topA;
  vec2 prevBot = botA;
  vec2 nextTop = innerATop;
  vec2 nextBot = innerABot;
  prevTop.x -= .35 * (innerATop.x - topA.x);
  prevBot.x -= .35 * (innerATop.x - topA.x);
  nextTop.x += .35 * (innerATop.x - topA.x);
  nextBot.x += .35 * (innerATop.x - topA.x);

  //draw compression band
  addBand(topA, botA, innerATop, innerABot, prevTop, prevBot, nextTop, nextBot, reductionRes, false);

  //setup for main band
  prevTop = innerBTop;
  prevBot = innerBBot;
  nextTop = innerATop;
  nextBot = innerABot;
  prevTop.x -= innerBTop.x - innerATop.x;
  prevBot.x -= innerBTop.x - innerATop.x;
  nextTop.x += innerBTop.x - innerATop.x;
  nextBot.x += innerBTop.x - innerATop.x;
  addMainBand(innerATop, innerABot, innerBTop, innerBBot, prevTop, prevBot, nextTop, nextBot, res - 2 * reductionRes);

  //setup for the right decompression band
  prevTop = innerBTop;
  prevBot = innerBBot;
  nextTop = topB;
  nextBot = botB;
  prevTop.x -= .35 * (topB.x - innerBTop.x);
  prevBot.x -= .35 * (topB.x - innerBTop.x);
  nextTop.x += .35 * (topB.x - innerBTop.x);
  nextBot.x += .35 * (topB.x - innerBTop.x);
  addBand(innerBTop, innerBBot, topB, botB, prevTop, prevBot, nextTop, nextBot, reductionRes, true);

  //EndPrimitive();
}

float getT(in float t,in vec2 p0,in vec2 p1) {
   float a = pow((p1.x-p0.x), 2.0f) + pow((p1.y-p0.y), 2.0f);
   float b = pow(a, .5f);
   float c = pow(b,alpha);
   return c+t;
}

void addBand(vec2 leftTop, vec2 leftBot, vec2 rightTop, vec2 rightBot, vec2 prevLeftTop, vec2 prevLeftBot, vec2 nextRightTop, vec2 nextRightBot, int count, bool finish){
  float t0a = 0;
  float t1a = getT(t0a, prevLeftTop, leftTop);
  float t2a = getT(t1a, leftTop, rightTop);
  float t3a = getT(t2a, rightTop, nextRightTop);
  float t0b = 0;
  float t1b = getT(t0b, prevLeftBot, leftBot);
  float t2b = getT(t1b, leftBot, rightBot);
  float t3b = getT(t2b, rightBot, nextRightBot);
  float distLeft = leftTop.y - leftBot.y, distRight = rightTop.y - rightBot.y;

  int offset = floatBitsToInt(data.d[9 + 2 * ids[1].x]);
  offset += int(ids[1].y) * floatBitsToInt(data.d[9 + 2 * ids[2].x + 1]) + int(ids[2].y); //adding the correct group number to get the final offset
  float colorAlpha = data.d[offset];
  float haloWidth = data.d[4];

  //double spline, always adding 2 vertx per iteration (1 in top spline, one in bot)
  int maxCount = count;
  if(finish) --maxCount;
  //for(float t = t1; t<t2+.00001f; t += (t2-t1)/res){
  for(int i = 0; i < count; ++i){
    vec2 a = leftTop, b = rightTop, prev_vtx = prevLeftTop, next_vtx = nextRightTop;
    float t0 = t0a, t1 = t1a, t2 = t2a, t3 = t3a;
    float t = mix(t1, t2, i / float(maxCount));
      vec2 A1 = (t1-t)/(t1-t0)*prev_vtx.xy + (t-t0)/(t1-t0)*a.xy;
      vec2 A2 = (t2-t)/(t2-t1)*a.xy + (t-t1)/(t2-t1)*b.xy;
      vec2 A3 = (t3-t)/(t3-t2)*b.xy + (t-t2)/(t3-t2)*next_vtx.xy;
  
      vec2 B1 = (t2-t)/(t2-t0)*A1 + (t-t0)/(t2-t0)*A2;
      vec2 B2 = (t3-t)/(t3-t1)*A2 + (t-t1)/(t3-t1)*A3;
  
      gl_Position = vec4((t2-t)/(t2-t1)*B1 + (t-t1)/(t2-t1)*B2,0,1);
      color = col[1];
    color.a *= colorAlpha;
    side = vec4(1,0,mix(distLeft, distRight, float(i) / (count - 1)), haloWidth); //indicates top vertex
    haloColor.x = data.d[5];
    haloColor.y = data.d[6];
    haloColor.z = data.d[7];
    haloColor.w = data.d[8];

      EmitVertex();
    a = leftBot, b = rightBot, prev_vtx = prevLeftBot, next_vtx = nextRightBot;
    t0 = t0b, t1 = t1b, t2 = t2b, t3 = t3b;
    t = mix(t1, t2, i / float(maxCount));
      A1 = (t1-t)/(t1-t0)*prev_vtx.xy + (t-t0)/(t1-t0)*a.xy;
      A2 = (t2-t)/(t2-t1)*a.xy + (t-t1)/(t2-t1)*b.xy;
      A3 = (t3-t)/(t3-t2)*b.xy + (t-t2)/(t3-t2)*next_vtx.xy;
  
      B1 = (t2-t)/(t2-t0)*A1 + (t-t0)/(t2-t0)*A2;
      B2 = (t3-t)/(t3-t1)*A2 + (t-t1)/(t3-t1)*A3;
  
      gl_Position = vec4((t2-t)/(t2-t1)*B1 + (t-t1)/(t2-t1)*B2,0,1);
      color = col[1];
    color.a *= colorAlpha;
    side = vec4(0, 1, mix(distLeft, distRight, float(i) / (count - 1)), haloWidth); //indicates bot vertex
    haloColor.x = data.d[5];
    haloColor.y = data.d[6];
    haloColor.z = data.d[7];
    haloColor.w = data.d[8];

      EmitVertex();

  }
}

void addMainBand(vec2 leftTop, vec2 leftBot, vec2 rightTop, vec2 rightBot, vec2 prevLeftTop, vec2 prevLeftBot, vec2 nextRightTop, vec2 nextRightBot, int count){
  float t0 = 0;
  float t1 = getT(t0, mix(prevLeftTop, prevLeftBot, .5), mix(leftTop, leftBot, .5));
  float t2 = getT(t1, mix(leftTop, leftBot, .5), mix(rightTop, rightBot, .5));
  float t3 = getT(t2, mix(rightTop, rightBot, .5), mix(nextRightTop, nextRightBot, .5));
  float distLeft = leftTop.y - leftBot.y, distRight = rightTop.y - rightBot.y;

  int offset = floatBitsToInt(data.d[9 + 2 * ids[1].x]);
  offset += int(ids[1].y) * floatBitsToInt(data.d[9 + 2 * ids[2].x + 1]) + int(ids[2].y); //adding the correct group number to get the final offset
  float colorAlpha = data.d[offset];
  float haloWidth = data.d[4];
  vec2 center;
  vec2 perp;

  //double spline, always adding 2 vertx per iteration (1 in top spline, one in bot)
  int maxCount = count;
  for(int i = 0; i < count; ++i){
    vec2 a = mix(leftBot, leftTop, .5), b = mix(rightTop, rightBot, .5), prev_vtx = mix(prevLeftTop, prevLeftBot, .5), next_vtx = mix(nextRightTop, nextRightBot, .5);
    float t = mix(t1, t2, i / float(maxCount));
      vec2 A1 = (t1-t)/(t1-t0)*prev_vtx.xy + (t-t0)/(t1-t0)*a.xy;
      vec2 A2 = (t2-t)/(t2-t1)*a.xy + (t-t1)/(t2-t1)*b.xy;
      vec2 A3 = (t3-t)/(t3-t2)*b.xy + (t-t2)/(t3-t2)*next_vtx.xy;
  
      vec2 B1 = (t2-t)/(t2-t0)*A1 + (t-t0)/(t2-t0)*A2;
      vec2 B2 = (t3-t)/(t3-t1)*A2 + (t-t1)/(t3-t1)*A3;

    center = (t2-t)/(t2-t1)*B1 + (t-t1)/(t2-t1)*B2;
    t -= t1;
    t /= t2 - t1;
    perp =  .5 * ((-prev_vtx + b) + 2 * t * (2 * prev_vtx - 5 * a + 4 * b - next_vtx) + 3 * t * t * (-prev_vtx + 3*a - 3*b + next_vtx));
    perp = vec2(-perp.y, perp.x);
    perp = normalize(perp);
    perp = vec2(0,1);
    //perp = vec2(0,1);
      color = col[1];
    color.a *= colorAlpha;
    side = vec4(1,0,mix(distLeft, distRight, float(i) / (count - 1)), haloWidth); //indicates top vertex
    haloColor.x = data.d[5];
    haloColor.y = data.d[6];
    haloColor.z = data.d[7];
    haloColor.w = data.d[8];
    gl_Position.xy = center + .5 * side.z * perp;

      EmitVertex();

    color = col[1];
    color.a *= colorAlpha;
    side = vec4(0,1,mix(distLeft, distRight, float(i) / (count - 1)), haloWidth); //indicates top vertex
    haloColor.x = data.d[5];
    haloColor.y = data.d[6];
    haloColor.z = data.d[7];
    haloColor.w = data.d[8];
    gl_Position.xy = center - .5 * side.z * perp;

      EmitVertex();
  }
}