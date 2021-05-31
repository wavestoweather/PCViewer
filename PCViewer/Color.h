#pragma once

typedef struct {
	double r;       // a fraction between 0 and 1
	double g;       // a fraction between 0 and 1
	double b;       // a fraction between 0 and 1
} rgb;

typedef struct {
	double h;       // angle in degrees
	double s;       // a fraction between 0 and 1
	double v;       // a fraction between 0 and 1
} hsv;

typedef struct {
	double h;
	double s;
	double l;
} hsl;

static hsv   rgb2hsv(rgb in);
static rgb   hsv2rgb(const hsv& in);
static rgb	 hsl2rgb(const hsl& in);

rgb hsv2rgb(const hsv& in)
{
	double      hh, p, q, t, ff;
	long        i;
	rgb         out;

	hh = in.h;
	if (hh >= 360.0) hh = 0.0;
	hh /= 60.0;
	i = (long)hh;
	ff = hh - i;
	p = in.v * (1.0 - in.s);
	q = in.v * (1.0 - (in.s * ff));
	t = in.v * (1.0 - (in.s * (1.0 - ff)));

	switch (i) {
	case 0:
		out.r = in.v;
		out.g = t;
		out.b = p;
		break;
	case 1:
		out.r = q;
		out.g = in.v;
		out.b = p;
		break;
	case 2:
		out.r = p;
		out.g = in.v;
		out.b = t;
		break;

	case 3:
		out.r = p;
		out.g = q;
		out.b = in.v;
		break;
	case 4:
		out.r = t;
		out.g = p;
		out.b = in.v;
		break;
	case 5:
	default:
		out.r = in.v;
		out.g = p;
		out.b = q;
		break;
	}
	return out;
}

rgb hsl2rgb(const hsl& in) {
	double      hh, c, x, m;
	long        i;
	rgb         out;

	hh = in.h;
	if (hh >= 360.0) hh = 0.0;
	hh /= 60.0;
	i = (long)hh;
	
	c = (1 - abs(2 * in.l - 1)) * in.s;
	x = c * (1 - abs((long)(in.h / 60) % 2 - 1));
	m = in.l - c / 2;

	switch (i) {
	case 0:
		out.r = c+m;
		out.g = x+m;
		out.b = m;
		break;
	case 1:
		out.r = x+m;
		out.g = c+m;
		out.b = m;
		break;
	case 2:
		out.r = m;
		out.g = c+m;
		out.b = x+m;
		break;

	case 3:
		out.r = m;
		out.g = x+m;
		out.b = c+m;
		break;
	case 4:
		out.r = x+m;
		out.g = m;
		out.b = c+m;
		break;
	case 5:
	default:
		out.r = c+m;
		out.g = m;
		out.b = x+m;
		break;
	}
	return out;
}