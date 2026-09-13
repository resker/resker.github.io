// Initech coffee mug in WebGPU. No libraries: geometry is generated here,
// shading is the WGSL below. Drag to orbit; the steam rises faster while the
// pointer is over the cup.

const canvas = document.querySelector('canvas');
const fallback = document.querySelector('p[hidden]');

const common = /* wgsl */ `
struct Scene {
  viewProj: mat4x4f,
  eye: vec3f,
  ambient: f32,
  light: vec3f,
  time: f32,
  steam: vec4f, // rgb tint, w strength
};
struct Instance {
  model: mat4x4f,
  color: vec4f,
  params: vec4f, // x: material (1 cup rings, 2 coffee)
};
@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
// Cheap value noise, used by the crema and the steam.
fn hash(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453); }
fn noise(p: vec2f) -> f32 {
  let i = floor(p); let f = fract(p); let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2f(1, 0)), u.x), mix(hash(i + vec2f(0, 1)), hash(i + vec2f(1, 1)), u.x), u.y);
}
`;

const litShader = common + /* wgsl */ `
fn hash2(p: vec2f) -> vec2f { return vec2f(hash(p), hash(p + vec2f(19.3, 7.7))); }
// Distance to the nearest of a jittered grid of points: packed-cell texture.
fn cells(p: vec2f) -> f32 {
  let cell = floor(p);
  let f = fract(p);
  var best = 2.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let n = vec2f(f32(x), f32(y));
      best = min(best, length(n + hash2(cell + n) - f));
    }
  }
  return best;
}
struct Out {
  @builtin(position) pos: vec4f,
  @location(0) worldPos: vec3f,
  @location(1) normal: vec3f,
  @location(2) localPos: vec3f,
  @location(3) @interpolate(flat) id: u32,
};

@vertex fn vs(@location(0) p: vec3f, @location(1) n: vec3f,
              @builtin(instance_index) id: u32) -> Out {
  let inst = instances[id];
  let world = inst.model * vec4f(p, 1);
  var o: Out;
  o.pos = scene.viewProj * world;
  o.worldPos = world.xyz;
  o.normal = normalize((inst.model * vec4f(n, 0)).xyz);
  o.localPos = p;
  o.id = id;
  return o;
}

@fragment fn fs(i: Out) -> @location(0) vec4f {
  let inst = instances[i.id];
  var base = inst.color.rgb;
  var gloss = 64.0;
  var specAmt = 0.35;
  let material = inst.params.x;
  let r = length(i.localPos.xz);
  let e = fwidth(i.localPos.y); // derivative must be taken in uniform control flow
  if (material > 0.5 && material < 1.5) {
    // Two dark rings around the outside of the cup, like the artwork.
    // Anti-aliased with fwidth; 'outside' = normal points away from the axis.
    let y = i.localPos.y;
    let ring1 = smoothstep(1.05 - e, 1.05 + e, y) - smoothstep(1.085 - e, 1.085 + e, y);
    let ring2 = smoothstep(1.13 - e, 1.13 + e, y) - smoothstep(1.165 - e, 1.165 + e, y);
    let outside = step(0.0, dot(normalize(i.normal.xz), normalize(i.localPos.xz)));
    base = mix(base, vec3f(0.12, 0.12, 0.13), clamp(ring1 + ring2, 0.0, 1.0) * outside);
  } else if (material > 1.5) {
    // Black coffee with crema. The body is dark and glossy with a faint,
    // streaked film of crema; foam collects in a band at the wall and in a
    // patch or two bulging in from it, rendered as packed micro-foam cells.
    let q = i.localPos.xz;
    let angle = atan2(q.y, q.x);
    let wall = 0.85;
    let hazelnut = vec3f(0.6, 0.38, 0.19);
    let amber = vec3f(0.36, 0.19, 0.07);
    // Organic streaks: domain-warped noise, stretched on one axis.
    let warp = vec2f(noise(q * 3.0 + 7.0), noise(q * 3.0 + 19.0)) - 0.5;
    let qs = vec2f(q.x, q.y * 2.4) + warp * 2.2;
    let swirl = noise(qs * 3.0) * 0.55 + noise(qs * 7.0 + 3.0) * 0.3 + noise(q * 14.0 + warp * 4.0) * 0.15;
    // Micro-foam cells.
    let d = cells(q * 170.0);
    let d2 = cells(q * 60.0 + 3.0);
    let cellTone = 0.9 + 0.28 * (1.0 - smoothstep(0.12, 0.5, d)) - 0.22 * smoothstep(0.45, 0.72, d)
                 + 0.1 * (1.0 - smoothstep(0.15, 0.45, d2));
    var foamCol = mix(amber, hazelnut, smoothstep(0.32, 0.7, swirl) * 0.8 + 0.1) * cellTone;
    let fleck = smoothstep(0.76, 0.86, noise(q * 55.0 + warp * 5.0)) * 0.5 + smoothstep(0.8, 0.9, noise(q * 110.0 + 11.0)) * 0.5;
    foamCol = mix(foamCol, amber * 0.65, fleck * 0.55);
    // Foam coverage: a band at the wall whose width wanders, plus patches.
    let bandW = 0.02 + noise(vec2f(angle * 3.0, 2.3)) * 0.05;
    let band = smoothstep(wall - bandW - 0.015, wall - bandW + 0.015, r);
    let blob = noise(q * 2.6 + 5.0) + (noise(q * 24.0) - 0.5) * 0.12;
    let blobs = smoothstep(0.69, 0.73, blob) * smoothstep(0.5, 0.72, r);
    let foam = clamp(band + blobs, 0.0, 1.0);
    // Body: dark coffee under a thin streaked film of crema.
    let film = smoothstep(0.35, 0.8, swirl) * 0.28 + 0.06;
    let body = mix(base, amber * 1.1, film);
    base = mix(body, foamCol, foam);
    // Coffee is glossy; foam is matte.
    gloss = mix(18.0, 8.0, foam); specAmt = mix(0.4, 0.1, foam);
  }
  let n = normalize(i.normal);
  let l = normalize(scene.light);
  let v = normalize(scene.eye - i.worldPos);
  let h = normalize(l + v);
  let diff = max(dot(n, l), 0.0);
  let spec = pow(max(dot(n, h), 0.0), gloss) * specAmt;
  let fill = max(dot(n, normalize(vec3f(-0.5, 0.3, -0.7))), 0.0) * 0.25;
  let lit = base * (scene.ambient + diff * 0.65 + fill) + vec3f(spec);
  return vec4f(lit, 1.0);
}`;

// Steam: a ray-marched volume above the cup. The density field is a rising,
// drifting plume envelope shaped by ridged noise, which gives the thin,
// curling sheets real vapour shows; Beer-Lambert absorption accumulates it
// through depth. Drawn as the front faces of a box; each fragment marches a
// ray through it.
const steamShader = common + /* wgsl */ `
fn hash3(p: vec3f) -> f32 { return fract(sin(dot(p, vec3f(127.1, 311.7, 74.7))) * 43758.5453); }
fn noise3(p: vec3f) -> f32 {
  let i = floor(p); let f = fract(p); let u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(mix(hash3(i), hash3(i + vec3f(1, 0, 0)), u.x), mix(hash3(i + vec3f(0, 1, 0)), hash3(i + vec3f(1, 1, 0)), u.x), u.y),
    mix(mix(hash3(i + vec3f(0, 0, 1)), hash3(i + vec3f(1, 0, 1)), u.x), mix(hash3(i + vec3f(0, 1, 1)), hash3(i + vec3f(1, 1, 1)), u.x), u.y),
    u.z);
}
// Ridged noise: thin bright sheets where the noise crosses its midpoint.
fn ridged(p: vec3f) -> f32 {
  let a = 1.0 - abs(2.0 * noise3(p) - 1.0);
  let b = 1.0 - abs(2.0 * noise3(p * 2.1 + 3.7) - 1.0);
  let c = 1.0 - abs(2.0 * noise3(p * 4.3 + 9.1) - 1.0);
  return (a + b * 0.45 + c * 0.15) / 1.6;
}

const BASE: f32 = 1.2;    // just above the coffee
const HEIGHT: f32 = 2.0;

fn density(p: vec3f) -> f32 {
  let hn = (p.y - BASE) / HEIGHT;
  if (hn < 0.0 || hn > 1.0) { return 0.0; }
  let time = scene.time;
  // Plume envelope: a soft column whose axis wanders and leans with height,
  // widening as it rises; density fades in above the surface and decays.
  let axis = vec2f(sin(time * 0.08) * 0.12 + hn * sin(time * 0.13) * 0.35,
                   cos(time * 0.07) * 0.12 + hn * cos(time * 0.11) * 0.3);
  let d = length(p.xz - axis);
  let rad = 0.18 + hn * 0.42;
  let env = exp(-(d * d) / (rad * rad) * 2.2) * smoothstep(0.0, 0.05, hn) * exp(-2.4 * hn) * (1.0 - smoothstep(0.45, 0.85, hn));
  if (env < 0.003) { return 0.0; }
  // Structure: ridged noise in coordinates that rise with time, domain-warped
  // by slower noise so the sheets fold; the warp grows with height as the
  // plume goes from laminar to turbulent.
  let q = vec3f(p.x, p.y - time * 0.24, p.z) * 2.2;
  let warp = vec3f(noise3(q * 0.5 + time * 0.07), noise3(q * 0.5 + 9.0), noise3(q * 0.5 + 17.0 - time * 0.06)) - 0.5;
  let qq = q + warp * (1.0 + hn * 2.8);
  let f = ridged(qq);
  return env * smoothstep(0.58, 0.95, f) * 2.4;
}

struct Out {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
};

@vertex fn vs(@location(0) p: vec3f, @location(1) n: vec3f) -> Out {
  var o: Out;
  o.world = p;
  o.pos = scene.viewProj * vec4f(p, 1);
  return o;
}

@fragment fn fs(i: Out) -> @location(0) vec4f {
  let ro = scene.eye;
  let rd = normalize(i.world - ro);
  // Slab intersection with the steam box.
  let bmin = vec3f(-1.1, BASE, -1.1);
  let bmax = vec3f(1.1, BASE + HEIGHT, 1.1);
  let inv = 1.0 / rd;
  let t1 = (bmin - ro) * inv;
  let t2 = (bmax - ro) * inv;
  let tn = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
  let tf = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
  if (tf <= max(tn, 0.0)) { discard; }
  // The cup wall hides anything below the rim unless the ray comes in
  // through the opening: find where the ray crosses rim height.
  let RIM = 1.38;
  let tRim = (RIM - ro.y) / rd.y;
  let atRim = ro + rd * tRim;
  let throughOpening = tRim > 0.0 && length(atRim.xz) < 0.86;
  // Key light for self-shadowing; a backlight behind the plume for the
  // bright, forward-scattered edges of backlit vapour.
  let L = normalize(scene.light);
  let back = normalize(vec3f(-rd.x, 0.35, -rd.z));
  let cosBack = dot(rd, back);
  let g = 0.55;
  let hg = (1.0 - g * g) / (4.0 * 3.14159 * pow(1.0 + g * g - 2.0 * g * cosBack, 1.5));
  let steps = 36.0;
  let ds = (tf - max(tn, 0.0)) / steps;
  // Jitter the start per pixel to trade banding for fine grain.
  var t = max(tn, 0.0) + ds * fract(sin(dot(i.pos.xy, vec2f(12.9898, 78.233))) * 43758.5453);
  var trans = 1.0;
  var lum = 0.0;
  for (var k = 0; k < 36; k++) {
    let p = ro + rd * t;
    var dens = density(p);
    if (p.y < RIM && !throughOpening) { dens = 0.0; }
    if (dens > 0.002) {
      let a = 1.0 - exp(-dens * ds * 5.0);
      // Self-shadowing: a short march toward the key light.
      var occ = 0.0;
      for (var j = 1; j <= 3; j++) { occ += density(p + L * (0.12 * f32(j))); }
      let shadow = exp(-occ * 0.55);
      // Ambient sky-ish fill + shadowed key + forward-scattered backlight,
      // which brightens the thin edges most.
      let thin = 1.0 - min(dens * 0.6, 1.0);
      let shade = 0.35 + 0.5 * shadow + hg * 2.2 * thin;
      lum += trans * a * shade;
      trans *= 1.0 - a;
      if (trans < 0.03) { break; }
    }
    t += ds;
  }
  let alpha = (1.0 - trans) * scene.steam.w;
  let tint = scene.steam.xyz;
  return vec4f(tint * lum * scene.steam.w, alpha); // premultiplied
}`;

// ---- geometry -------------------------------------------------------------

// Surface of revolution around Y from a (radius, height) profile.
function lathe(profile, segments = 64) {
  const pos = [], nrm = [], idx = [];
  const rows = profile.length;
  for (let i = 0; i < rows; i++) {
    const [r, y] = profile[i];
    const [r0, y0] = profile[Math.max(i - 1, 0)];
    const [r1, y1] = profile[Math.min(i + 1, rows - 1)];
    // Normal is perpendicular to the profile tangent, pointing outward.
    let tx = r1 - r0, ty = y1 - y0;
    const len = Math.hypot(tx, ty) || 1;
    tx /= len; ty /= len;
    for (let j = 0; j <= segments; j++) {
      const a = (j / segments) * Math.PI * 2, c = Math.cos(a), s = Math.sin(a);
      pos.push(r * c, y, r * s);
      nrm.push(ty * c, -tx, ty * s);
    }
  }
  for (let i = 0; i < rows - 1; i++) {
    for (let j = 0; j < segments; j++) {
      const a = i * (segments + 1) + j, b = a + segments + 1;
      idx.push(a, b, a + 1, a + 1, b, b + 1);
    }
  }
  return { pos, nrm, idx };
}

// Tube swept along a polyline in the XY plane (the handle). `tube` may be a
// function of position along the path (0..1) for a varying radius.
function sweep(path, tubeSpec, ring = 16) {
  const pos = [], nrm = [], idx = [];
  const tubeAt = typeof tubeSpec === 'function' ? tubeSpec : () => tubeSpec;
  for (let i = 0; i < path.length; i++) {
    const [px, py] = path[i];
    const tube = tubeAt(i / (path.length - 1));
    const [ax, ay] = path[Math.max(i - 1, 0)], [bx, by] = path[Math.min(i + 1, path.length - 1)];
    let tx = bx - ax, ty = by - ay;
    const len = Math.hypot(tx, ty) || 1;
    tx /= len; ty /= len;
    const nx = -ty, ny = tx; // in-plane normal; binormal is +Z
    for (let j = 0; j <= ring; j++) {
      const b = (j / ring) * Math.PI * 2, cb = Math.cos(b), sb = Math.sin(b);
      pos.push(px + nx * cb * tube, py + ny * cb * tube, sb * tube);
      nrm.push(nx * cb, ny * cb, sb);
    }
  }
  for (let i = 0; i < path.length - 1; i++) {
    for (let j = 0; j < ring; j++) {
      const a = i * (ring + 1) + j, b = a + ring + 1;
      idx.push(a, a + 1, b, a + 1, b + 1, b);
    }
  }
  return { pos, nrm, idx };
}

// Box for the steam volume: front faces only are drawn; the shader marches
// through the box from there.
function box(min, max) {
  const c = [[min[0], min[1], min[2]], [max[0], min[1], min[2]], [max[0], max[1], min[2]], [min[0], max[1], min[2]],
             [min[0], min[1], max[2]], [max[0], min[1], max[2]], [max[0], max[1], max[2]], [min[0], max[1], max[2]]];
  const faces = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [0, 4, 7, 3], [1, 2, 6, 5]];
  const pos = [], nrm = [], idx = [];
  for (const f of faces) {
    const b = pos.length / 3;
    for (const v of f) { pos.push(...c[v]); nrm.push(0, 1, 0); }
    idx.push(b, b + 1, b + 2, b, b + 2, b + 3);
  }
  return { pos, nrm, idx };
}

// Catmull-Rom spline through 2D points, sampled evenly per segment.
function spline(points, per = 8) {
  const out = [];
  for (let i = 0; i < points.length - 1; i++) {
    const p0 = points[Math.max(i - 1, 0)], p1 = points[i], p2 = points[i + 1], p3 = points[Math.min(i + 2, points.length - 1)];
    for (let j = 0; j < per; j++) {
      const t = j / per, t2 = t * t, t3 = t2 * t;
      out.push([0, 1].map((k) => 0.5 * ((2 * p1[k]) + (-p0[k] + p2[k]) * t +
        (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2 + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3)));
    }
  }
  out.push(points.at(-1));
  return out;
}

// Cup: a smooth bowl. Outside from the foot up to the rim, then the inside
// back down. The rim is left sharp by splining the two halves separately.
const outerWall = spline([
  [0.38, 0.0], [0.46, 0.03], [0.58, 0.13], [0.72, 0.36], [0.84, 0.7],
  [0.905, 1.02], [0.93, 1.3], [0.93, 1.355],
]);
// Rounded lip: a semicircle from the outer wall over to the inner wall, as
// glaze rounds a real rim.
const lip = [];
for (let i = 0; i <= 10; i++) {
  const a = (i / 10) * Math.PI;
  lip.push([0.9 + Math.cos(a) * 0.03, 1.355 + Math.sin(a) * 0.03]);
}
const innerWall = spline([
  [0.87, 1.355], [0.862, 1.28], [0.82, 0.95], [0.74, 0.62], [0.6, 0.36], [0.38, 0.22], [0.0, 0.2],
]);
const cupProfile = [[0.0, 0.0], ...outerWall, ...lip.slice(1, -1), ...innerWall];
function cupOuterRadius(y) {
  for (let i = 1; i < outerWall.length; i++) {
    const [r0, y0] = outerWall[i - 1], [r1, y1] = outerWall[i];
    if (y <= y1) return r0 + (r1 - r0) * ((y - y0) / (y1 - y0 || 1));
  }
  return outerWall.at(-1)[0];
}
// Outward normal of the cup wall at a point (x, y, z) on or near it.
function cupOuterNormal(x, y, z) {
  let i = 1;
  while (i < outerWall.length - 1 && outerWall[i][1] < y) i++;
  const [r0, y0] = outerWall[i - 1], [r1, y1] = outerWall[i];
  const [nr, ny] = norm([y1 - y0, -(r1 - r0), 0]);
  const a = Math.atan2(z, x);
  return [nr * Math.cos(a), ny, nr * Math.sin(a)];
}

// Saucer: foot ring underneath, central well with a ridge that seats the cup,
// a gently rising dish, and a rolled rim. About 1.5x the cup's rim.
// Underside: a shallow recess in the centre, a foot ring, then a smooth
// curve up to the rim. Top: rolled rim, dish curving down, the seating well
// with its ridge. Splined in two halves so the rim stays crisp.
const saucerUnder = spline([
  [0.0, -0.05], [0.3, -0.05], [0.42, -0.07], [0.5, -0.1], [0.58, -0.09], [0.66, -0.05],
  [0.9, -0.02], [1.15, 0.03], [1.36, 0.11], [1.42, 0.16],
]);
const saucerTop = spline([
  [1.41, 0.19], [1.34, 0.18], [1.12, 0.11], [0.88, 0.06], [0.78, 0.07], [0.72, 0.08], [0.69, 0.04], [0.6, 0.02], [0.0, 0.02],
]);
const saucerProfile = [...saucerUnder, ...saucerTop];
const coffeeLevel = 1.18;
const coffeeProfile = [[0.85, coffeeLevel], [0.0, coffeeLevel]]; // right-to-left so the normal faces up

// Handle: one smooth cubic Bezier that leaves the wall just below the rings,
// loops outward, and re-enters at mid-cup. Both ends sit inside the wall.
function handlePath() {
  const top = 0.98, bottom = 0.5, inset = 0.1;
  const p0 = [cupOuterRadius(top) - inset, top], p3 = [cupOuterRadius(bottom) - inset, bottom];
  // Leaves the wall rising a little, arcs over, and comes down to the
  // lower attachment; the upper root is near the handle's highest point.
  const p1 = [p0[0] + 0.5, top + 0.24], p2 = [p3[0] + 0.68, bottom - 0.12];
  const path = [];
  for (let i = 0; i <= 48; i++) {
    const t = i / 48, u = 1 - t;
    path.push([0, 1].map((k) => u * u * u * p0[k] + 3 * u * u * t * p1[k] + 3 * u * t * t * p2[k] + t * t * t * p3[k]));
  }
  return path;
}

// Blend the handle's normals into the cup wall's over the fillet zone so the
// shading runs continuously across the joint instead of showing a seam.
function blendRoots(mesh, zone = 0.2) {
  const ring = 17, rows = mesh.pos.length / 3 / ring;
  for (let i = 0; i < rows; i++) {
    const t = i / (rows - 1), end = Math.min(t, 1 - t) / zone;
    if (end >= 1) continue;
    const w = Math.pow(1 - end, 2);
    for (let j = 0; j < ring; j++) {
      const k = (i * ring + j) * 3;
      const wall = cupOuterNormal(mesh.pos[k], mesh.pos[k + 1], mesh.pos[k + 2]);
      const n = norm([
        mesh.nrm[k] * (1 - w) + wall[0] * w,
        mesh.nrm[k + 1] * (1 - w) + wall[1] * w,
        mesh.nrm[k + 2] * (1 - w) + wall[2] * w,
      ]);
      mesh.nrm[k] = n[0]; mesh.nrm[k + 1] = n[1]; mesh.nrm[k + 2] = n[2];
    }
  }
  return mesh;
}

// ---- tiny matrix helpers (column-major, like WGSL) ------------------------

const mat = {
  identity: () => new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
  multiply(a, b) {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
    return o;
  },
  translate(x, y, z) { const m = mat.identity(); m[12] = x; m[13] = y; m[14] = z; return m; },
  perspective(fov, aspect, near, far) {
    const f = 1 / Math.tan(fov / 2), m = new Float32Array(16);
    m[0] = f / aspect; m[5] = f; m[10] = far / (near - far); m[11] = -1;
    m[14] = (near * far) / (near - far);
    return m;
  },
  lookAt(eye, target, up) {
    const z = norm(sub(eye, target)), x = norm(cross(up, z)), y = cross(z, x);
    return new Float32Array([
      x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
      -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
    ]);
  },
};
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

// ---- setup ------------------------------------------------------------------

async function main() {
  const adapter = await navigator.gpu?.requestAdapter();
  const device = await adapter?.requestDevice();
  if (!device) { canvas.hidden = true; fallback.hidden = false; return; }

  const ctx = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'premultiplied' });

  const upload = (m, floats) => {
    const v = new Float32Array(floats);
    m.vbuf = device.createBuffer({ size: v.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(m.vbuf, 0, v);
    const ix = new Uint32Array(m.idx);
    m.ibuf = device.createBuffer({ size: ix.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(m.ibuf, 0, ix);
    m.count = ix.length;
    return m;
  };
  const meshes = {
    cup: lathe(cupProfile),
    saucer: lathe(saucerProfile),
    coffee: lathe(coffeeProfile, 48),
    // Radius flares toward each end and blends into a short fillet where it
    // meets the body, as a luted, glazed handle does.
    handle: blendRoots(sweep(handlePath(), (t) => {
      // A modest flare only: a convex 'fillet' bump reads as a lump on the
      // wall from inside the loop, so the blend is left to the normals.
      const end = Math.min(t, 1 - t) / 0.14; // 0 at the wall, 1 a little way out
      const flare = 1 + 0.25 * Math.pow(1 - Math.min(end, 1), 2);
      return 0.085 * flare;
    })),
  };
  for (const m of Object.values(meshes)) {
    const v = [];
    for (let i = 0; i < m.pos.length; i += 3) v.push(...m.pos.slice(i, i + 3), ...m.nrm.slice(i, i + 3));
    upload(m, v);
  }
  const steamBox = box([-1.1, coffeeLevel + 0.02, -1.1], [1.1, coffeeLevel + 2.02, 1.1]);
  { const v = []; for (let i = 0; i < steamBox.pos.length; i += 3) v.push(...steamBox.pos.slice(i, i + 3), ...steamBox.nrm.slice(i, i + 3)); upload(steamBox, v); }

  // Instance table: 0 cup, 1 saucer, 2 coffee, 3 handle.
  const INST_FLOATS = 24; // mat4 (16) + color (4) + params (4)
  const instData = new Float32Array(4 * INST_FLOATS);
  const instBuf = device.createBuffer({ size: instData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const setInstance = (i, model, color, params = [0, 0, 0, 0]) => {
    instData.set(model, i * INST_FLOATS);
    instData.set(color, i * INST_FLOATS + 16);
    instData.set(params, i * INST_FLOATS + 20);
  };
  const ceramic = [0.93, 0.93, 0.94, 1];
  setInstance(0, mat.identity(), ceramic, [1, 0, 0, 0]);
  setInstance(1, mat.identity(), [0.88, 0.88, 0.89, 1]);
  setInstance(2, mat.identity(), [0.24, 0.13, 0.05, 1], [2, 0, 0, 0]);
  setInstance(3, mat.identity(), ceramic);
  device.queue.writeBuffer(instBuf, 0, instData);

  const sceneData = new Float32Array(16 + 4 + 4 + 4);
  const sceneBuf = device.createBuffer({ size: sceneData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const layout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
  ] });
  const bindGroup = device.createBindGroup({ layout, entries: [
    { binding: 0, resource: { buffer: sceneBuf } },
    { binding: 1, resource: { buffer: instBuf } },
  ] });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const litModule = device.createShaderModule({ code: litShader });
  const opaque = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: { module: litModule, entryPoint: 'vs', buffers: [{ arrayStride: 24, attributes: [
      { shaderLocation: 0, offset: 0, format: 'float32x3' },
      { shaderLocation: 1, offset: 12, format: 'float32x3' },
    ] }] },
    fragment: { module: litModule, entryPoint: 'fs', targets: [{ format }] },
    primitive: { cullMode: 'none' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    multisample: { count: 4 },
  });
  const steamModule = device.createShaderModule({ code: steamShader });
  const translucent = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: { module: steamModule, entryPoint: 'vs', buffers: [{ arrayStride: 24, attributes: [
      { shaderLocation: 0, offset: 0, format: 'float32x3' },
      { shaderLocation: 1, offset: 12, format: 'float32x3' },
    ] }] },
    fragment: { module: steamModule, entryPoint: 'fs', targets: [{ format, blend: {
      color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
    } }] },
    primitive: { cullMode: 'back' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less' },
    multisample: { count: 4 },
  });

  // ---- state ----------------------------------------------------------------

  const dark = matchMedia('(prefers-color-scheme: dark)');
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let yaw = 0.6, pitch = 0.42, dragging = null, hover = false, steamTime = 0;

  canvas.addEventListener('pointerdown', (e) => { dragging = { x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    yaw += (e.clientX - dragging.x) * 0.01;
    pitch = Math.min(1.2, Math.max(-0.2, pitch + (e.clientY - dragging.y) * 0.01));
    dragging = { x: e.clientX, y: e.clientY };
  });
  canvas.addEventListener('pointerup', () => { dragging = null; });
  canvas.addEventListener('pointerenter', () => { hover = true; });
  canvas.addEventListener('pointerleave', () => { hover = false; dragging = null; });

  let color, depth, size = [0, 0];
  const resize = () => {
    const dpr = Math.min(devicePixelRatio, 2);
    const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
    if (w === size[0] && h === size[1] || !w || !h) return;
    size = [w, h]; canvas.width = w; canvas.height = h;
    color?.destroy(); depth?.destroy();
    color = device.createTexture({ size, sampleCount: 4, format, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    depth = device.createTexture({ size, sampleCount: 4, format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });
  };
  new ResizeObserver(resize).observe(canvas);

  let last = performance.now();
  function frame(now) {
    const dt = Math.min((now - last) / 1000, 0.1); last = now;
    resize();
    if (!color) { requestAnimationFrame(frame); return; }

    if (!dragging && !reducedMotion.matches) yaw += dt * 0.25;
    steamTime += dt * (reducedMotion.matches ? 0 : hover ? 2.2 : 1);

    // Camera.
    // Pulled back far enough that the saucer stays in frame at the steepest pitch.
    const dist = 5.9, target = [0, 0.65, 0];
    const eye = [target[0] + Math.sin(yaw) * Math.cos(pitch) * dist, target[1] + Math.sin(pitch) * dist, target[2] + Math.cos(yaw) * Math.cos(pitch) * dist];
    // Fit the scene to the narrower dimension so nothing is cropped in portrait.
    const aspect = size[0] / size[1];
    const fov = 2 * Math.atan(Math.tan(0.3) / Math.min(aspect, 1));
    const viewProj = mat.multiply(mat.perspective(fov, aspect, 0.1, 50), mat.lookAt(eye, target, [0, 1, 0]));
    // Key light rides with the camera: above and to the viewer's left.
    const fwd = norm(sub(target, eye)), right = norm(cross(fwd, [0, 1, 0]));
    const light = norm([-fwd[0] - right[0] * 0.7, 1.1, -fwd[2] - right[2] * 0.7]);
    sceneData.set(viewProj, 0);
    sceneData.set(eye, 16); sceneData[19] = dark.matches ? 0.25 : 0.5;
    sceneData.set(light, 20); sceneData[23] = steamTime;
    // Steam is condensed water, seen by scattering: whitish on a dark page,
    // a soft grey on a light one.
    sceneData.set(dark.matches ? [0.95, 0.93, 0.9, 0.42] : [0.74, 0.77, 0.82, 0.3], 24);
    device.queue.writeBuffer(sceneBuf, 0, sceneData);

    const bg = dark.matches ? [0, 0, 0, 1] : [1, 1, 1, 1];
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: color.createView(), resolveTarget: ctx.getCurrentTexture().createView(), clearValue: bg, loadOp: 'clear', storeOp: 'discard' }],
      depthStencilAttachment: { view: depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
    });
    pass.setBindGroup(0, bindGroup);
    pass.setPipeline(opaque);
    const draw = (m, first, n = 1) => { pass.setVertexBuffer(0, m.vbuf); pass.setIndexBuffer(m.ibuf, 'uint32'); pass.drawIndexed(m.count, n, 0, 0, first); };
    draw(meshes.cup, 0); draw(meshes.saucer, 1); draw(meshes.coffee, 2); draw(meshes.handle, 3);
    pass.setPipeline(translucent);
    draw(steamBox, 0);
    pass.end();
    device.queue.submit([enc.finish()]);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main().catch(() => { canvas.hidden = true; fallback.hidden = false; });
