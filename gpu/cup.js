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
};
struct Instance {
  model: mat4x4f,
  color: vec4f,
  params: vec4f, // x: material (1 cup rings, 2 coffee); wisps: phase, speed, sway, height
};
@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
`;

const litShader = common + /* wgsl */ `
// Cheap value noise for the crema.
fn hash(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453); }
fn noise(p: vec2f) -> f32 {
  let i = floor(p); let f = fract(p); let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2f(1, 0)), u.x), mix(hash(i + vec2f(0, 1)), hash(i + vec2f(1, 1)), u.x), u.y);
}
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
    // Coffee. The surface is dark and glossy. Foam is a thin continuous band
    // against the wall plus a patch or two bulging in from it; inside, it is
    // densely packed tiny cells, matte and cream-coloured.
    let q = i.localPos.xz;
    let angle = atan2(q.y, q.x);
    let wall = 0.85;
    // Coverage: a band whose width wanders around the rim, and patches.
    let bandW = 0.02 + noise(vec2f(angle * 3.0, 2.3)) * 0.05;
    let band = smoothstep(wall - bandW - 0.01, wall - bandW + 0.01, r);
    let blob = noise(q * 2.6 + 5.0) + (noise(q * 24.0) - 0.5) * 0.12;
    let blobs = smoothstep(0.69, 0.73, blob) * smoothstep(0.5, 0.72, r);
    let foam = clamp(band + blobs, 0.0, 1.0);
    // Texture inside the foam: cell centres lighter, walls darker, a few
    // pinpoint highlights where a bubble catches the light.
    let d = cells(q * 130.0);
    let d2 = cells(q * 55.0 + 3.0);
    let cellTone = 0.7 + 0.3 * (1.0 - smoothstep(0.15, 0.6, d)) - 0.15 * smoothstep(0.35, 0.55, d2);
    let glint = (1.0 - smoothstep(0.0, 0.12, d)) * 0.35;
    // Thicker foam (patch centres, right at the wall) reads lighter.
    let thick = 0.85 + 0.25 * max(smoothstep(0.7, 0.85, blob), smoothstep(wall - 0.03, wall, r));
    let cream = vec3f(0.74, 0.58, 0.38) * cellTone * thick + vec3f(glint);
    // Thin light film near the wall under the foam, then the foam itself.
    base = mix(base, base * 1.6, smoothstep(0.7, wall, r) * 0.3);
    base = mix(base, cream, foam);
    // Coffee is glossy; foam is matte.
    gloss = mix(20.0, 6.0, foam); specAmt = mix(0.45, 0.06, foam);
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

// Steam: each wisp is a camera-facing ribbon. The vertex shader bends it with
// travelling waves so it curls as it rises; alpha fades at both ends and at the
// ribbon's edges so it reads as vapour rather than a strip.
const steamShader = common + /* wgsl */ `
struct Out {
  @builtin(position) pos: vec4f,
  @location(0) t: f32,
  @location(1) side: f32,
  @location(2) fade: f32,
  @location(3) @interpolate(flat) id: u32,
};

@vertex fn vs(@location(0) a: vec2f, @builtin(instance_index) id: u32) -> Out {
  let inst = instances[id];
  let t = a.x;                       // 0 at the coffee, 1 at the top
  let phase = inst.params.x;
  let speed = inst.params.y;
  let sway = inst.params.z;
  let height = inst.params.w;
  let time = scene.time * speed;
  // Curl: two travelling sine waves per axis, growing with height.
  let grow = 0.25 + t * 1.1;
  let x = (sin(t * 7.0 - time * 1.9 + phase) * 0.55 + sin(t * 15.0 - time * 3.1 + phase * 2.0) * 0.18) * sway * grow;
  let z = (cos(t * 6.0 - time * 1.6 + phase * 1.3) * 0.45 + sin(t * 12.0 - time * 2.6) * 0.15) * sway * grow;
  let base = (inst.model * vec4f(0, 0, 0, 1)).xyz;
  let centre = base + vec3f(x, t * height, z);
  // Billboard the width across the camera's right vector.
  let toEye = normalize(scene.eye - centre);
  let flat = vec3f(toEye.x, 0.0, toEye.z);
  let right = normalize(cross(vec3f(0, 1, 0), select(flat, vec3f(1, 0, 0), length(flat) < 1e-3)));
  let width = (0.05 + t * 0.16) * (1.0 - t * 0.35);
  var o: Out;
  o.pos = scene.viewProj * vec4f(centre + right * a.y * width, 1);
  o.t = t;
  o.side = a.y;
  // Seen from overhead a ribbon is edge-on and meaningless; fade it out.
  o.fade = 1.0 - smoothstep(0.55, 0.9, toEye.y);
  o.id = id;
  return o;
}

@fragment fn fs(i: Out) -> @location(0) vec4f {
  let inst = instances[i.id];
  let along = pow(sin(i.t * 3.14159), 1.4) * (1.0 - i.t * 0.3);
  let across = 1.0 - i.side * i.side;
  let a = inst.color.a * along * across * across * i.fade;
  return vec4f(inst.color.rgb * a, a); // premultiplied
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

// Tube swept along a polyline in the XY plane (the handle).
function sweep(path, tube, ring = 16) {
  const pos = [], nrm = [], idx = [];
  for (let i = 0; i < path.length; i++) {
    const [px, py] = path[i];
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

// Ribbon for a steam wisp: (t, side) pairs, shaped in the vertex shader.
function ribbon(steps = 48) {
  const pos = [], idx = [];
  for (let i = 0; i <= steps; i++) { pos.push(i / steps, -1, i / steps, 1); }
  for (let i = 0; i < steps; i++) { const a = i * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
  return { pos, idx };
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
  [0.905, 1.02], [0.93, 1.3], [0.93, 1.38],
]);
const innerWall = spline([
  [0.87, 1.38], [0.86, 1.28], [0.82, 0.95], [0.74, 0.62], [0.6, 0.36], [0.38, 0.22], [0.0, 0.2],
]);
const cupProfile = [[0.0, 0.0], ...outerWall, ...innerWall];
function cupOuterRadius(y) {
  for (let i = 1; i < outerWall.length; i++) {
    const [r0, y0] = outerWall[i - 1], [r1, y1] = outerWall[i];
    if (y <= y1) return r0 + (r1 - r0) * ((y - y0) / (y1 - y0 || 1));
  }
  return outerWall.at(-1)[0];
}

// Saucer: foot ring underneath, central well with a ridge that seats the cup,
// a gently rising dish, and a rolled rim. About 1.5x the cup's rim.
const saucerProfile = [
  [0.0, -0.09], [0.48, -0.09], [0.52, -0.05], [0.62, -0.05], [1.05, 0.0],
  [1.36, 0.1], [1.42, 0.15], [1.41, 0.18], [1.34, 0.17], [1.05, 0.09],
  [0.82, 0.05], [0.76, 0.08], [0.72, 0.07], [0.7, 0.02], [0.0, 0.02],
];
const coffeeLevel = 1.18;
const coffeeProfile = [[0.85, coffeeLevel], [0.0, coffeeLevel]]; // right-to-left so the normal faces up

// Handle: one smooth cubic Bezier that leaves the wall just below the rings,
// loops outward, and re-enters at mid-cup. Both ends sit inside the wall.
function handlePath() {
  const top = 0.98, bottom = 0.5, inset = 0.05;
  const p0 = [cupOuterRadius(top) - inset, top], p3 = [cupOuterRadius(bottom) - inset, bottom];
  const p1 = [p0[0] + 0.62, top + 0.08], p2 = [p3[0] + 0.66, bottom - 0.14];
  const path = [];
  for (let i = 0; i <= 48; i++) {
    const t = i / 48, u = 1 - t;
    path.push([0, 1].map((k) => u * u * u * p0[k] + 3 * u * u * t * p1[k] + 3 * u * t * t * p2[k] + t * t * t * p3[k]));
  }
  return path;
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
    handle: sweep(handlePath(), 0.085),
  };
  for (const m of Object.values(meshes)) {
    const v = [];
    for (let i = 0; i < m.pos.length; i += 3) v.push(...m.pos.slice(i, i + 3), ...m.nrm.slice(i, i + 3));
    upload(m, v);
  }
  const wispMesh = ribbon();
  const wisp = upload(wispMesh, wispMesh.pos);

  // Instance table: 0 cup, 1 saucer, 2 coffee, 3 handle, then the wisps.
  const WISPS = 4, STEAM_START = 4;
  const INST_FLOATS = 24; // mat4 (16) + color (4) + params (4)
  const instData = new Float32Array((STEAM_START + WISPS) * INST_FLOATS);
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
  // Wisps: base position, tint, [phase, speed, sway, height].
  const wispSpecs = [
    [[-0.15, 0.1], 1.0, 0.9, 0.22, 1.05],
    [[0.18, -0.05], 2.7, 0.7, 0.28, 1.25],
    [[0.02, 0.2], 4.4, 1.1, 0.18, 0.95],
    [[-0.05, -0.18], 5.9, 0.55, 0.3, 1.35],
  ];
  const setWisps = (dark) => {
    const tint = dark ? [0.9, 0.84, 0.72, 0.4] : [0.7, 0.58, 0.42, 0.2];
    wispSpecs.forEach(([[x, z], phase, speed, sway, height], w) => {
      setInstance(STEAM_START + w, mat.translate(x, coffeeLevel + 0.02, z), tint, [phase, speed, sway, height]);
    });
  };

  const sceneData = new Float32Array(16 + 4 + 4);
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
    vertex: { module: steamModule, entryPoint: 'vs', buffers: [{ arrayStride: 8, attributes: [
      { shaderLocation: 0, offset: 0, format: 'float32x2' },
    ] }] },
    fragment: { module: steamModule, entryPoint: 'fs', targets: [{ format, blend: {
      color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
    } }] },
    primitive: { cullMode: 'none' },
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
    const viewProj = mat.multiply(mat.perspective(0.6, size[0] / size[1], 0.1, 50), mat.lookAt(eye, target, [0, 1, 0]));
    // Key light rides with the camera: above and to the viewer's left.
    const fwd = norm(sub(target, eye)), right = norm(cross(fwd, [0, 1, 0]));
    const light = norm([-fwd[0] - right[0] * 0.7, 1.1, -fwd[2] - right[2] * 0.7]);
    sceneData.set(viewProj, 0);
    sceneData.set(eye, 16); sceneData[19] = dark.matches ? 0.25 : 0.5;
    sceneData.set(light, 20); sceneData[23] = steamTime;
    device.queue.writeBuffer(sceneBuf, 0, sceneData);
    setWisps(dark.matches);
    device.queue.writeBuffer(instBuf, 0, instData);

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
    draw(wisp, STEAM_START, WISPS);
    pass.end();
    device.queue.submit([enc.finish()]);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main().catch(() => { canvas.hidden = true; fallback.hidden = false; });
