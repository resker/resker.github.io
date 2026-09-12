// Initech coffee mug in WebGPU. No libraries: geometry is generated here,
// shading is the WGSL below. Drag to orbit; the steam rises faster while the
// pointer is over the cup.

const canvas = document.querySelector('canvas');
const fallback = document.querySelector('p[hidden]');

const shader = /* wgsl */ `
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
  rings: f32,
};
@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;

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
  // Steam is translucent and unlit; fading alpha toward the silhouette
  // turns each sphere into a soft puff instead of a hard-edged ball.
  if (inst.color.a < 1.0) {
    let facing = max(dot(normalize(i.normal), normalize(scene.eye - i.worldPos)), 0.0);
    return vec4f(inst.color.rgb, inst.color.a * pow(facing, 2.0));
  }
  var base = inst.color.rgb;
  // Two dark rings around the outside of the cup, like the artwork.
  if (inst.rings > 0.5) {
    let r = length(i.localPos.xz);
    let y = i.localPos.y;
    let outside = r > 0.86;
    let band = (y > 1.05 && y < 1.085) || (y > 1.13 && y < 1.165);
    if (outside && band) { base = vec3f(0.12, 0.12, 0.13); }
  }
  let n = normalize(i.normal);
  let l = normalize(scene.light);
  let v = normalize(scene.eye - i.worldPos);
  let h = normalize(l + v);
  let diff = max(dot(n, l), 0.0);
  let spec = pow(max(dot(n, h), 0.0), 64.0) * 0.35;
  let fill = max(dot(n, normalize(vec3f(-0.5, 0.3, -0.7))), 0.0) * 0.25;
  let lit = base * (scene.ambient + diff * 0.65 + fill) + vec3f(spec);
  return vec4f(lit, inst.color.a);
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

// Tube swept along an arc in the XY plane (the handle).
function arcTube(center, radius, tube, from, to, steps = 32, ring = 16) {
  const pos = [], nrm = [], idx = [];
  for (let i = 0; i <= steps; i++) {
    const a = from + (to - from) * (i / steps);
    const cx = center[0] + Math.cos(a) * radius, cy = center[1] + Math.sin(a) * radius;
    const rx = Math.cos(a), ry = Math.sin(a); // radial direction in XY
    for (let j = 0; j <= ring; j++) {
      const b = (j / ring) * Math.PI * 2, cb = Math.cos(b), sb = Math.sin(b);
      const nx = rx * cb, ny = ry * cb, nz = sb;
      pos.push(cx + nx * tube, cy + ny * tube, nz * tube);
      nrm.push(nx, ny, nz);
    }
  }
  for (let i = 0; i < steps; i++) {
    for (let j = 0; j < ring; j++) {
      const a = i * (ring + 1) + j, b = a + ring + 1;
      idx.push(a, a + 1, b, a + 1, b + 1, b);
    }
  }
  return { pos, nrm, idx };
}

function sphere(r = 1, seg = 12) {
  const profile = [];
  for (let i = 0; i <= seg; i++) {
    const t = (i / seg) * Math.PI;
    profile.push([Math.sin(t) * r + 1e-4, -Math.cos(t) * r]);
  }
  return lathe(profile, seg * 2);
}

const cupProfile = [
  [0.0, 0.02], [0.36, 0.02], [0.5, 0.06], [0.66, 0.2], [0.8, 0.5],
  [0.88, 0.85], [0.92, 1.15], [0.93, 1.34], [0.93, 1.38], [0.87, 1.38],
  [0.86, 1.3], [0.84, 1.05], [0.76, 0.6], [0.6, 0.3], [0.4, 0.2], [0.0, 0.18],
];
const saucerProfile = [
  [0.0, -0.08], [0.55, -0.08], [0.6, -0.04], [1.3, 0.0], [1.7, 0.08],
  [1.82, 0.14], [1.8, 0.17], [1.66, 0.14], [1.25, 0.06], [0.62, 0.02], [0.0, 0.02],
];
const coffeeLevel = 1.18;
const coffeeProfile = [[0.85, coffeeLevel], [0.0, coffeeLevel]]; // right-to-left so the normal faces up

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
  scale(s) { const m = mat.identity(); m[0] = m[5] = m[10] = s; return m; },
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

  const meshes = {
    cup: lathe(cupProfile),
    saucer: lathe(saucerProfile),
    coffee: lathe(coffeeProfile, 48),
    handle: arcTube([1.05, 0.85], 0.36, 0.075, -Math.PI * 0.7, Math.PI * 0.7),
    wisp: sphere(1, 12),
  };
  for (const m of Object.values(meshes)) {
    const v = new Float32Array(m.pos.length * 2);
    for (let i = 0, j = 0; i < m.pos.length; i += 3, j += 6) {
      v.set(m.pos.slice(i, i + 3), j); v.set(m.nrm.slice(i, i + 3), j + 3);
    }
    m.vbuf = device.createBuffer({ size: v.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(m.vbuf, 0, v);
    const ix = new Uint32Array(m.idx);
    m.ibuf = device.createBuffer({ size: ix.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(m.ibuf, 0, ix);
    m.count = ix.length;
  }

  // Instance table: 0 cup, 1 saucer, 2 coffee, 3 handle, 4.. steam spheres.
  const WISPS = 3, PER_WISP = 24, STEAM_START = 4;
  const instanceCount = STEAM_START + WISPS * PER_WISP;
  const INST_FLOATS = 24; // mat4 (16) + color (4) + rings (1) + pad (3)
  const instData = new Float32Array(instanceCount * INST_FLOATS);
  const instBuf = device.createBuffer({ size: instData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const setInstance = (i, model, color, rings = 0) => {
    instData.set(model, i * INST_FLOATS);
    instData.set(color, i * INST_FLOATS + 16);
    instData[i * INST_FLOATS + 20] = rings;
  };
  const ceramic = [0.93, 0.93, 0.94, 1];
  setInstance(0, mat.identity(), ceramic, 1);
  setInstance(1, mat.identity(), [0.86, 0.86, 0.87, 1]);
  setInstance(2, mat.identity(), [0.27, 0.15, 0.06, 1]);
  setInstance(3, mat.identity(), ceramic);

  const sceneData = new Float32Array(16 + 4 + 4);
  const sceneBuf = device.createBuffer({ size: sceneData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const module = device.createShaderModule({ code: shader });
  const layout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
  ] });
  const bindGroup = device.createBindGroup({ layout, entries: [
    { binding: 0, resource: { buffer: sceneBuf } },
    { binding: 1, resource: { buffer: instBuf } },
  ] });
  const makePipeline = (blend) => device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    vertex: { module, entryPoint: 'vs', buffers: [{ arrayStride: 24, attributes: [
      { shaderLocation: 0, offset: 0, format: 'float32x3' },
      { shaderLocation: 1, offset: 12, format: 'float32x3' },
    ] }] },
    fragment: { module, entryPoint: 'fs', targets: [{ format, blend: blend ? {
      color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
    } : undefined }] },
    primitive: { cullMode: 'none' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: !blend, depthCompare: 'less' },
    multisample: { count: 4 },
  });
  const opaque = makePipeline(false), translucent = makePipeline(true);

  // ---- state ----------------------------------------------------------------

  const dark = matchMedia('(prefers-color-scheme: dark)');
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let yaw = 0.6, pitch = 0.42, dragging = null, hover = false, steamPhase = 0;

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
    steamPhase += dt * (reducedMotion.matches ? 0 : hover ? 2.2 : 1);

    // Camera.
    const dist = 5.2, target = [0, 0.55, 0];
    const eye = [target[0] + Math.sin(yaw) * Math.cos(pitch) * dist, target[1] + Math.sin(pitch) * dist, target[2] + Math.cos(yaw) * Math.cos(pitch) * dist];
    const viewProj = mat.multiply(mat.perspective(0.6, size[0] / size[1], 0.1, 50), mat.lookAt(eye, target, [0, 1, 0]));
    sceneData.set(viewProj, 0);
    // Key light rides with the camera: above and to the viewer's left.
    const fwd = norm(sub(target, eye)), right = norm(cross(fwd, [0, 1, 0]));
    const light = norm([-fwd[0] - right[0] * 0.7, 1.1, -fwd[2] - right[2] * 0.7]);
    sceneData.set(eye, 16); sceneData[19] = dark.matches ? 0.25 : 0.5;
    sceneData.set(light, 20); sceneData[23] = now / 1000;
    device.queue.writeBuffer(sceneBuf, 0, sceneData);

    // Steam: three wisps, each a column of spheres drifting up on a sine path.
    for (let w = 0; w < WISPS; w++) {
      const period = w === 2 ? 15 : 7, offset = w * 2.4;
      for (let k = 0; k < PER_WISP; k++) {
        const t = ((steamPhase + offset + k * (period / PER_WISP)) % period) / period; // 0..1 lifetime
        const y = coffeeLevel + 0.05 + t * 2.0;
        const x = Math.sin(t * 9 + w * 2.1) * (0.08 + t * 0.3) + (w - 1) * 0.18;
        const z = Math.cos(t * 7 + w) * (0.06 + t * 0.2);
        const fade = Math.sin(t * Math.PI) ** 2;
        const r = 0.05 + t * 0.13;
        const m = mat.multiply(mat.translate(x, y, z), mat.scale(r));
        setInstance(STEAM_START + w * PER_WISP + k, m, [0.92, 0.7, 0.3, 0.16 * fade]);
      }
    }
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
    draw(meshes.wisp, STEAM_START, WISPS * PER_WISP);
    pass.end();
    device.queue.submit([enc.finish()]);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main().catch(() => { canvas.hidden = true; fallback.hidden = false; });
