"use strict";
// In-browser 3D stick-figure player for the fast preview (three.js r128, global THREE).
// Consumes /api/session/<sid>/skeleton (flat Z-up joints) and animates the 22-joint skeleton.
(function () {
  let scene, camera, renderer, controls, group, boneLines, jointPts, container;
  let frames = null, nFrames = 0, nJoints = 22, bones = [], fps = 15, floorZ = 0;
  let cur = 0, curF = 0, playing = true, lastT = 0, raf = null;
  let onFrameCb = null;

  const UP = 0.0; // feet target height

  function init(id) {
    container = document.getElementById(id);
    if (!container || !window.THREE) return false;
    const w = container.clientWidth || 640, h = container.clientHeight || 480;
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0e13);
    camera = new THREE.PerspectiveCamera(42, w / h, 0.05, 200);
    camera.position.set(0, 1.1, 3.4);
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h);
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0.9, 0);
    controls.enableDamping = true; controls.dampingFactor = 0.08;
    controls.minDistance = 1.2; controls.maxDistance = 12;

    scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    const dir = new THREE.DirectionalLight(0xffffff, 0.9); dir.position.set(2, 5, 3); scene.add(dir);
    const grid = new THREE.GridHelper(8, 16, 0x2c3644, 0x171d26); grid.position.y = 0; scene.add(grid);

    group = new THREE.Group(); group.rotation.x = -Math.PI / 2; scene.add(group); // Z-up -> Y-up

    window.addEventListener("resize", onResize);
    animate();
    return true;
  }

  function onResize() {
    if (!renderer || !container) return;
    const w = container.clientWidth, h = container.clientHeight;
    if (w && h) { camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.setSize(w, h); }
  }

  function load(data) {
    const f = Float32Array.from(data.joints);
    nJoints = data.n_joints || 22;
    bones = data.bones || []; fps = data.fps || 15;
    cur = 0;
    // ground the dancer: lift so the lowest point across the whole clip sits on the floor (y=0)
    floorZ = Infinity;
    for (let i = 2; i < f.length; i += 3) if (f[i] < floorZ) floorZ = f[i];
    if (!isFinite(floorZ)) floorZ = 0;

    if (boneLines) { group.remove(boneLines); boneLines.geometry.dispose(); }
    if (jointPts) { group.remove(jointPts); jointPts.geometry.dispose(); }
    const bg = new THREE.BufferGeometry();
    bg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(bones.length * 2 * 3), 3));
    boneLines = new THREE.LineSegments(bg, new THREE.LineBasicMaterial({ color: 0x8fd6ff }));
    group.add(boneLines);
    const jg = new THREE.BufferGeometry();
    jg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(nJoints * 3), 3));
    jointPts = new THREE.Points(jg, new THREE.PointsMaterial({ color: 0x35d07f, size: 0.05, sizeAttenuation: true }));
    group.add(jointPts);

    frames = f; nFrames = data.n_frames;         // assign LAST so the animate loop can't race ahead
    curF = 0;
    setFrame(0);
    playing = true; lastT = performance.now();
  }

  function _j(t, j) {                          // joint (t,j) -> [x,y,z] (motion Z-up)
    const o = (t * nJoints + j) * 3;
    return [frames[o], frames[o + 1], frames[o + 2]];
  }

  // Render an (interpolated) pose at fractional frame position `fpos`. Interpolating between the
  // downsampled keyframes gives buttery display-rate playback from a low source fps.
  function _renderPose(fpos) {
    if (!frames || !nFrames || !jointPts || !boneLines) return;
    fpos = Math.max(0, Math.min(nFrames - 1e-4, fpos));
    const i0 = Math.floor(fpos), i1 = Math.min(nFrames - 1, i0 + 1), t = fpos - i0;
    const base0 = i0 * nJoints * 3, base1 = i1 * nJoints * 3;
    const pos = (b, j, k) => frames[b + j * 3 + k];
    const at = (j, k) => pos(base0, j, k) * (1 - t) + pos(base1, j, k) * t;   // lerp keyframes
    const cx = at(0, 0), cy = at(0, 1);                      // center on root (dance in place)
    const jp = jointPts.geometry.attributes.position.array;
    for (let j = 0; j < nJoints; j++) {
      jp[j * 3] = at(j, 0) - cx; jp[j * 3 + 1] = at(j, 1) - cy; jp[j * 3 + 2] = at(j, 2) - floorZ;
    }
    jointPts.geometry.attributes.position.needsUpdate = true;
    const bp = boneLines.geometry.attributes.position.array;
    for (let k = 0; k < bones.length; k++) {
      const c = bones[k][0], pa = bones[k][1], b = k * 6;
      bp[b] = at(c, 0) - cx; bp[b + 1] = at(c, 1) - cy; bp[b + 2] = at(c, 2) - floorZ;
      bp[b + 3] = at(pa, 0) - cx; bp[b + 4] = at(pa, 1) - cy; bp[b + 5] = at(pa, 2) - floorZ;
    }
    boneLines.geometry.attributes.position.needsUpdate = true;
    cur = Math.round(fpos);
    if (onFrameCb) onFrameCb(cur, nFrames, fpos / fps);
  }

  function setFrame(i) {                        // scrub to an exact frame
    curF = Math.max(0, Math.min(nFrames - 1, i | 0));
    _renderPose(curF);
  }

  function animate() {
    raf = requestAnimationFrame(animate);
    const now = performance.now();
    const dt = Math.min(0.05, (now - lastT) / 1000);        // clamp big gaps (tab switch)
    lastT = now;
    if (playing && frames && nFrames > 1) {
      curF += dt * fps;                                     // accumulate time (no fractional loss)
      if (curF >= nFrames) curF -= nFrames;
      _renderPose(curF);
    }
    if (controls) controls.update();
    if (renderer) renderer.render(scene, camera);
  }

  window.Skel3D = {
    init, load, setFrame,
    play: () => { playing = true; lastT = performance.now(); },
    pause: () => { playing = false; },
    toggle: () => { playing = !playing; lastT = performance.now(); return playing; },
    isPlaying: () => playing,
    frame: () => cur, total: () => nFrames, fps: () => fps,
    onFrame: (cb) => { onFrameCb = cb; },
    resize: onResize,
  };
})();
