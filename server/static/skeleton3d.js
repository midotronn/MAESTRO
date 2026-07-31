"use strict";
// In-browser 3D stick-figure player for the fast preview (three.js r128, global THREE).
// Consumes /api/session/<sid>/skeleton (flat Z-up joints) and animates the 22-joint skeleton.
(function () {
  let scene, camera, renderer, controls, group, boneLines, jointPts, container;
  let frames = null, nFrames = 0, nJoints = 22, bones = [], fps = 15, floorZ = 0;
  let cur = 0, playing = true, lastT = 0, raf = null;
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
    setFrame(0);
    playing = true; lastT = performance.now();
  }

  function _j(t, j) {                          // joint (t,j) -> [x,y,z] (motion Z-up)
    const o = (t * nJoints + j) * 3;
    return [frames[o], frames[o + 1], frames[o + 2]];
  }

  function setFrame(i) {
    if (!frames || !nFrames || !jointPts || !boneLines) return;
    i = Math.max(0, Math.min(nFrames - 1, i | 0));
    cur = i;
    // horizontal center on the root (joint 0) so the dancer stays framed; keep vertical
    const r = _j(i, 0);
    const cx = r[0], cy = r[1];
    const jp = jointPts.geometry.attributes.position.array;
    for (let j = 0; j < nJoints; j++) {
      const p = _j(i, j);
      jp[j * 3] = p[0] - cx; jp[j * 3 + 1] = p[1] - cy; jp[j * 3 + 2] = p[2] - floorZ;
    }
    jointPts.geometry.attributes.position.needsUpdate = true;
    const bp = boneLines.geometry.attributes.position.array;
    for (let k = 0; k < bones.length; k++) {
      const c = _j(i, bones[k][0]), pa = _j(i, bones[k][1]);
      const b = k * 6;
      bp[b] = c[0] - cx; bp[b + 1] = c[1] - cy; bp[b + 2] = c[2] - floorZ;
      bp[b + 3] = pa[0] - cx; bp[b + 4] = pa[1] - cy; bp[b + 5] = pa[2] - floorZ;
    }
    boneLines.geometry.attributes.position.needsUpdate = true;
    if (onFrameCb) onFrameCb(cur, nFrames, cur / fps);
  }

  function animate() {
    raf = requestAnimationFrame(animate);
    const now = performance.now();
    if (playing && frames && nFrames > 1) {
      const adv = ((now - lastT) / 1000) * fps;
      if (adv >= 1) { cur = (cur + Math.floor(adv)) % nFrames; setFrame(cur); lastT = now; }
    } else { lastT = now; }
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
