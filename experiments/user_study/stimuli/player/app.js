const $ = (id) => document.getElementById(id);
const video = $("comparison");
let study = null;
let excerptDuration = 16;

function formatTime(seconds) {
  const safe = Number.isFinite(seconds) ? seconds : 0;
  return `${Math.floor(safe / 60)}:${String(Math.floor(safe % 60)).padStart(2, "0")}`;
}

function updateNavigation(index) {
  $("previous").disabled = index === 0;
  $("next").disabled = index === study.sequence.length - 1;
}

function loadComparison() {
  const index = Number($("comparisonSelect").value);
  const selected = study.sequence[index];
  if (!selected) return;

  const order = selected.lanes.join("");
  const revision = encodeURIComponent(study.protocol || "current");

  video.pause();
  video.src = `videos/${selected.excerpt}/${order}.mp4?v=${revision}`;
  video.load();
  $("comparisonProgress").textContent = `Comparison ${index + 1} of ${study.sequence.length}`;
  $("statusLabel").textContent = `Synchronized ${excerptDuration}-second excerpt`;
  $("play").textContent = "▶ Play";
  $("scrub").value = "0";
  $("time").value = `0:00 / ${formatTime(excerptDuration)}`;
  updateNavigation(index);
}

function populateComparisons() {
  $("comparisonSelect").replaceChildren(...study.sequence.map((_, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `Comparison ${index + 1}`;
    return option;
  }));
  $("comparisonSelect").value = "0";
  loadComparison();
}

function changeComparison(delta) {
  const current = Number($("comparisonSelect").value);
  const next = Math.max(0, Math.min(study.sequence.length - 1, current + delta));
  if (next !== current) {
    $("comparisonSelect").value = String(next);
    loadComparison();
  }
}

video.addEventListener("play", () => {
  $("play").textContent = "❚❚ Pause";
});
video.addEventListener("pause", () => {
  $("play").textContent = "▶ Play";
});
video.addEventListener("timeupdate", () => {
  $("scrub").max = String(video.duration || excerptDuration);
  $("scrub").value = String(video.currentTime);
  $("time").value =
    `${formatTime(video.currentTime)} / ${formatTime(video.duration || excerptDuration)}`;
});
video.addEventListener("loadedmetadata", () => {
  if (video.currentTime === 0) video.currentTime = 0.001;
});
video.addEventListener("error", () => {
  $("statusLabel").textContent = "Video failed to load; refresh before continuing";
});

$("comparisonSelect").addEventListener("change", loadComparison);
$("play").addEventListener("click", () => video.paused ? video.play() : video.pause());
$("restart").addEventListener("click", () => {
  video.currentTime = 0;
  video.play();
});
$("scrub").addEventListener("input", () => {
  video.currentTime = Number($("scrub").value);
});
$("previous").addEventListener("click", () => changeComparison(-1));
$("next").addEventListener("click", () => changeComparison(1));
$("fullscreen").addEventListener("click", () => document.querySelector(".stage").requestFullscreen());
document.addEventListener("keydown", (event) => {
  if (event.target.matches("select, input, button")) return;
  if (event.code === "Space") {
    event.preventDefault();
    video.paused ? video.play() : video.pause();
  } else if (event.key === "ArrowLeft") changeComparison(-1);
  else if (event.key === "ArrowRight") changeComparison(1);
});

async function init() {
  const [studyResponse, configResponse] = await Promise.all([
    fetch("assignments.json"),
    fetch("config.json"),
  ]);
  if (!studyResponse.ok || !configResponse.ok) {
    throw new Error("Study configuration could not be loaded");
  }
  study = await studyResponse.json();
  const config = await configResponse.json();
  if (!Array.isArray(study.sequence) || study.sequence.length === 0) {
    throw new Error("The comparison sequence is empty");
  }
  excerptDuration = Number(config.excerpt_duration_seconds) || excerptDuration;
  $("formLink").href = config.response_form_url;
  populateComparisons();
}

init().catch((error) => {
  $("statusLabel").textContent = error.message;
});
