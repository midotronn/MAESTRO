const $ = (id) => document.getElementById(id);
const video = $("comparison");
let assignments = null;
let excerptDuration = 6;

function formatTime(seconds) {
  const safe = Number.isFinite(seconds) ? seconds : 0;
  return `${Math.floor(safe / 60)}:${String(Math.floor(safe % 60)).padStart(2, "0")}`;
}

function phaseRows() {
  return assignments.phases[$("phase").value] || [];
}

function populateParticipants(preferred, preferredTriplet = 0) {
  const rows = phaseRows();
  $("participant").replaceChildren(...rows.map((row, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = row.participant_id;
    return option;
  }));
  $("participant").value = String(Math.min(Number(preferred) || 0, Math.max(0, rows.length - 1)));
  loadParticipant(preferredTriplet);
}

function loadParticipant(preferredTriplet = 0) {
  const participant = phaseRows()[Number($("participant").value)];
  if (!participant) return;
  $("assignmentCode").textContent = participant.assignment_code;
  $("triplet").replaceChildren(...participant.triplets.map((triplet, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `Triplet ${index + 1} · ${triplet.excerpt}`;
    return option;
  }));
  $("triplet").value = String(Math.min(Number(preferredTriplet) || 0, participant.triplets.length - 1));
  loadTriplet();
}

function loadTriplet() {
  const participantIndex = Number($("participant").value);
  const tripletIndex = Number($("triplet").value);
  const participant = phaseRows()[participantIndex];
  const selected = participant.triplets[tripletIndex];
  const order = selected.lanes.join("");
  const revision = encodeURIComponent(assignments.protocol || "current");

  video.pause();
  video.src = `videos/${selected.excerpt}/${order}.mp4?v=${revision}`;
  video.load();
  $("excerptLabel").textContent =
    `${selected.excerpt} · synchronized ${excerptDuration}-second excerpt`;
  $("play").textContent = "▶ Play";
  $("scrub").value = "0";
  $("time").value = `0:00 / ${formatTime(excerptDuration)}`;
  history.replaceState(
    null,
    "",
    `?phase=${$("phase").value}&participant=${participantIndex + 1}&triplet=${tripletIndex + 1}`,
  );
  localStorage.setItem("maestro-study-phase", $("phase").value);
  localStorage.setItem("maestro-study-participant", String(participantIndex));
  localStorage.setItem("maestro-study-triplet", String(tripletIndex));
}

function changeTriplet(delta) {
  const count = $("triplet").options.length;
  const next = Math.max(0, Math.min(count - 1, Number($("triplet").value) + delta));
  if (next !== Number($("triplet").value)) {
    $("triplet").value = String(next);
    loadTriplet();
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
  $("excerptLabel").textContent = "Video failed to load; refresh before continuing";
});

$("phase").addEventListener("change", () => populateParticipants(0, 0));
$("participant").addEventListener("change", () => loadParticipant(0));
$("triplet").addEventListener("change", loadTriplet);
$("play").addEventListener("click", () => video.paused ? video.play() : video.pause());
$("restart").addEventListener("click", () => {
  video.currentTime = 0;
  video.play();
});
$("scrub").addEventListener("input", () => {
  video.currentTime = Number($("scrub").value);
});
$("previous").addEventListener("click", () => changeTriplet(-1));
$("next").addEventListener("click", () => changeTriplet(1));
$("fullscreen").addEventListener("click", () => document.querySelector(".stage").requestFullscreen());
document.addEventListener("keydown", (event) => {
  if (event.target.matches("select, input, button")) return;
  if (event.code === "Space") {
    event.preventDefault();
    video.paused ? video.play() : video.pause();
  } else if (event.key === "ArrowLeft") changeTriplet(-1);
  else if (event.key === "ArrowRight") changeTriplet(1);
});

async function init() {
  const [assignmentResponse, configResponse] = await Promise.all([
    fetch("assignments.json"),
    fetch("config.json"),
  ]);
  if (!assignmentResponse.ok || !configResponse.ok) {
    throw new Error("Study configuration could not be loaded");
  }
  assignments = await assignmentResponse.json();
  const config = await configResponse.json();
  excerptDuration = Number(config.excerpt_duration_seconds) || excerptDuration;
  $("formLink").href = config.response_form_url;

  const query = new URLSearchParams(location.search);
  const phase = query.get("phase") || localStorage.getItem("maestro-study-phase") || "pilot";
  $("phase").value = phase in assignments.phases ? phase : "pilot";
  const participant = query.has("participant")
    ? Math.max(0, Number(query.get("participant")) - 1)
    : Math.max(0, Number(localStorage.getItem("maestro-study-participant")) || 0);
  const triplet = query.has("triplet")
    ? Math.max(0, Number(query.get("triplet")) - 1)
    : Math.max(0, Number(localStorage.getItem("maestro-study-triplet")) || 0);
  populateParticipants(participant, triplet);
}

init().catch((error) => {
  $("excerptLabel").textContent = error.message;
});
