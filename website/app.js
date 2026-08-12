const PLAYBACK_RATE = 0.75;
const FPS = 15;
const FORECAST_STEPS = 15;
const DREAM_PANEL = { x: 0, y: 0, width: 256, height: 256 };
const REALITY_PANEL = { x: 256, y: 8, width: 256, height: 240 };

const reel = document.querySelector("#reel");
const stage = document.querySelector(".stage");
const dreamCanvas = document.querySelector("#dream-canvas");
const dreamContext = dreamCanvas.getContext("2d", { alpha: false });
const gameplayCanvas = document.querySelector("#gameplay-canvas");
const gameplayContext = gameplayCanvas.getContext("2d", { alpha: false });
const pairedVideo = document.querySelector("#paired-source");
const dreamWindowLabel = document.querySelector("#dream-window-label");

pairedVideo.playbackRate = PLAYBACK_RATE;
pairedVideo.addEventListener("loadeddata", () => {
  pairedVideo.playbackRate = PLAYBACK_RATE;
  void pairedVideo.play().catch(() => {});
  renderPairedFrame();
}, { once: true });

pairedVideo.addEventListener("timeupdate", updateForecastLabel);
window.addEventListener("resize", resizeCanvases);
window.addEventListener("scroll", updateScrollProgress, { passive: true });

resizeCanvases();
updateScrollProgress();

function renderPairedFrame() {
  if (pairedVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    drawDreamPanel();
    drawGameplayPanel();
  }
  if ("requestVideoFrameCallback" in pairedVideo) {
    pairedVideo.requestVideoFrameCallback(renderPairedFrame);
  } else {
    window.requestAnimationFrame(renderPairedFrame);
  }
}

function drawDreamPanel() {
  const width = dreamCanvas.width;
  const height = dreamCanvas.height;
  const source = DREAM_PANEL;
  dreamContext.imageSmoothingEnabled = false;
  dreamContext.fillStyle = "#101510";
  dreamContext.fillRect(0, 0, width, height);

  const square = Math.min(width, height);
  const offsetX = Math.floor((width - square) / 2);
  const offsetY = Math.floor((height - square) / 2);

  if (width >= height) {
    dreamContext.drawImage(
      pairedVideo,
      source.x,
      source.y,
      8,
      source.height,
      0,
      offsetY,
      offsetX,
      square,
    );
    dreamContext.drawImage(
      pairedVideo,
      source.x + source.width - 8,
      source.y,
      8,
      source.height,
      offsetX + square,
      offsetY,
      width - offsetX - square,
      square,
    );
  } else {
    dreamContext.drawImage(
      pairedVideo,
      source.x,
      source.y,
      source.width,
      8,
      offsetX,
      0,
      square,
      offsetY,
    );
    dreamContext.drawImage(
      pairedVideo,
      source.x,
      source.y + source.height - 8,
      source.width,
      8,
      offsetX,
      offsetY + square,
      square,
      height - offsetY - square,
    );
  }

  dreamContext.drawImage(
    pairedVideo,
    source.x,
    source.y,
    source.width,
    source.height,
    offsetX,
    offsetY,
    square,
    square,
  );
}

function drawGameplayPanel() {
  const source = REALITY_PANEL;
  gameplayContext.imageSmoothingEnabled = false;
  gameplayContext.drawImage(
    pairedVideo,
    source.x,
    source.y,
    source.width,
    source.height,
    0,
    0,
    gameplayCanvas.width,
    gameplayCanvas.height,
  );
}

function updateForecastLabel() {
  const forecastStep = Math.floor(pairedVideo.currentTime * FPS) % FORECAST_STEPS + 1;
  dreamWindowLabel.textContent =
    `Step ${pad(forecastStep)} / 15 \u00b7 synchronized \u00b7 0.75\u00d7`;
}

function resizeCanvases() {
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  dreamCanvas.width = Math.max(1, Math.floor(dreamCanvas.clientWidth * pixelRatio));
  dreamCanvas.height = Math.max(1, Math.floor(dreamCanvas.clientHeight * pixelRatio));
  gameplayCanvas.width = Math.max(
    1,
    Math.floor(gameplayCanvas.clientWidth * pixelRatio),
  );
  gameplayCanvas.height = Math.max(
    1,
    Math.floor(gameplayCanvas.clientHeight * pixelRatio),
  );
}

function updateScrollProgress() {
  const maxScroll = Math.max(1, reel.offsetHeight - window.innerHeight);
  const progress = Math.min(1, Math.max(0, window.scrollY / maxScroll));
  stage.style.setProperty("--scroll-progress", progress.toFixed(3));
}

function pad(value) {
  return String(value).padStart(2, "0");
}
