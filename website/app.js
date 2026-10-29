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
  // Clipchamp fits the original 2:1 recording inside its exported canvas.
  const scale = Math.min(pairedVideo.videoWidth / 512, pairedVideo.videoHeight / 256);
  const offsetX = (pairedVideo.videoWidth - 512 * scale) / 2;
  const offsetY = (pairedVideo.videoHeight - 256 * scale) / 2;
  Object.assign(DREAM_PANEL, { x: offsetX, y: offsetY, width: 256 * scale, height: 256 * scale });
  Object.assign(REALITY_PANEL, { x: offsetX + 256 * scale, y: offsetY + 8 * scale, width: 256 * scale, height: 240 * scale });
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

  // Limit each stretched side band to 10% of the background width.
  const square = Math.max(Math.min(width, height), Math.round(width * 0.8));
  const offsetX = Math.floor((width - square) / 2);
  // Keep the lower edge, including the watermark, visible when zooming in.
  const offsetY = width >= height ? height - square : Math.floor((height - square) / 2);
  // The bottom 16 original pixels contain the burned-in name. Extend only
  // clean scenery into the margins; keep the watermark in the main square.
  const cleanHeight = source.height * 15 / 16;
  const cleanDestinationHeight = square * 15 / 16;
  const edgeWidth = source.width / 32;
  const cleanRowHeight = source.height / 256;

  if (width >= height) {
    const margins = [
      { sourceX: source.x, x: 0, width: offsetX },
      {
        sourceX: source.x + source.width - edgeWidth,
        x: offsetX + square,
        width: width - offsetX - square,
      },
    ];
    for (const margin of margins) {
      if (margin.width <= 0) continue;
      dreamContext.drawImage(
        pairedVideo,
        margin.sourceX, source.y, edgeWidth, cleanHeight,
        margin.x, offsetY, margin.width, cleanDestinationHeight,
      );
      dreamContext.drawImage(
        pairedVideo,
        margin.sourceX, source.y + cleanHeight - cleanRowHeight,
        edgeWidth, cleanRowHeight,
        margin.x, offsetY + cleanDestinationHeight,
        margin.width, square - cleanDestinationHeight,
      );
    }
  } else {
    dreamContext.drawImage(
      pairedVideo,
      source.x,
      source.y,
      source.width,
      source.width / 32,
      offsetX,
      0,
      square,
      offsetY,
    );
    dreamContext.drawImage(
      pairedVideo,
      source.x,
      source.y + cleanHeight - source.height / 32,
      source.width,
      source.width / 32,
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
